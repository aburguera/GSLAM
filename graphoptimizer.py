#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : GraphOptimizer
# Description : Graph SLAM front end aimed at reducing the amount of false
#               positives. The back-end (graph optimizer) is:
#               https://python-graphslam.readthedocs.io/en/stable/
#               It is advisable to remove/comment the progress printing in
#               the above mentioned GraphSLAM implementation to avoid too
#               much verbosity.
#               Please, check test_graphoptimized for an usage example.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 3-August-2021 - Creation
#               9-September-2021 - Added the doFilter option
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from numpy.linalg import inv
from transform2d import compose_references,invert_reference,normalize_angle
from transform2d import least_squares_cartesian,compose_point
from math import sqrt
from plotutil import draw_trajectory
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.pose.se2 import PoseSE2
from graphslam.edge.edge_odometry import EdgeOdometry

###############################################################################
# GRAPH OPTIMIZER FRONT END CLASS
###############################################################################

class GraphOptimizer():

    ###########################################################################
    # CONSTRUCTOR
    # Copies the parameters in the object.
    # Input  : initialPose - The absolute reference frame of the trajectory.
    #          initialID   - The initialPose is stored as a graph vertex. This
    #                        is the ID assigned to that vertex.
    #          minLoops    - Minimum number of loops that pass the odometric
    #                        checks and the non-repetition check to start the
    #                        MSE filter.
    #          maxAngularDistance - Maximum angular difference between a loop
    #                        estimate and the estimate according to the graph
    #                        to consider the loop a possible candidate.
    #          maxMotionDistance - Same as before but with motion distance.
    #          errorThreshold - The threshold to accept/reject with the MSE
    #                           filter.
    #          maxSameID   - Maximum number of repeated IDs within a loop can-
    #                        didate set.
    #          doFilter    - True: Normal operation. False: No loop rejection
    # Note   : Note that by setting doFilter=False this class is just an
    #          almost useless wrapper for graphslam (see imports). Setting
    #          it to False is only intended for experimentation/testing/
    #          comparison purposes. In real operation it should be set to
    #          True or, maybe better, the doFilter option be removed.
    ###########################################################################
    def __init__(self,initialPose=np.zeros((3,1)),initialID=-1,minLoops=5,maxAngularDistance=3.14/4,maxMotionDistance=500,errorThreshold=500,maxSameID=2,doFilter=True):
        self.theEdges=[]
        self.isLoop=[]
        self.candidateEdges=[]
        self.theVertices=[Vertex(initialID,PoseSE2(initialPose[:2,0],initialPose[2,0]))]
        self.minLoops=minLoops
        self.maxAngularDistance=maxAngularDistance
        self.maxMotionDistance=maxMotionDistance
        self.errorThreshold=errorThreshold
        self.previousToID=-2
        self.maxSameID=maxSameID
        self.readyToOptimize=False
        self.doFilter=doFilter

    ###########################################################################
    # ADDS AN ODOMETRIC ESTIMATE
    # Input  : newPoseID - The ID of the new vertex.
    #          Xodo      - The relative motion according to odometry between
    #                      the most recent vertex and the one to be added.
    #          Podo      - Covariance of the odometric motion.
    ###########################################################################
    def add_odometry(self,newPoseID,Xodo,Podo):
        # Get the most recent pose and ID.
        lastPose=np.array(self.theVertices[-1].pose)
        lastID=self.theVertices[-1].id
        # Compute the new pose
        newPose=compose_references(lastPose,Xodo)
        # Add it as a vertex
        self.theVertices.append(Vertex(newPoseID,PoseSE2(newPose[:2,0],newPose[2,0])))
        # Add the odometric estimate as an edge
        self.theEdges.append(EdgeOdometry([lastID,newPoseID],inv(Podo),PoseSE2(Xodo[:2,0],Xodo[2,0])))
        # Remember this edge is not a loop (only for plotting purposes)
        self.isLoop.append(False)

    ###########################################################################
    # ADDS A LOOP
    # Adds a loop into the candidate set if the number of candidate loops with
    # the provided fromID or toID does not exceed maxSameID and if the
    # loop passes the odometric check.
    # Input  : fromID - The ID of the loop start vertex.
    #          toID   - The ID of the loop end vertex.
    #          Xloop  - The motion from the start to the end vertices.
    #          Ploop  - The covariance of the motion from start to end.
    # Output : fromExceeded - The number of candidate loops with the same
    #                         source vertex ID surpasses maxSameID.
    #          toExceeded   - Same as before, but with the destination ID.
    # Note   : from/toExceeded can be used (if necessary) to break an outer
    #          loop and speed up computation.
    ###########################################################################
    def add_loop(self,fromID,toID,Xloop,Ploop):
        # If no filtering, simply add the loop without any check and return
        # False for both fromExceeded and toExceeded
        if not self.doFilter:
            self.candidateEdges.append(EdgeOdometry([fromID,toID],inv(Ploop),PoseSE2(Xloop[:2,0],Xloop[2,0])))
            return False,False

        # Check if too many loops with the same from/to exist
        fromExceeded=sum(1 for e in self.candidateEdges if e.vertex_ids[0]==fromID)>=self.maxSameID
        toExceeded=sum(1 for e in self.candidateEdges if e.vertex_ids[1]==toID)>=self.maxSameID
        # If not too many loops, try to include the loop
        if not (fromExceeded or toExceeded):
            # Get the poses in the stored vertex list
            fromPose=np.array(self._get_vertex_by_id(fromID).pose)
            toPose=np.array(self._get_vertex_by_id(toID).pose)
            # Check heuristic odometric consistency
            if GraphOptimizer.odometric_check(Xloop,fromPose,toPose,self.maxAngularDistance,self.maxMotionDistance):
                # A loop only involves an adge, since it does not create new nodes.
                # This edge is added to the candidate set, since each loop has to be
                # confirmed before optimization and/or inclusion in the edge list.
                self.candidateEdges.append(EdgeOdometry([fromID,toID],inv(Ploop),PoseSE2(Xloop[:2,0],Xloop[2,0])))
        # Return what is exceeded (if something)
        return fromExceeded,toExceeded

    ###########################################################################
    # VALIDATES THE CANDIDATE LOOPS
    # First, it checks if the candidate set has, at least, minLoops candidates.
    # It it has, then the loops are validated using the MSE filter. The
    # remaining loops, if any, should be used to optimize the graph using
    # GraphSLAM (see comments in this file's header).
    # Output : selectedFromTo : List of from-to IDs of the loops that have
    #          been included to optimize the map (or None if any). This is
    #          only necessary to compute stats. It can be removed when in
    #          production.
    ###########################################################################
    def validate(self):
        # If no filtering is to be done, check if there are enough loops and
        # optimize.
        # This part of the code has some copy/paste from the doFilter==True
        # part next. This is to easily remove the doFilter==False case if
        # not necessary.
        if not self.doFilter:
            selectedFromTo=[]
            numLoops=len(self.candidateEdges)
            if numLoops>self.minLoops:
                # Add the edges to the list of accepted edges
                selectedFromTo=[[e.vertex_ids[0],e.vertex_ids[1]] for e in self.candidateEdges]
                self.theEdges=self.theEdges+self.candidateEdges

                # Remember that these edges come from a loop (basically for
                # plotting purposes)
                self.isLoop+=([True]*len(self.candidateEdges))

                # Remember that the graph should be optimized
                self.readyToOptimize=True
                self.candidateEdges=[]
            return selectedFromTo

        # If there are not enough loops, do nothing
        selectedFromTo=[]
        numLoops=len(self.candidateEdges)
        if numLoops>self.minLoops:

            # Prepare some data storage
            XWS=np.zeros((3,numLoops))
            XWD=np.zeros((3,numLoops))
            XSD=np.zeros((3,numLoops))

            # Build the source, destination and loop data structures
            for iEdge in range(numLoops):
                curEdge=self.candidateEdges[iEdge]

                # Get the from and to IDs
                fromID=curEdge.vertex_ids[0]
                toID=curEdge.vertex_ids[1]

                # Get the from and to poses from the vertices
                fromPose=np.array(self._get_vertex_by_id(fromID).pose)
                toPose=np.array(self._get_vertex_by_id(toID).pose)

                # Store them
                XWS[:,iEdge]=fromPose
                XWD[:,iEdge]=toPose

                # Store the current loop estimate
                XSD[:,iEdge]=curEdge.estimate

            # Validate the loops
            loopIndices=GraphOptimizer.validate_loops(XWS,XWD,XSD,self.errorThreshold)

            # If some loops remain after validation
            if not loopIndices is None:
                # Add the edges to the list of accepted edges
                selectedEdges=[self.candidateEdges[i] for i in loopIndices]
                selectedFromTo=[[e.vertex_ids[0],e.vertex_ids[1]] for e in selectedEdges]
                self.theEdges=self.theEdges+selectedEdges

                # Remember that these edges come from a loop (basically for
                # plotting purposes)
                self.isLoop+=([True]*len(selectedEdges))

                # Remember that the graph is ready to be optimized
                self.readyToOptimize=True

            self.candidateEdges=[]
        return selectedFromTo

    ###########################################################################
    # OPTIMIZES THE GRAPH IF READY
    # If validate decided that the graph can be optimized, do it.
    # Note : It is separated from validate (instead of being called autmatica-
    #        lly from validate) to easy externally measuring the time consump-
    #        tion.
    ###########################################################################
    def optimize(self):
        if self.readyToOptimize:
            g=Graph(self.theEdges,self.theVertices)
            g.optimize()
            self.readyToOptimize=False

    ###########################################################################
    # PLOTS THE GRAPH
    # Input  : plotLoops - True/False - Plot lines between the vertices invol-
    #                      ved in loops.
    #          remaining parameters - See draw_trajectory in plotutil
    ###########################################################################
    def plot(self,plotLoops=True,mainStep=1,secondaryStep=5,drawLast=True):
        # Build the trajectory to plot. Store the vertex ID.
        Xhist=np.zeros((4,len(self.theVertices)))
        for i in range(Xhist.shape[1]):
            Xhist[0:3,i]=self.theVertices[i].pose
            Xhist[3,i]=self.theVertices[i].id

        theLoops=None
        if plotLoops:
            # Get the indices within the trajectory corresponding to the IDs in
            # the edges that belong to loops. This can be slow, since for each
            # loop edge it has to search within the trajectory. It can be sped
            # up by assuming that indices and IDs are the same (if they are,
            # like in this particular case).
            theLoops=np.array([[np.where(Xhist[3,:]==self.theEdges[i].vertex_ids[0])[0][0],np.where(Xhist[3,:]==self.theEdges[i].vertex_ids[1])[0][0]] for i in range(len(self.theEdges)) if self.isLoop[i]]).transpose()
            if theLoops.shape[0]==0:
                theLoops=None

        # Draw it
        draw_trajectory(Xhist,theLoops,mainStep,secondaryStep,drawLast)

    ###########################################################################
    # GIVEN A VERTEX ID, OUTPUTS THE VERTEX DATA STRUCTURE
    # Input  : theID - Vertex ID to search.
    # Output : v     - The Vertex data structure.
    ###########################################################################
    def _get_vertex_by_id(self,theID):
        # Search the vertex with the specified ID. When found, return it thus
        # stopping the search (that is, IDs are assumed to be unique)
        for v in self.theVertices:
            if v.id==theID:
                return v
        # If not found, return None
        return None

###############################################################################
# IMMEDIATE METHODS (TO TEST OTHER APPROACHES)
###############################################################################

    ###########################################################################
    # ADDS A LOOP INTO THE MAP WITHOUT ANY FILTERING
    # Input  : fromID - The ID of the loop start vertex.
    #          toID   - The ID of the loop end vertex.
    #          Xloop  - The motion from the start to the end vertices.
    #          Ploop  - The covariance of the motion from start to end.
    ###########################################################################
    def add_loop_immediate(self,fromID,toID,Xloop,Ploop):
        self.theEdges.append(EdgeOdometry([fromID,toID],inv(Ploop),PoseSE2(Xloop[:2,0],Xloop[2,0])))

    ###########################################################################
    # OPTIMIZES THE GRAPH WITHOUT ANY CHECK
    ###########################################################################
    def optimize_immediate(self):
        # Optimize the graph (this also optimizes the edges in this
        # object).
        g=Graph(self.theEdges,self.theVertices)
        g.optimize()

###############################################################################
# STATIC METHODS
###############################################################################

    ###########################################################################
    # GETS ALL THE COMBINATIONS OF ITEMS FROM 0 TO NUMITEMS-1 WITH
    # AT LEAST MINITEMS
    # Input  : numItems - Items to combine range from 0 to numItems-1
    #          minItems - The combinations must have at least minItems items.
    # Output : theCombinations - List of combinations.
    #          theLengths - Numpy array with the lengths of the combinations.
    # TODO: Optimize the algorithm.
    ###########################################################################
    @staticmethod
    def get_combinations(numItems,minItems):
        theCombinations=[]
        theLengths=[]
        for i in range(1,2**numItems):
            curComb=[i for i,v in enumerate(bin(i)[2:].zfill(numItems)) if v=='1']
            curLen=len(curComb)
            if curLen>=minItems:
                theCombinations.append(curComb)
                theLengths.append(curLen)
        return theCombinations,np.array(theLengths)

    ###########################################################################
    # PERFORMS THE ODOMETRIC TEST: CHECKS IF A LOOP IS (HEURISTICALLY) CONSIS-
    # TENT WITH THE SPECIFIED FROM AND TO POSES.
    # Input  : Xloop - Loop motion [x,y,o] to check
    #          fromPose - Absolute pose of the loop start point (according, for
    #                     example, to current graph estimate)
    #          toPose   - Absolute pose of the loop end point (according, for
    #                     example, to current graph estimate)
    #          maxAngularDistance - Maximum difference between the orientation
    #                     in Xloop and the relative orientation from fromPose
    #                     to toPose. If the difference is larger, the loop
    #                     does not pass the test.
    #          maxMotionDistance - Same as previous but with motion distance.
    # Output : testPassed - The loop can be a candidate? (True/False)
    ###########################################################################
    @staticmethod
    def odometric_check(Xloop,fromPose,toPose,maxAngularDistance,maxMotionDistance):
        # Compute the motion according to fromPose and toPose
        Xodo=compose_references(invert_reference(fromPose),toPose)

        # Compute the Euclidean distance between the fromPose-toPose based motion
        # and the loop estimate.
        theDistance=Xodo-Xloop
        theDistance=sqrt(theDistance[0,0]**2+theDistance[1,0]**2)

        # If the differences between the loop and the fromPose-toPose based motion
        # are too large, return False. Otherwise, return True.
        return (abs(normalize_angle(Xodo[2,0]-Xloop[2,0]))<=maxAngularDistance) and (theDistance<=maxMotionDistance)

    ###########################################################################
    # VALIDATES A SET OF LOOPS USING THE MSE TEST.
    # Input  : XWS            - 3xN nparray. Each column is the pose of a
    #                           source vertex involved in a loop.
    #          XWD            - 3xN nparray. Each column is the pose of a
    #                           destination vertex involved in a loop.
    #          XSD            - 3xN nparray. Motion, according to the loop,
    #                           from source to destination vertices.
    #          errorThreshold - The maximum allowable MSE error to accept the
    #                           loop combination.
    # Output : theCombination - Combination of loops that passes the test.
    # Note   : XWS[:,i], XWD[:,i] and XSD[:,i] refer to the same loop.
    ###########################################################################
    @staticmethod
    def validate_loops(XWS,XWD,XSD,errorThreshold):
        # Pre-compute some stuff
        numLoops=XWS.shape[1]

        # Compute the centers of masses
        XWA=np.mean(XWS,axis=1)
        XWB=np.mean(XWD,axis=1)
        XWA[2]=normalize_angle(XWA[2])
        XWB[2]=normalize_angle(XWB[2])

        # Pre-compute some stuff
        XAW=invert_reference(XWA)
        XBW=invert_reference(XWB)

        # Express XWS with respect to XWA and XWD with respect to XWB
        XAS=np.zeros((3,numLoops))
        XBD=np.zeros((3,numLoops))
        for i in range(numLoops):
             XAS[:,i]=compose_references(XAW,XWS[:,i]).reshape((3,))
             XBD[:,i]=compose_references(XBW,XWD[:,i]).reshape((3,))

        # Compute the points as they are to be used by the ICP optimizer
        XAD=np.zeros((2,numLoops))
        XBD=XBD[0:2,:]
        for i in range(numLoops):
            curPoint=compose_references(XAS[:,i],XSD[:,i])
            XAD[:,i]=curPoint[0:2,0]

        # Get the possible loop combinations with at least the half of the
        # existing loops.
        theCombinations,allSizes=GraphOptimizer.get_combinations(numLoops,1);

        # Evaluate every possible combination
        allErrors=np.empty(len(theCombinations))
        for i in range(len(theCombinations)):
            # The currently selected loops are the ones in combinations plus the
            # XAB constraint
            selectedLoops=theCombinations[i]

            # Do least squares
            XABnew=least_squares_cartesian(XAD[:,selectedLoops], XBD[:,selectedLoops])

            # Compute the resulting error
            XADnew=compose_point(XABnew,XBD[:,selectedLoops])
            theError=XADnew-XAD[:,selectedLoops]
            theError=theError*theError
            theError=np.sum(theError)/allSizes[i]
            allErrors[i]=theError

        involvedSizes=allSizes[allErrors<=errorThreshold]
        if involvedSizes.shape[0]==0:
            return None
        else:
            return theCombinations[np.argmax(involvedSizes)]