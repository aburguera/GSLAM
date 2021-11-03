#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : evaluate_graphoptimizer
# Description : Simple script to evaluate the GraphOptimizer
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 2-August-2021 - Creation
#               9-September-2021 - Minor improvements
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
# Note        : This file is just to provide a simple, separate, script to
#               evaluate some aspects of GraphOptimizer. The same results
#               can be obtained using evaluate_all with the following
#               parameters:
#               APPLY_NN=False
#               APPLY_RANSAC=False
#               APPLY_REJECTION=True
#               MOTION_ESTIMATOR=0
# Note        : Online plot may NOT work depending on your runtime. In this
#               case, just set PLOT_ONLINE=False
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from util import progress_bar,compute_quality_metrics
from datasimulator import DataSimulator
from graphoptimizer import GraphOptimizer

###############################################################################
# GLOBAL PARAMETERS
###############################################################################

# Only images grabbed LOOP_MARGIN time steps before the current one are
# considered as loop candidates. Images are compared with the current one in
# LOOP_SEPARATION steps. The larger this value the faster the execution and
# the more the loops that can be missed.
LOOP_MARGIN=5
LOOP_SEPARATION=5

# Plot trajectory, loops and robot while accessing to the data. Illustrative
# but time consuming.
PLOT_ONLINE=False

###############################################################################
# CREATE AND INITIALIZE OBJECTS/DATA/...
###############################################################################

# Create the data simulator object
dataSimulator=DataSimulator('../../DATA/GRIDSMALL/',minOverlap=1)

# Pick covariances from the dataSimulator
odoCovariance=loopCovariance=dataSimulator.odoCovariance

# Get the first image
preID,preImage=dataSimulator.get_image()

# Initialize storage of images/image IDs.
allID=[preID]

# Initialize storage for loops. The loops information in theLoops is only used
# to plot them and to compute quality metrics. The structure is:
# Row 0: From ID
# Row 1: To ID
# Row 2: Do Match (Ground truth)
# Row 3: Match detected (unfiltered, with simulated failures)
# Row 4: Match detected (filtered)
theLoops=np.empty((5,0),dtype='int')

# Create the optimizer
theOptimizer=GraphOptimizer(initialID=preID,minLoops=5)

# Create a figure for online plot
if PLOT_ONLINE:
    plt.figure()

###############################################################################
# SLAM
###############################################################################

# While there is more data
while dataSimulator.update():

    # Get the current image
    curID,curImage=dataSimulator.get_image()

    # Match current and previous image. Since this is odometry, noise is added
    # but no failures are assumed to exist.
    _,theMotion,_=dataSimulator.match_images(preID,curID,addNoise=True,simulateFailures=False)

    # Include the odometric estimate into the graph
    theOptimizer.add_odometry(curID,theMotion.reshape((3,1)),odoCovariance)

    # Check loops between the current image and the previous ones discarding
    # the LOOP_MARGIN most recent. Iterate in steps of LOOP_SEPARATION
    # to reduce computation time.
    for candidateID in allID[:-LOOP_MARGIN:LOOP_SEPARATION]:

        # Match the two images. doMatch is picked only for evaluation. Since
        # this is loop detection, both noise and failures are simulated.
        matchDetected,loopMotion,doMatch=dataSimulator.match_images(candidateID,curID,addNoise=True,simulateFailures=True)

        # Add the loop information. The filtered information is set to 0 so
        # that those accepted are set to 1 after optimization.
        theLoops=np.concatenate((theLoops,[[candidateID],[curID],[doMatch],[matchDetected],[0]]),axis=1)

        # If is a match, add it to the graph
        if matchDetected:
            fromExceeded,toExceeded=theOptimizer.add_loop(candidateID,curID,loopMotion.reshape((3,1)),loopCovariance)
            # If curID is repeated too many times, exit the loop (since within
            # this loop all the loop attempts have the same curID).
            #if toExceeded:
            #    break

    # Optimize
    selectedLoops=theOptimizer.validate()
    theOptimizer.optimize()

    # Update the stats
    for curSelected in selectedLoops:
        for i in range(theLoops.shape[1]):
            if theLoops[0,i]==curSelected[0] and theLoops[1,i]==curSelected[1]:
                theLoops[4,i]=1
                break

    # Store the current ID (an image in a real implementation) for further
    # loop detection.
    allID.append(curID)

    # Make the current image to be the previous in next iteration
    preID,preImage=curID,curImage

    # Plot if online plot is requested
    if PLOT_ONLINE:
        plt.cla()
        theOptimizer.plot(plotLoops=True,mainStep=10,secondaryStep=0)
        plt.axis('equal')
        plt.show()
        plt.pause(.05)

    # Show the progress bar
    progress_bar(dataSimulator.curStep-dataSimulator.startStep,dataSimulator.endStep-dataSimulator.startStep)

###############################################################################
# SHOW RESULTS
###############################################################################

# If not online plot requested, plot at the end
if not PLOT_ONLINE:
        plt.cla()
        theOptimizer.plot(plotLoops=True,mainStep=1,secondaryStep=20)
        plt.axis('equal')
        plt.show()

# Compute the stats before filtering
preTP=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[3,:]==1))
preFP=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[3,:]==1))
preTN=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[3,:]==0))
preFN=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[3,:]==0))
preAccuracy,prePrecision,preRecall,preFallout=compute_quality_metrics(preTP,preFP,preTN,preFN)

# Compute the stats after filtering
postTP=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[4,:]==1))
postFP=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[4,:]==1))
postTN=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[4,:]==0))
postFN=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[4,:]==0))
postAccuracy,postPrecision,postRecall,postFallout=compute_quality_metrics(postTP,postFP,postTN,postFN)

# Print the results
print('TP        : %8d -> %8d'%(preTP,postTP))
print('FP        : %8d -> %8d'%(preFP,postFP))
print('TN        : %8d -> %8d'%(preTN,postTN))
print('FN        : %8d -> %8d'%(preFN,postFN))

print('ACCURACY  : %8.3f -> %8.3f'%(preAccuracy,postAccuracy))
print('PRECISION : %8.3f -> %8.3f'%(prePrecision,postPrecision))
print('RECALL    : %8.3f -> %8.3f'%(preRecall,postRecall))
print('FALLOUT   : %8.3f -> %8.3f'%(preFallout,postFallout))