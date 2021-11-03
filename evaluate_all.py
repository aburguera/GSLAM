#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : evaluate_all
# Description : Simple script to evaluate the GraphOptimizer together with
#               a loop detector.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-August-2021 - Creation
#               9-September-2021 - Minor improvements
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
# Note        : Online plot may NOT work depending on your runtime. In this
#               case, just set PLOT_ONLINE=False
###############################################################################

###############################################################################
# SET GPU
###############################################################################

# These must be the first lines to execute. Restart the kernel before.
# If you don't have GPU/CUDA or this does not work, just set it to False or
# (preferrably) remove these two lines.
from util import set_gpu
set_gpu(True)

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from time import time
from util import progress_bar,compute_quality_metrics,evaluate_trajectory
from util import compute_absolute_trajectory_error
from transform2d import compose_trajectory
from datasimulator import DataSimulator
from graphoptimizer import GraphOptimizer
from loopmodel import LoopModel
from imagematcher import ImageMatcher

###############################################################################
# GLOBAL PARAMETERS
###############################################################################

# Path to dataset folder
PATH_DATASET='../../DATA/GRIDSMALL/'
#PATH_DATASET='../../DATA/RANDSMALL/'

# Path to trained Siamese Neural Network (loop detector) base name (file name
# without extension).
PATH_NNMODEL='../../DATA/MODELS/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_32_16_EPOCHS10_CATEGORICAL_G0_05_G05_2_ENCODERTRAINED_LABELSNOTINVERTED'

# Enable/disable components. Components are used in the following order:
# 1.- Loops are detected (NN or Simulated)
# 2.- First filtering is performed (RANSAC or none)
# 3.- Second filtering is performed (MSE or none)
# To select how these components operate, the following flags can be used:
# ENABLE_NN: Use NN to detect loops (True) or simulate the detection (False)
# ENABLE_RANSAC: Use RANSAC to filter loops (True) or not (False)
# ENABLE_REJECTION: Use the MSE loop rejection approach (True) or not (False)
# If a component is disabled, the output of the previous one is used as its
# input
ENABLE_NN=False
ENABLE_RANSAC=False
ENABLE_REJECTION=True

# How motion between loop closing images is estimated. Possible values are:
# 0: Simulated, 1: RANSAC
MOTION_ESTIMATOR=0

# Only images grabbed LOOP_MARGIN time steps before the current one are
# considered as loop candidates. Images are compared with the current one in
# LOOP_SEPARATION steps. The larger this value the faster the execution and
# the more the loops that can be missed.
LOOP_MARGIN=5
LOOP_SEPARATION=5

# Plot trajectory, loops and robot while accessing to the data. Illustrative
# but time consuming.
PLOT_ONLINE=False

# Image size and number of channels. Must coincide with the ones used by the
# Siamese Neural Network model.
IMG_WIDTH=64
IMG_HEIGHT=64
IMG_NCHAN=3

# Conversion factor from pixel motions to world motions (see dataset info)
PIX_TO_WORLD=np.array([10,10,1]).reshape((3,1))

# Data simulator noise parameters
DS_NOISES=[[0.625,3.14/(180*4)],[2.5,3.14/180],[5,2*3.14/180]]
DS_NOISELEVEL=0

###############################################################################
# CREATE AND INITIALIZE OBJECTS/DATA/...
###############################################################################

# Initialize time measurements
timeNN=0
timeRANSAC=0
timeFilter=0
timeOptimizer=0

# Create the data simulator object
dataSimulator=DataSimulator(PATH_DATASET,loadImages=True,minOverlap=1,motionSigma=DS_NOISES[DS_NOISELEVEL][0],angleSigma=DS_NOISES[DS_NOISELEVEL][1])

# Pick covariances from the dataSimulator
odoCovariance=loopCovariance=dataSimulator.odoCovariance

# Create the Siamese NN loop detector and load the parameters
loopModel=LoopModel()
loopModel.load(PATH_NNMODEL)

# Create the image mather
theMatcher=ImageMatcher()

# Get the first image and ID
preID,preImage=dataSimulator.get_image()

# Initialize images/image IDs history
allID=[preID]
allImages=[preImage]

# Initialize storage for loops. The loops information in theLoops is only used
# to compute quality metrics. The structure depends on the used components and
# is:
# Row 0: From ID
# Row 1: To ID
# Row 2: Do Match (Ground truth)
# Row 3: Match detected (NN or Simulated)
# Row 4: Match detected (NN+[RANSAC or none])
# Row 5: Match detected (NN+[RANSAC or nonr]+[Loop rejection or none])
theLoops=np.empty((6,0),dtype='int')

# Create the optimizer
theOptimizer=GraphOptimizer(initialID=preID,minLoops=5,doFilter=ENABLE_REJECTION)

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

    # Build the initial candidate set
    candidateIDs=allID[:-LOOP_MARGIN:LOOP_SEPARATION]

    # If there is data in the candidate set
    if len(candidateIDs)>0:

        # Build a batch of candidate images
        candidateImages=np.array(allImages[:-LOOP_MARGIN:LOOP_SEPARATION])

        if ENABLE_NN:
            # Repeat the current image to form a batch of the same size
            curImageRepeated=np.repeat(curImage.reshape((1,IMG_HEIGHT,IMG_WIDTH,IMG_NCHAN)),candidateImages.shape[0],axis=0)

            # Predict the loops using the neural network
            tStart=time()
            currentPredictions=np.argmax(loopModel.predict([curImageRepeated,candidateImages]),axis=1)
            tEnd=time()
            # Update NN time consumption
            timeNN+=(tEnd-tStart)

        # For each image in the initial candidate set
        for curCandidateIndex,curCandidateID in enumerate(candidateIDs):

            # Get the ground truth (for quality metrics) and the simulated 
            # current prediction (loop/not loop)
            curPrediction,matcherMotion,doMatch=dataSimulator.match_images(curCandidateID,curID,addNoise=True,simulateFailures=True)

            # If NN is used, get the NN prediction
            if ENABLE_NN:
                curPrediction=currentPredictions[curCandidateIndex]

            # Add loop information, ground truth and Neural Network prediction
            currentLoopStats=[curCandidateID,curID,doMatch,curPrediction,curPrediction,0]

            # If a loop is detected, apply RANSAC
            if curPrediction:
                # Define the images (either for RANSAC or least squares estimation)
                theMatcher.define_images(candidateImages[curCandidateIndex],curImage)

                # Filter loops using RANSAC if enabled
                if ENABLE_RANSAC or MOTION_ESTIMATOR==1:
                    tStart=time()
                    theMatcher.estimate()
                    tEnd=time()

                if ENABLE_RANSAC:
                    # Update the stored prediction (for quality metrics)
                    currentLoopStats[4]=int(not theMatcher.hasFailed)
                    # Update RANSAC time consumption
                    timeRANSAC+=(tEnd-tStart)

                # Compute the motion (least squares) if enabled
                if MOTION_ESTIMATOR==1:
                    matcherMotion=theMatcher.theMotion

                # Add the loop and motion if RANSAC enabled and suceeded or
                # always if RANSAC disabled.
                if not ENABLE_RANSAC or not theMatcher.hasFailed:
                    # Adapt shape
                    matcherMotion=matcherMotion.reshape((3,1))
                    # If motion has been estimated from images, convert units
                    # from pixel to world
                    if MOTION_ESTIMATOR==1:
                        matcherMotion=matcherMotion*PIX_TO_WORLD

                    tStart=time()
                    fromExceeded,toExceeded=theOptimizer.add_loop(curCandidateID,curID,matcherMotion,loopCovariance)
                    tEnd=time()

                    # Update filtering time
                    if ENABLE_REJECTION:
                        timeFilter+=(tEnd-tStart)

                    # If curID is repeated too many times, exit the loop (since within
                    # this loop all the loop attempts have the same curID).
                    # Comment it to obtain stats, uncomment it for real operation.
#                    if toExceeded:
#                        break

            # Update the stats
            theLoops=np.concatenate((theLoops,np.array(currentLoopStats).transpose().reshape((6,1))),axis=1)

        # Second loop rejection (if enabled) and graph optimization.
        tStart=time()
        selectedLoops=theOptimizer.validate()
        tEnd=time()

        # Update filtering time
        if ENABLE_REJECTION:
            timeFilter+=(tEnd-tStart)

        # Optimize if possible
        tStart=time()
        theOptimizer.optimize()
        tEnd=time()

        # Update optimization time
        timeOptimizer+=(tEnd-tStart)

        # Update the stats
        for curSelected in selectedLoops:
            for i in range(theLoops.shape[1]):
                if theLoops[0,i]==curSelected[0] and theLoops[1,i]==curSelected[1]:
                    theLoops[5,i]=1
                    break

    # Store the current ID and image for further loop detection.
    allID.append(curID)
    allImages.append(curImage)

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
# GET SOME STATS
###############################################################################

# Plot the final trajectory/loops/...
plt.figure()
plt.cla()
theOptimizer.plot(plotLoops=True,mainStep=1,secondaryStep=20)
plt.axis('equal')
plt.show()

# Compute the NN stats
nnTP=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[3,:]==1))
nnFP=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[3,:]==1))
nnTN=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[3,:]==0))
nnFN=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[3,:]==0))
nnAccuracy,nnPrecision,nnRecall,nnFallout=compute_quality_metrics(nnTP,nnFP,nnTN,nnFN)

# Compute the NN+RANSAC stats
ransacTP=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[4,:]==1))
ransacFP=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[4,:]==1))
ransacTN=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[4,:]==0))
ransacFN=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[4,:]==0))
ransacAccuracy,ransacPrecision,ransacRecall,ransacFallout=compute_quality_metrics(ransacTP,ransacFP,ransacTN,ransacFN)

# Compute the NN+RANSAC+filtering stats
filterTP=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[5,:]==1))
filterFP=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[5,:]==1))
filterTN=np.sum(np.logical_and(theLoops[2,:]==0,theLoops[5,:]==0))
filterFN=np.sum(np.logical_and(theLoops[2,:]==1,theLoops[5,:]==0))
filterAccuracy,filterPrecision,filterRecall,filterFallout=compute_quality_metrics(filterTP,filterFP,filterTN,filterFN)

# Print the results
print()
print('                NN          RANSAC      FILTER')
print('TP        : %8d -> %8d -> %8d'%(nnTP,ransacTP,filterTP))
print('FP        : %8d -> %8d -> %8d'%(nnFP,ransacFP,filterFP))
print('TN        : %8d -> %8d -> %8d'%(nnTN,ransacTN,filterTN))
print('FN        : %8d -> %8d -> %8d'%(nnFN,ransacFN,filterFN))

print('ACCURACY  : %8.3f -> %8.3f -> %8.3f'%(nnAccuracy,ransacAccuracy,filterAccuracy))
print('PRECISION : %8.3f -> %8.3f -> %8.3f'%(nnPrecision,ransacPrecision,filterPrecision))
print('RECALL    : %8.3f -> %8.3f -> %8.3f'%(nnRecall,ransacRecall,filterRecall))
print('FALLOUT   : %8.3f -> %8.3f -> %8.3f'%(nnFallout,ransacFallout,filterFallout))

# Print time consumption
print()
print('                TIME CONSUMPTION')
print('NN        : %8d s'%timeNN)
print('RANSAC    : %8d s'%timeRANSAC)
print('Filtering : %8d s'%timeFilter)
print('Optimizer : %8d s'%timeOptimizer)

###############################################################################
# TRAJECTORY EVALUATION
###############################################################################

# This comparison only evaluates the final trajectory considering NN+RANSAC+
# loop rejection.

# Get the trajectory and the ground truth in the proper format
theTrajectory=np.array([np.array(v.pose) for v in theOptimizer.theVertices]).transpose()
gtOdom=dataSimulator.gtOdom[0:3,1:]

# Get the error
print()
print('[EVALUATING TRAJECTORY]')
avgError=evaluate_trajectory(theTrajectory,gtOdom,True)
# Print results
print()
print(' * AVERAGE ERROR : %5.3f UNITS PER TRAVELLED UNIT'%avgError)
print('[EVALUATION DONE]')

# Compute absolute trajectory error (ATE)
print()
print('[COMPUTING ABSOLUTE TRAJECTORY ERROR]')
theErrors,meanError,stdError=compute_absolute_trajectory_error(theTrajectory,compose_trajectory(gtOdom))
# Plot the error
plt.figure()
plt.plot(theErrors)
plt.xlabel('Time step')
plt.ylabel('Absolute error')
plt.show()
# Print the results
print(' * MEAN OF TRAJECTORY ABSOLUTE ERROR %5.3f'%meanError)
print(' * STANDARD DEVIATION OF TRAJECTORY ABSOLUTE ERROR %5.3f'%stdError)
print('[ATE COMPUTED]')