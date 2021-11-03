#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : test_datasimulator
# Description : Simple script to demonstrate how to use DataSimulator
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-June-2021 - Creation
#               9-June-2021 - Loop matrix changed to contain indices instead
#                             of image IDs to make it more realistic.
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
# Note        : Online plot may NOT work depending on your runtime. In this
#               case, just set PLOT_ONLINE=False
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from transform2d import compose_references
from plotutil import draw_trajectory
from util import progress_bar
from datasimulator import DataSimulator
import matplotlib.pyplot as plt

###############################################################################
# GLOBAL PARAMETERS
###############################################################################

# Only images grabbed LOOP_MARGIN time steps before the current one are
# considered as loop candidates. Images are compared with the current one in
# LOOP_SEPARATION steps. The larger this value the faster the execution and
# the more the loops that can be missed.
# Please note that loop searching/closing in this example are just a placehol-
# der. They are not meant to reproduce an actual loop searching or an actual
# map optimization (there is no map optimization at all in this example).
LOOP_MARGIN=10
LOOP_SEPARATION=10

# Plot trajectory, loops and robot while accessing to the data. Illustrative
# but time consuming.
PLOT_ONLINE=False

###############################################################################
# CREATE AND INITIALIZE OBJECTS/DATA/...
###############################################################################

# Create the data simulator object with dedault parameters
dataSimulator=DataSimulator('../../DATA/GRIDLARGE/',minOverlap=30)

# Initialize X (current pose) and Xhist (all poses, used to plot later)
X=np.zeros((3,1))
Xhist=X

# Get the first image
preID,preImage=dataSimulator.get_image()

# Initialize storage for loops. Loops in this example are just stored and
# plotted, they are not used to optimize anything.
allID=[preID]
theLoops=np.empty((2,0),dtype='int')

# Create a figure if online plot is requested
if PLOT_ONLINE:
    plt.figure()

###############################################################################
# VISUAL ODOMETRY
###############################################################################

# While there is more data
while dataSimulator.update():

    # Get the current image
    curID,curImage=dataSimulator.get_image()

    # Match current and previous image, without noise nor failures. Try
    # other configurations to see the effects.
    _,theMotion,_=dataSimulator.match_images(preID,curID,addNoise=False,simulateFailures=False)

    # Loops are stored as pairs of indices referred to the stored trajectory.
    # The index of the current pose is len(allID) (or Xhist.shape[1])
    # instead of len(allID)-1 because the current pose/ID has not been yet
    # stored.
    curIndex=len(allID)

    # Check loops between the current image and the previous ones discarding
    # the LOOP_MARGIN most recent. Iterate in steps of LOOP_SEPARATION to
    # reduce computation time.
    for candidateIndex,candidateID in enumerate(allID[:-LOOP_MARGIN:LOOP_SEPARATION]):

        # Try to execute the code with simulateFailures=True and be amazed by
        # the power of the false positives!
        matchDetected,_,_=dataSimulator.match_images(candidateID,curID,addNoise=False,simulateFailures=False)
        if matchDetected:
            # If match is detected, store the loop. The candidateIndex is
            # multiplied times LOOP_SEPARATION since enumerate provides indices
            # of the partitioned array, not the whole one.
            theLoops=np.concatenate((theLoops,[[candidateIndex*LOOP_SEPARATION],[curIndex]]),axis=1)

    # Store the current ID (an image in a real implementation) for further
    # loop detection.
    allID.append(curID)

    # Update the current pose
    X=compose_references(X,theMotion)

    # Add the current pose to Xhist to plot it later.
    Xhist=np.concatenate((Xhist,X),axis=1)

    # Make the current image to be the previous in next iteration
    preID,preImage=curID,curImage

    # Plot if online plot is requested
    if PLOT_ONLINE:
        plt.cla()
        draw_trajectory(Xhist,theLoops=theLoops,secondaryStep=0)
        plt.axis('equal')
        plt.show()
        plt.pause(.05)
    # If no plot requested, print some progress
    else:
        progress_bar(dataSimulator.curStep-dataSimulator.startStep,dataSimulator.endStep-dataSimulator.startStep)

# After grabbing all the data, plot it
plt.figure()
draw_trajectory(Xhist,theLoops=theLoops)
plt.axis('equal')
plt.show()

###############################################################################
# GET SOME STATS
###############################################################################

# Initialize true and false positives and negatives
tp=fp=tn=fn=0

# Perform 10000 trials
for i in range(10000):

    # Select two random images in the dataset
    fromID,toID=np.random.randint(dataSimulator.endStep+1,size=2)

    # Simulate failures and simulate if they match. Pick ground truth informa-
    # tion (isCorrect) to update the stats.
    matchDetected,_,doMatch=dataSimulator.match_images(fromID,toID,addNoise=False,simulateFailures=True)

    # Update the stats.
    tn+=(not matchDetected and not doMatch)
    fn+=(not matchDetected and doMatch)
    fp+=(matchDetected and not doMatch)
    tp+=(matchDetected and doMatch)

# Print the processed stats
print('THEORETICAL TPR : %.2f'%dataSimulator.truePositiveRate)
print('THEORETICAL TNR : %.2f'%dataSimulator.trueNegativeRate)
print('EMPIRICAL   TPR : %.2f'%(tp/(fn+tp)))
print('EMPIRICAL   TNR : %.2f'%(tn/(tn+fp)))

# Print the raw, empirical, stats
print('EMPIRICAL TP=%d, FP=%d, TN=%d, FN=%d'%(tp,fp,tn,fn))

###############################################################################
# EXTREMELY IMPORTANT REMARK ABOUT THE DATA INTERPRETATION
###############################################################################

# Be VERY careful about the meaning of the true and false positive
# rates and, so, of the TP, FP, TN and FN used by the datasimulator. Even a
# very low false positive ratio leads to a large amount of false positives,
# even larger than the amount of true positives, if the number of non-loop
# comparisons largely exceeds the number of loop comparisons.

# Let's print it to remark it even it more

print('Be VERY careful about the meaning of the true and false positive')
print('rates and, so, of the TP, FP, TN and FN used by the datasimulator. Even a')
print('very low false positive ratio leads to a large amount of false positives,')
print('even larger than the amount of true positives, if the number of non-loop')
print('comparisons largely exceeds the number of loop comparisons.')