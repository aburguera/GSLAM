#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : util
# Description : General purpose utility functions
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 25-May-2021 - Creation
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from pickle import load,dump
import sys
from math import sqrt
from transform2d import compose_references,invert_reference

###############################################################################
# FUNCTIONS
###############################################################################

###############################################################################
# SET GPU ON/OFF
# MUST BE THE FIRST CALL. CANNOT BE CHANGED WHEN SET. TO CHANGE, RESTART KERNEL
###############################################################################
def set_gpu(useGPU=True):
    # Please note that this is a mix of cabala and magic (probably black magic)
    # from:
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    # This works on MY computer (Ubuntu 20 and CUDA Toolkit 10.1)
    if not useGPU:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import tensorflow as tf
        physicalDevices=tf.config.experimental.list_physical_devices('GPU')
        assert len(physicalDevices) > 0, "Not enough GPU hardware devices available"
        config=tf.config.experimental.set_memory_growth(physicalDevices[0], True)

###############################################################################
# LOAD CSV
# Alternate way to np.loadtxt to read large CSV files (only numbers)
# into nparray.
# Source: https://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy
# Input       : fileName  - The CSV filename to load
#               delimiter - The CSV column delimiter
#               skiprows  - Number of header rows to skip
#               dtype     - Data type
# Output      : data      - The CSV contents in nparray format.
###############################################################################
def loadcsv(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        loadcsv.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, loadcsv.rowlength))
    return data

###############################################################################
# LOAD A PYTHON OBJECT SAVED AS PICKLE BINARY FILE. WARNING: THE PICKLE BINARY
# FILE MIGHT NOT BE PORTABLE.
# Input       : fileName  - The pickle filename to load
# Output      : data      - The loaded object
###############################################################################
def load_pickle_data(fileName):
    with open(fileName,'rb') as inFile:
        return load(inFile)

###############################################################################
# SAVE A PYTHON OBJECT AS A PICKLE BINARY FILE. WARNING: THE PICKLE BINARY
# FILE MIGHT NOT BE PORTABLE.
# Input       : fileName  - The pickle filename to save
#               theData   - The data/object to save
###############################################################################
def save_pickle_data(fileName,theData):
    with open(fileName,'wb') as outFile:
        dump(theData,outFile)

###############################################################################
# SIMPLE TEXT PROGRESS BAR
# Input : curValue - Current value of the progress bar
#         maxValue - Maximum value of the progress bar
# Note  : The progress bar size is hardcoded to 50
###############################################################################
def progress_bar(curValue,maxValue):
    thePercentage=curValue/maxValue
    curSize=int(50*thePercentage)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*curSize, int(thePercentage*100)))
    sys.stdout.flush()

###############################################################################
# COMPUTE SOME BASIC STATS
# Input  : tp,fp,tn,fn : Number of true positives, false positives, true
#                        negatives and false negatives respectively.
# Output : theAccuracy, thePrecision, theRecall, theFallout : Some basic
#                        quality metrics.
###############################################################################
def compute_quality_metrics(tp,fp,tn,fn):
    theAccuracy=(tp+tn)/(tp+fp+tn+fn)
    thePrecision=tp/(tp+fp)
    theRecall=tp/(tp+fn)
    theFallout=fp/(fp+tn)
    return theAccuracy,thePrecision,theRecall,theFallout

###############################################################################
# EVALUATES A TRAJECTORY COMPARING WITH GROUND TRUTH ODOMETRY
# Input  : theTrajectory: 3xN matrix. Each column t is the ABSOLUTE pose
#                         to evaluate estimated at time step t. An absolute
#                         pose is expressed as x,y,o
#          gtOdom       : 3xN matrix. Each column t is the RELATIVE motion from
#                         time step t to time step t+1. A relative motion is
#                         expressed as DeltaX,DeltaY,DeltaO
#          showProgress : Show progress bar (False/True)
# Output : avgError     : Average error per traveled distance.
# Note   : The function computes the distance from the estimated motion i->j
#          to the ground truth motion i->j and divides it by the traveled
#          distance from i to j. It performs it for every possible pair i,j
#          without repetition (that is, if i->j is considered then j->i is not)
#          and outputs the average.
# Note   : The trajectory to evaluate must contain ABSOLUTE poses. The ground
#          truth must contain RELATIVE motions. This is to ease the computation
#          of the traveled distance.
# Note   : The orientation is not taken into account.
###############################################################################
def evaluate_trajectory(theTrajectory,gtOdom,showProgress=False):
    # Precompute some values
    nItems=gtOdom.shape[1]
    allErrors=[]

    # For every pose i (except the last) in the trajectory
    for i in range(nItems-1):
        # Distance from time i to time i+1
        curDistance=sqrt(gtOdom[0,i]**2+gtOdom[1,i]**2)
        # Motion from time i to time i+1
        gtPose=gtOdom[:,i]
        # For every pose j from the next one to the end
        for j in range(i+1,nItems):
            # Compuute estimated motion from i to j
            estPose=compose_references(invert_reference(theTrajectory[:,i]),theTrajectory[:,j])
            # The error is the distance from the true motion i->j to the estimated
            # motion i->j divided by the distance of this motion. Thus, we have
            # error per traveled distance.
            allErrors.append((sqrt((estPose[0]-gtPose[0])**2+(estPose[1]-gtPose[1])**2))/curDistance)
            # Update the distance (by adding the motion from j to j+1)
            curDistance+=sqrt(gtOdom[0,j]**2+gtOdom[1,j]**2)
            # Compute the new ground truth pose (by composing with the motion from
            # j to j+1)
            gtPose=compose_references(gtPose,gtOdom[:,j])
        if showProgress:
            progress_bar(i,nItems-2)

    # Average the errors.
    avgError=np.mean(allErrors)
    return avgError

###############################################################################
# COMPUTES THE ABSOLUTE TRAJECTORY ERROR AS DESCRIBED IN RAWSEEDS
# Input  : X          : The absolute trajectory to evaluate. 3xN nparray. Each
#                       column is a pose [x,y,o]
#          Xgt        : The absolute ground truth poses. Same format as X
#          indexAlign : X is compared from indesAlign onward. Use this to
#                       align vector positions (i.e. timestamps).
#          Xalign     : Transformation to apply to each estimated pose in X
#                       to properly align it with the ground truth.
# Output : theErrors  : The ATE
#          meanError  : Mean ATE
#          stdError   : Standard deviation of ATE
###############################################################################
def compute_absolute_trajectory_error(X,Xgt,indexAlign=1,Xalign=[0,0,0]):
    theErrors=[]
    # Loop through all the poses
    for i in range(indexAlign,X.shape[1]):
        # Align the pose
        alignedPose=compose_references(Xalign,X[:,i]).flatten()
        # Get the ground truth
        gtPose=Xgt[:,i-indexAlign]
        # Compute the translation error
        theDifference=alignedPose[:2]-gtPose[:2]
        theError=np.sqrt(np.sum(theDifference**2))
        # Store the error
        theErrors.append(theError)
    # Return the errors and their mean and standard deviation
    return theErrors,np.mean(theErrors),np.std(theErrors)