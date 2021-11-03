#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : DataSimulator
# Description : Simple camera and image matching simulation. Uses the ground
#               truth of an UCAMGEN dataset and the user provided hit and
#               failure ratios and noise levels to simulate image matching.
#               Please check test_datasimulator.py for a usage example.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-June-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import os
import numpy as np
from util import load_pickle_data
from skimage.io import imread
from scipy.sparse import find
from skimage import img_as_float

###############################################################################
# DATA SIMULATOR CLASS
###############################################################################

class DataSimulator():

    ###########################################################################
    # CLASS MEMBERS
    ###########################################################################

    # Pre-processed UCAMGEN file names.
    FNAME_OVERLAP='OVERLAP.pkl'     # Overlap matrix
    FNAME_RELPOSX='IMGRPOSX.pkl'    # Relative motions in X
    FNAME_RELPOSY='IMGRPOSY.pkl'    # Relative motions in Y
    FNAME_RELPOSO='IMGRPOSO.pkl'    # Relative motions in O
    FNAME_ODOM='ODOM.pkl'            # Ground truth odometry

    ###########################################################################
    # CONSTRUCTOR
    # Loads the pre-processed UCAMGEN dataset. Go to the UCAMGEN repository for
    # info about these datasets. Use preprocess_dataset.py, provided in this
    # repository, to preprocess the data or to learn how to preprocess it.
    # Input  : dataPath    - Path to the pre-processed UCAMGEN dataset.
    #          loadImages  - True/False - Load the images from disk when
    #                        calling the update method or not.
    #          minOverlap  - The minimum percentage of overlap between images
    #                        according to the ground truth to be considered
    #                        as candidate true positive match when compared.
    #                        Note that "candidate" is used here since simulated
    #                        data corruption can change its status.
    #          motionSigma - Standard deviation of the random Gaussian noise
    #                        introduced in the simulated matching process in
    #                        the X and Y axes. Note that the units depends on
    #                        those of the UCAMGEN dataset in use.
    #          angleSigma  - Same as motionSigma but for the angular noise.
    #                        Units are radians.
    #          TN,FP,FN,TP - True and false positives and negatives to be
    #                        simulated. They are normalized internally.
    # Note   : motionSigma and angleSigma are only applied if addNoise==True
    #          when calling match_images.
    #          TN,FP,FN,TP are only applied if simulateFailures==True when
    #          calling match_images.
    ###########################################################################
    def __init__(self,dataPath,loadImages=False,minOverlap=50,motionSigma=5,angleSigma=2*3.14/180,
                 TN=.99,FP=.01,FN=0.01,TP=.99):

        # Initialize random seed
        np.random.seed(0)

        # Store input parameters.
        self.dataPath=dataPath
        self.loadImages=loadImages
        self.minOverlap=minOverlap
        self.odoCovariance=np.array([[motionSigma**2,0,0],[0,motionSigma**2,0],[0,0,angleSigma**2]])

        # Compute and store data to simulate errors.
        self.trueNegativeRate=TN/(TN+FP)
        self.truePositiveRate=TP/(FN+TP)

        # Load the dataset files
        self.theOverlap=load_pickle_data(os.path.join(dataPath,DataSimulator.FNAME_OVERLAP))
        self.relPosX=load_pickle_data(os.path.join(dataPath,DataSimulator.FNAME_RELPOSX))
        self.relPosY=load_pickle_data(os.path.join(dataPath,DataSimulator.FNAME_RELPOSY))
        self.relPosO=load_pickle_data(os.path.join(dataPath,DataSimulator.FNAME_RELPOSO))

        # Load the ground truth
        self.gtOdom=load_pickle_data(os.path.join(dataPath,DataSimulator.FNAME_ODOM))

        # Get the min and max relative pose values to simulate motions in case
        # of false positives.
        (_,_,z)=find(self.relPosX)
        self.xMin=np.min(z)
        self.xMax=np.max(z)
        (_,_,z)=find(self.relPosY)
        self.yMin=np.min(z)
        self.yMax=np.max(z)
        (_,_,z)=find(self.relPosO)
        self.oMin=np.min(z)
        self.oMax=np.max(z)

        # Initialize internal state
        self.startStep=0
        self.endStep=self.theOverlap.shape[0]-1
        self.curStep=self.startStep
        self.curImageID=None
        self.curImage=None

        # Perform the first step
        self.update()

    ###########################################################################
    # SET SIMULATION START AND END STEPS
    # Input : startStep - New start step or None. If None, the start step is
    #                     not modified.
    #         endStep   - New end step or None. If None, the end step is not
    #                     modified.
    # Note  : endStep<=startStep raises an error. It start or end are outside
    #         boundaries, they are corrected.
    # Note  : curStep is set to the new startStep if defined.
    ###########################################################################
    def set_start_end(self,startStep=None,endStep=None):
        if not startStep is None:
            self.startStep=max(0,startStep)
            self.curStep=self.startStep
        if not endStep is None:
            self.endStep=min(endStep,self.theOverlap.shape[0]-1)
        assert self.startStep<self.endStep

    ###########################################################################
    # UPDATES THE SIMULATION ONE STEP
    # Loads image if requested, advances step counter and checks if end reached
    # Output : True - Another step is possible.
    #          False - This was the last step.
    # Note   : Updates self.curImage with the image or None depending on
    #          loadImage.
    #          Uses curStep as the Image identifier (self.curImageID)
    #          The image filename is constructed with curStep+1 because
    #          the UCAMGEN image numbering starts at 1.
    ###########################################################################
    def update(self):
        self.curImage=None
        if self.loadImages:
            fileName=os.path.join(self.dataPath,'IMAGES','IMAGE%05d.png'%(self.curStep+1))
            self.curImage=img_as_float(imread(fileName))
        self.curImageID=self.curStep
        self.curStep+=1
        return self.curStep<=self.endStep

    ###########################################################################
    # RETURNS CURRENT IMAGE ID AND THE IMAGE (IF AVAILABLE)
    # Gets the image id and the image if available.
    # Output : curImageID - The image identifier
    #          curImage   - The image, or None if loadImages is False
    # Note   : The function will return the same info between updates.
    ###########################################################################
    def get_image(self):
        return self.curImageID,self.curImage

    ###########################################################################
    # SIMULATES IMAGE MATCHING USING THE GROUND TRUTH
    # Input  : firstID, secondID - ID of the images to match.
    #          addNoise - If True, random Gaussian noise is added to the
    #                     true estimate (in case of True Positives). The noise
    #                     mean is [0,0,0] and the standard deviation is the
    #                     one defined when creating the object.
    #          simulateFailures - If True, the system simulates false positi-
    #                     ves and negatives. The probabilities of these
    #                     failures depend on the number of TP, FP, TN and FN
    #                     specified when creating the object. In case of
    #                     false positive, the estimated motion is randomly
    #                     generated between the min and max values in the
    #                     actual data.
    # Output : matchDetected - Simulated match detection (True/False).
    #          theMotion - Simulated motion between images. In case of no
    #                      match detected, theMotion is set to zero.
    #          doMatch   - States whether the two images actually overlap or
    #                      not (ground truth to evaluate the system)
    ###########################################################################
    def match_images(self,firstID,secondID,addNoise=True,simulateFailures=False):
        # Check if the two image IDs correspond to matching images
        doMatch=(self.theOverlap[firstID,secondID]>=self.minOverlap)

        # If failures are to be simulated
        if simulateFailures:
            # Pick a random number to simulate failures
            randomNumber=np.random.random()
            # If positive
            if doMatch:
                # If random number is below TPR, we have a TP.
                matchDetected=randomNumber<=self.truePositiveRate
                # If we have a TP, pick the motion
                if matchDetected:
                    theMotion=np.array([self.relPosX[firstID,secondID],self.relPosY[firstID,secondID],self.relPosO[firstID,secondID]])
            # If negative
            else:
                # If random number is below TNR, we have a TN
                matchDetected=randomNumber>self.trueNegativeRate
                # If we have a FP, let's compute a random motion
                if matchDetected:
                    theMotion=np.array([np.random.uniform(low=self.xMin,high=self.xMax),
                                        np.random.uniform(low=self.yMin,high=self.yMax),
                                        np.random.uniform(low=self.oMin,high=self.oMax)])

        # No failures to simulate
        else:
            theMotion=np.array([self.relPosX[firstID,secondID],self.relPosY[firstID,secondID],self.relPosO[firstID,secondID]])
            matchDetected=doMatch

        # In all cases: if no match detected, output a 0 motion.
        if not matchDetected:
            theMotion=np.array([0.0,0.0,0.0])
        # In all cases: if match detected and noise requested, add it.
        elif addNoise:
            theMotion+=np.random.multivariate_normal([0,0,0],self.odoCovariance)

        # Return the simulated detection and motion as well as the ground truth
        return matchDetected,theMotion,doMatch