#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : LoopModel
# Description : Simple wrapper to ease the acces to the loop detector model.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 06-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from keras.layers import Input, BatchNormalization,concatenate,Dense,Flatten
from modelwrapper import ModelWrapper
from keras.models import Model

class LoopModel(ModelWrapper):

###############################################################################
# CREATES AND COMPILES THE MODEL
###############################################################################
    def create(self,
               featureExtractorModel,
               denseLayers=[128,16],
               categoricalClasses=2,
               lossFunction='categorical_crossentropy',
               theMetrics=['categorical_accuracy'],
               doRegression=False,
               trainFeatureExtractor=False):

        # Get the input shape from the feature extractor
        inputShape=featureExtractorModel.layers[0].input_shape[0][1:]

        # Prepare feature extractor for training/not training
        for layer in featureExtractorModel.layers:
            layer.trainable=trainFeatureExtractor

        # Input layers
        firstInput=Input(shape=inputShape)
        secondInput=Input(shape=inputShape)

        # Siamese outputs
        firstBranchOutput=featureExtractorModel(firstInput)
        secondBranchOutput=featureExtractorModel(secondInput)

        # Join the siamese outputs
        x=concatenate([firstBranchOutput,secondBranchOutput])
        x=Flatten()(x)
        x=BatchNormalization()(x)

        # Add the dense layers
        for curLayer in denseLayers:
            x=Dense(curLayer,activation='sigmoid')(x)

        # Add the output layer. If this is a regression problem or a classifi-
        # cation problem without categorical representation, output one
        # value between 0 and 1 using sigmoid activation. If it is a classifi-
        # cation problem with categorical representation, output
        # categoricalClasses neurons with softmax activation.
        if doRegression or categoricalClasses==0:
            theOutput=Dense(1,activation='sigmoid')(x)
        else:
            theOutput=Dense(categoricalClasses,activation='softmax')(x)

        # Create the model
        self.theModel=Model([firstInput,secondInput],theOutput)

        # Compile the model
        self.theModel.compile(optimizer='adam',loss=lossFunction,metrics=theMetrics)