#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : preprocess_dataset
# Description : Preprocesses an UCAMGEN dataset as follows:
#               * Saves a sparse version of the overlap matrix using pickle
#               * Sets the motion between non-overlapping images to 0,
#                 sparsifies the motion matrices and saves them using pickle.
#               * Saves the perfect odometry in pickle format.
# Usage       : Change the PATH_DATASET variable to point to the UCAMGEN
#               dataset to preprocess and execute the script.
# Note        : The main goal of this pre-processing is to speed up
#               further accessess as well as reduce the size of files/data
#               to avoid out of memory problems when using large datasets.
#               If out-of-memory happens, re-run the script. It should work
#               since partial results are saved while the script is running.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 25-May-2021 - Creation
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import os
import numpy as np
from scipy.sparse import lil_matrix
from util import save_pickle_data,load_pickle_data,loadcsv

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def get_filenames(thePath,baseName):
    basePath=os.path.join(thePath,baseName)
    return basePath+'.csv',basePath+'.pkl'

###############################################################################
# PATH AND FILENAMES
###############################################################################
PATH_DATASET='../../DATA/GRIDSMALL'

FNAME_OVERLAP='OVERLAP'
FNAME_RELPOSX='IMGRPOSX'
FNAME_RELPOSY='IMGRPOSY'
FNAME_RELPOSO='IMGRPOSO'
FNAME_RELPOSH='IMGRPOSH'
FNAME_ODOM='ODOM'

###############################################################################
# PRE-PROCESS DATA
###############################################################################

#------------------------------------------------------------------------------
# LOAD, SPARSIFY AND SAVE OVERLAP MATRIX
#------------------------------------------------------------------------------
print('[ OVERLAP MATRIX ]')
csvName,pklName=get_filenames(PATH_DATASET,FNAME_OVERLAP)
pklNoLoopName=os.path.join(PATH_DATASET,'NOLOOPIDX.pkl')
if not (os.path.exists(pklName) and os.path.exists(pklNoLoopName)):
    print('  * PROCESSING.')
    overlapMatrix=loadcsv(csvName)
    noLoopIndexes=np.where(overlapMatrix<=0)
    save_pickle_data(pklNoLoopName,noLoopIndexes)
    overlapMatrix=lil_matrix(overlapMatrix)
    save_pickle_data(pklName,overlapMatrix)
else:
    noLoopIndexes=load_pickle_data(pklNoLoopName)
    print('  * ALREADY SAVED. SKIPPING.')

#------------------------------------------------------------------------------
# LOAD, REMOVE NON-LOOPS, SPARSIFY AND SAVE RELATIVE MOTION MATRICES
#------------------------------------------------------------------------------
relPosFNames=[os.path.join(PATH_DATASET,FNAME_RELPOSX),os.path.join(PATH_DATASET,FNAME_RELPOSY),os.path.join(PATH_DATASET,FNAME_RELPOSO),os.path.join(PATH_DATASET,FNAME_RELPOSH)]
for i,baseName in enumerate(relPosFNames):
    print('[ RELPOS %d OF %d ]'%(i+1,len(relPosFNames)))
    csvName,pklName=get_filenames(PATH_DATASET,baseName)
    if not os.path.exists(pklName):
        print('  * PROCESSING.')
        relPos=loadcsv(csvName,delimiter=',')
        relPos[noLoopIndexes]=0
        save_pickle_data(pklName,lil_matrix(relPos))
    else:
        print('  * ALREADY SAVED. SKIPPING.')

#------------------------------------------------------------------------------
# SAVE THE ODOMETRY IN PICKLE FORMAT
#------------------------------------------------------------------------------
print('[ ODOMETRY ]')
csvName,pklName=get_filenames(PATH_DATASET,FNAME_ODOM)
if not os.path.exists(pklName):
    print('  * PROCESSING.')
    thePoses=loadcsv(csvName)
    save_pickle_data(pklName,thePoses)
else:
    print('  * ALREADY SAVED. SKIPPING.')
