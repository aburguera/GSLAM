#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : plotutil
# Description : Plot related utilities
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 3-August-2021 - Creation
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
from transform2d import compose_point

###############################################################################
# Draws a 2-sigma 2D uncertainty ellipse
# Input       : X - Mean
#               P - Covariance
#               theColor - Outline color in matplotlib.pyplot format
###############################################################################
def draw_ellipse(X,P,theColor='k'):
    tita=np.linspace(0,2*math.pi,20)
    theCircle=np.array([np.cos(tita),np.sin(tita)])
    D,V=np.linalg.eig(P[:2,:2])
    idx=np.argsort(D)[::-1]
    D=D[idx]
    V=V[idx,:]
    ejes=np.sqrt(9.2103*D)
    theEllipse=np.dot(np.dot(V,np.diag(ejes)),theCircle)+X[:2].reshape((2,1))
    plt.plot(theEllipse[0,:],theEllipse[1,:],theColor)

###############################################################################
# Draws a triangle representing the robot
# Input       : X - Robot pose - nparray [x,y,o]
#               P - Robot pose covariance - nparray 3x3 or None
#               theColor - Outline color in matplotlib.pyplot format
###############################################################################
def draw_robot(X,theSize=100,theColor='r'):
    thePoints=compose_point(X,np.array([[-0.25,0,0.25,-0.25],[0.5,-0.5,0.5,0.5]])*theSize)
    plt.plot(thePoints[0,:],thePoints[1,:],theColor)

###############################################################################
# Draws a robot trajectory
# Input       : Xhist - nparray (3xN) containing N absolute robot poses.
#               theLoops - nparray (2xM) relating source (top row) and
#                          destination (bottom row) columns in Xhist that
#                          close a loop.
#               mainStep - Step to plot the trajectory
#               secondaryStep - Step to plot robots within the trajectory to
#                               illustrate the orientation. If 0 or <0, no
#                               robots are drawn.
#               drawLast - Plot a robot at the end of the trajectory.
###############################################################################
def draw_trajectory(Xhist,theLoops=None,mainStep=1,secondaryStep=5,drawLast=True,robotSize=100):
    # Plot the loops if available
    if not theLoops is None:
        theX=np.stack((Xhist[0,theLoops[0,:]],Xhist[0,theLoops[1,:]]))
        theY=np.stack((Xhist[1,theLoops[0,:]],Xhist[1,theLoops[1,:]]))
        plt.plot(theX,theY,'b')

    # Plot the trajectory
    plt.plot(Xhist[0,::mainStep],Xhist[1,::mainStep],'k')

    # Plot a few "robots" to see the orientations at some points
    if secondaryStep>=1:
        for i in range(0,Xhist.shape[1],secondaryStep):
            draw_robot(Xhist[:,i],theSize=robotSize)

    # Draw the most recent pose
    if drawLast:
        draw_robot(Xhist[:,-1],theColor='g',theSize=robotSize)