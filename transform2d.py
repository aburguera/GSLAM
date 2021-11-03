#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : transform2d
# Description : 2D transformation utilities
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 4-June-2021 - Creation
# Note        : I have some "Matlab-related bad habits", which include
#               representing motions as matrices, even if they are vectors.
#               That is why motions have three rows and 1 column. This leads
#               to strange situations here and there where I convert to
#               vectors and then back to matrices. Don't you like it? Well...
#               How much have you paid for this software? Nothing? That
#               settles the question: deal with it.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import math

###############################################################################
# Makes an angle to be in the interval -pi,pi
###############################################################################
def normalize_angle(theAngle):
    while theAngle>np.pi:
        theAngle-=2*np.pi
    while theAngle<-np.pi:
        theAngle+=2*np.pi
    return theAngle

###############################################################################
# Makes an array of angles to be in the interval -pi,pi
# Use this version instead of normalize_angle only if several angles have
# to be normalized. The previous one is, in general, faster than this one
# for a single angle.
###############################################################################
def normalize_angles(theAngles):
    return theAngles+(2*np.pi)*np.floor((np.pi-theAngles)/(2*np.pi))

###############################################################################
# Point composition
# Given a 2D transformation X1=[x,y,theta], this function applies it to
# all the points in X2.
# X2 has 2 rows (x,y) and N columns (the points)
###############################################################################
def compose_point(X1,X2):
    s=math.sin(X1[2])
    c=math.cos(X1[2])
    return np.vstack((X1[0]+np.matmul([c,-s],X2),X1[1]+np.matmul([s,c],X2)))

###############################################################################
# Composes two 2D poses
# Input       : X1 - Transformation from A to B
#               X2 - Transformation from B to C
#               P1 - Covariance associated to X1
#               P2 - Covariance associated to X2
# Output      : X3 - Transformation from A to C
#               P3 - Provided only if P1!=None and P2!=None. Covariance asso-
#                    ciated to X3
###############################################################################
def compose_references(X1,X2,P1=None,P2=None):
    s=math.sin(X1[2])
    c=math.cos(X1[2])
    X3=np.zeros((3,1))
    X3[0,0]=X1[0]+X2[0]*c-X2[1]*s
    X3[1,0]=X1[1]+X2[0]*s+X2[1]*c
    X3[2,0]=normalize_angle(X1[2]+X2[2])
    if not (P1 is None or P2 is None):
        J1=np.array([[1,0,-X2[0]*s-X2[1]*c],
                     [0,1,X2[0]*c-X2[1]*s],
                     [0,0,1]])
        J2=np.array([[c,-s,0],
                     [s,c,0],
                     [0,0,1]])
        P3=np.dot(np.dot(J1,P1),J1.transpose())+np.dot(np.dot(J2,P2),J2.transpose())
        return X3,P3
    else:
        return X3

###############################################################################
# Inverts a pose/reference
# Input       : X1 - Transformation from A to B
#               P1 - Covariance associated to X1
# Output      : Xnew - Transformation from B to A
#               Pinv - Provided only if P1!=None. Covariance associated to Xnew
###############################################################################
def invert_reference(X1,P1=None):
    Xnew=np.zeros((3,1))
    s=math.sin(X1[2])
    c=math.cos(X1[2])
    Xnew[0,0]=-X1[0]*c-X1[1]*s
    Xnew[1,0]=X1[0]*s-X1[1]*c
    Xnew[2,0]=normalize_angle(-X1[2])
    if not P1 is None:
        Jinv=np.array([[-c, -s, -X1[0]*s-X1[1]*c],
                       [s, -c, X1[0]*c+X1[1]*s],
                       [0, 0, -1]])
        Pinv=np.dot(np.dot(Jinv,P1),Jinv.transpose())
        return Xnew,Pinv
    else:
        return Xnew

###############################################################################
# Given a vector of odometric (relative) motion estimates, outputs the
# corresponding vector with absolute poses.
# Input  : Xodo - nparray (3xN) Each column is a motion [deltax,deltay,deltao]
# Output : Xhist - nparray (3xN) : Each column is a pose [x,y,o]
###############################################################################
def compose_trajectory(Xodo):
    Xcur=np.zeros((3,1))
    Xhist=np.zeros((3,Xodo.shape[1]))
    for i in range(Xodo.shape[1]):
        Xcur=compose_references(Xcur,Xodo[:3,i]).reshape((3))
        Xhist[:,i]=Xcur
    return Xhist

###############################################################################
# Motion that minimizes sum of squared distances
# Given two 2D point clouds (Sref and Scur) it computes the 2D motion so that
# when applied to Scur, the sum of squared distances between points is
# minimum. Sref and Scur must have the same number of points, and the i-th
# point in Scur must correspond to the i-th point in Sref.
# Sref and Scur have 2 rows (x,y) and N columns (the points).
# Input  : Sref - nparray (2xN)
#          Scur - nparray (2xN)
# Output : theMotion - nparray (3x1) : Estimated motion from the reference
#                      frame of Sref to the reference frame of Scur.
###############################################################################
def least_squares_cartesian(Sref,Scur):
    mx=np.mean(Scur[0,:])
    my=np.mean(Scur[1,:])
    mx2=np.mean(Sref[0,:])
    my2=np.mean(Sref[1,:])
    Sxx=np.sum((Scur[0,:]-mx)*(Sref[0,:]-mx2))
    Syy=np.sum((Scur[1,:]-my)*(Sref[1,:]-my2))
    Sxy=np.sum((Scur[0,:]-mx)*(Sref[1,:]-my2))
    Syx=np.sum((Scur[1,:]-my)*(Sref[0,:]-mx2))
    o=math.atan2(Sxy-Syx,Sxx+Syy)
    x=mx2-(mx*math.cos(o)-my*math.sin(o))
    y=my2-(mx*math.sin(o)+my*math.cos(o))
    return np.array([[x],[y],[o]])