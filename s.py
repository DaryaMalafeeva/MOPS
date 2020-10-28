#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:22:14 2020

@author: daryamalafeeva
"""


import numpy as np
from numpy.linalg import inv
import dz1_MOPS_Malafeeva as dz


H = np.array([[1, 2, 4, 6, 7],
              [2, 4, 5, 3, 6],
              [1, 3, 2, 5, 4],
              [6, 7, 4, 3, 6],
              [1, 5, 4, 6, 7]])

J = -inv(H)

aa = dz.y1
print(aa)