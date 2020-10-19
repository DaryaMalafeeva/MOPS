#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:18:01 2020

@author: daryamalafeeva
"""
import numpy as np
import codecs

"""--------------------------Параметры моделирования------------------------"""
f_s = 47.5*1e6
T   = 1/f_s
M   = 2048

"""----------------------------Моделирование шума---------------------------"""
sigma_n = 10
mu_n    = 0
n       = np.random.normal(mu_n, sigma_n, M)

#lam = np.array([A1], [A2], [w], [phi_0], [delta_phi])

Dn = np.eye(5) * sigma_n


y1_list = []
y2_list = []
y3_list = []
y4_list = []

with codecs.open('Input_Y0toT.txt', "r", encoding='utf-8', errors='ignore') as log:
    for line in log:
        str_massive = line.split()
        y1          = int(str_massive[0])
        y1_list.append(y1)
        y2          = int(str_massive[1])
        y2_list.append(y2)
        y3          = int(str_massive[2])
        y3_list.append(y3)
        y4          = int(str_massive[3])
        y4_list.append(y4)
        
        
        # выходной сигнал с дискриминатора с нормированной крутизной ДХ
#        u_d_n = 
        
        
        # оценка максимального правдоподобия
        lam_m = lam_op + u_d_n
        
        




#   y1 = A1 * math.cos(w * k * T + phi_0) + n
#    y2 = A1 * math.cos(w * k * T + phi_0) + n