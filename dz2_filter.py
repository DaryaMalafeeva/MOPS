# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:59:32 2020

@author: malafeeva_dd
"""
import numpy as np

"""---------------------------Инициализация--------------------------------"""
sigma_Omega_form = 0.1
sigma_v_form     = 0.1

alpha  = 1
Omega  = 1
v      = 1
X_corr = np.array[(Omega, v)]

D_x_corr = np.array([[34**2,     0],\
                     [    0, 34**2]])
I        = np.eye(2)
H        = np.array([1, 0])
K        = np.array([[0],\
                     [0]])

G        = np.array([[sigma_Omega_form,            0],\
                     [0,                sigma_v_form]])

    
X_corr_list = []    
    
M = 1000

for k in range(0, M, 1):
    """----------------------------Экстраполяция----------------------------"""
    F            = np.array([[1, k],\
                             [0, (1 - alpha * k)]])
        
    X_extr       = F.dot(X_corr)
    
    D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())

    D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())

    D_x_extr     = D_extr_pt1 + D_extr_pt2
    
    """------------------------------Коррекция------------------------------"""
    D_x_corr     = (I - K.dot(H)).dot(D_x_extr)
    
    K_pt1        = D_x_extr.dot(H.transpose())
    
    K_pt2        = inv((H.dot(D_x_extr)).dot(H.tanspose()) + D_n)
    
    K            = K_pt1.dot(K_pt2)
    
    X_corr       = X_extr + K.dot(y - H.dot(X_extr))
    
    X_corr_list.append(X_corr)
    
    
Omega_corr_list = X_corr_list[0,:]
v_corr_list     = X_corr_list[1,:]