# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:59:32 2020

@author: malafeeva_dd
"""
import numpy as np
import math
from numpy.linalg import inv

"""-----------------------------Параметры моделирования---------------------"""
T  = 10*1e-3  
M  = 1000

c  = 3*1e8
w0 = 2 * math.pi * 1602 * 1e6

"""-----------------------Моделируем шум------------------------------------"""
# формирующий шум скорости
alpha     = 1
sigma_a   = 10
S_ksi     = 2 * (sigma_a**2) * alpha * (w0/c)**2
sigma_ksi = S_ksi/2*T
ksi_list  = np.random.uniform(0, 1, M) * sigma_ksi 

# шум наблюдения
q_c_n0  = 10 ** (0.1 * 30)
N0      = (2 / q_c_n0 * T**2) * (1 + 1/2 * q_c_n0 * T)
sigma_n = N0/2 * T
n_list  = np.random.uniform(0, 1, M) * sigma_n


# в v_true берутся отсчеты ksi с предыдущего шага
ksi_k   = ksi_list[0]

"""---------------------------Инициализация--------------------------------"""

# начальные значения истинных параметров
Omega_true = 100
v_true     = 100

# начальные значения скорректированных оценок
X_corr_list = []
Omega_corr  = 0
v_corr      = 0
X_corr      = np.array([[Omega_corr],\
                        [v_corr]])                               # 2x1
    
# матрицы фильтра
I        = np.eye(2)                                             # 2x2

H        = np.array([[1, 0]])                                    # 1x2

K        = np.array([[0],\
                     [0]])                                       # 2x1

D_x_corr = np.array([[34**2,     0],\
                     [    0, 34**2]])                            # 2x2
   
G        = np.array([[0],\
                     [1]])                                       # 2x1

D_ksi    = np.array([[sigma_ksi**2]])                            # 1x1

D_n      = np.array([[sigma_n**2]])  

F        = np.array([[1,               0 * T],\
                     [0, (1 - alpha * 0 * T)]])                  # 2x2
    
"""-------------------------------Фильтрация--------------------------------""" 
for k in range(0, M, 1):
    """----------------------Входное воздействие----------------------------"""
    
    Omega_true = Omega_true + v_true * k * T
    
    v_true     = v_true * (1 - alpha * k * T) + alpha * k * T * ksi_k
    
    ksi_k      = ksi_list[k]

    n          = n_list[k]
    
    y          = Omega_true + n
            
    """----------------------------Экстраполяция----------------------------"""
            
    X_extr       = F.dot(X_corr)                                   # 2x1
    
    D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())            # 2x2

    D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())               # 2x2

    D_x_extr     = D_x_extr_pt1 + D_x_extr_pt2                     # 2x2
    
    F            = np.array([[1,               k * T],\
                             [0, (1 - alpha * k * T)]])            # 2x2
        
    """------------------------------Коррекция------------------------------"""
        
    K            = D_x_extr.dot(H.transpose()).dot(inv((H.dot(D_x_extr)).dot(H.transpose()) + D_n))  # 2x1
    
    D_x_corr     = (I - K.dot(H)).dot(D_x_extr)                    # 2x2        
    
    X_corr       = X_extr + K.dot(y - H.dot(X_extr))               # 2x1
    
    X_corr_list.append(X_corr)
    
