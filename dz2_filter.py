# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:59:32 2020

@author: malafeeva_dd
"""
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


"""---------------------------Параметры моделирования-----------------------"""
T      = 10e-3  
M      = 500000
c      = 3e8
w0     = 2 * math.pi * 1602e6

t_list = []
"""-------------------------------Моделируем шум----------------------------"""
# формирующий шум скорости
alpha     = 1
sigma_a   = 10
S_ksi     = 2 * (sigma_a**2) * alpha * ((w0/c)**2)
sigma_ksi = math.sqrt(S_ksi/2*T)
ksi_list  = np.random.normal(loc = 0.0, scale = sigma_ksi * 1, size = M)

# первый остчет
ksi_k     = np.array([[ksi_list[0]]])

# шум наблюдения
q_c_n0    = 10 ** (0.1 * 30)
N0        = (2 / (q_c_n0 * T**2)) * (1 + (1/(2 * q_c_n0 * T)))
sigma_n   = math.sqrt(N0/(2 * T))
n_list    = np.random.normal(loc = 0.0, scale = sigma_n * 1, size = M)

"""----------------------------Инициализация--------------------------------"""
# начальные значения истинных параметров
Omega_true            = 100
v_true                = 100
Omega_true_list       = []
v_true_list           = []

# начальные значения скорректированных оценок
Omega_corr            = 0
v_corr                = 0
Omega_corr_list       = [] 
v_corr_list           = [] 
sigma_Omega_corr_list = []

# матрицы фильтра
X_true   = np.array([[Omega_true],\
                         [v_true]])

X_corr   = np.array([[Omega_corr],\
                        [v_corr]])                                  # 2x1
    
I        = np.eye(2)                                                # 2x2

H        = np.array([[1, 0]])                                       # 1x2

K        = np.array([[0],\
                     [0]])                                          # 2x1

D_x_corr = np.array([[34**2,     0],\
                     [    0, 340**2]])                              # 2x2
   
G        = np.array([[0],\
                     [alpha * T]])                                  # 2x1

D_ksi    = np.array([[sigma_ksi**2]])                               # 1x1

D_n      = np.array([[sigma_n**2]])  

F        = np.array([[1,              T],\
                     [0, (1 - alpha * T)]])                         # 2x2
    
for k in range(0, M, 1):
    t = k * T
    t_list.append(t)
    """----------------------Входное воздействие----------------------------"""
    X_true     = F.dot(X_true) + G.dot(ksi_k)
    
    ksi_k      = np.array([[ksi_list[k]]])
    
    Omega_true_list.append(X_true[0])
    
    v_true_list.append(X_true[1])
       
    y          = H.dot(X_true) + n_list[k]
    
    """------------------------------Фильтрация-----------------------------"""
    """----------------------------Экстраполяция----------------------------"""
            
    X_extr       = F.dot(X_corr)                                   # 2x1
    
    D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())            # 2x2

    D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())               # 2x2

    D_x_extr     = D_x_extr_pt1 + D_x_extr_pt2                     # 2x2
        
    """------------------------------Коррекция------------------------------"""
    K            = (D_x_extr.dot(H.transpose())).dot(inv(((H.dot(D_x_extr)).dot(H.transpose())) + D_n))  # 2x1
    
    D_x_corr     = (I - K.dot(H)).dot(D_x_extr)                    # 2x2        
    
    X_corr       = X_extr + K.dot(y - H.dot(X_extr))               # 2x1
    
    Omega_corr   = X_corr[0]
    Omega_corr_list.append(Omega_corr)
    
    v_corr       = X_corr[1]
    v_corr_list.append(v_corr)    
    
    sigma_Omega_corr = math.sqrt(D_x_corr[0,0])
    sigma_Omega_corr_list.append(sigma_Omega_corr)
    
    

plt.figure(1)
plt.plot(t_list[0::], Omega_true_list[0::], '.-', color = 'green', linewidth = 1)
plt.plot(t_list[0::], Omega_corr_list[0::], '.-', color = 'hotpink', linewidth = 1)
plt.xlabel('t')
plt.ylabel('Omega_true(t), Omega_corr(t)')
plt.legend(['Omega_true(t)','Omega_corr(t)'])
plt.grid()
plt.show()  

plt.figure(2)
plt.plot(t_list[0::], v_true_list[0::], '.-', color = 'green', linewidth = 1)
plt.plot(t_list[0::], v_corr_list[0::], '.-', color = 'hotpink', linewidth = 1)
plt.xlabel('t')
plt.ylabel('v_true(t), v_corr(t)')
plt.legend(['v_true(t)','v_corr(t)'])
plt.grid()
plt.show()  

plt.figure(3)
plt.plot(t_list[0::], sigma_Omega_corr_list[0::], '.-', color = 'green', linewidth = 1)
plt.xlabel('t')
plt.ylabel('D_Omega_corr(t)')
plt.legend(['D_Omega_corr(t)'])
plt.grid()
plt.show() 
