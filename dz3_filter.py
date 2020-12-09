# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:21:03 2020

@author: malafeeva_dd
"""
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

"""-----------------------------Параметры моделирования---------------------"""
T      = 10 * 1e-3 
T_d    = 0.2 * 1e-6
M      = 100000
c      = 3* 1e8
w0     = 2 * math.pi * 1602e6
w_p    = 2 * math.pi * 2 * 1e6
alpha  = 1
q_c_n0 = 10 ** (0.1 * 30)
t_list = []
N      = T/T_d

"""-----------------------Моделируем шум------------------------------------"""
# формирующий шум пси
sigma_psi = 0.5
psi_list  = np.random.normal(loc = 0.0, scale = sigma_psi * 1, size = M)
psi_i     = np.array([[psi_list[0]]])


# формирующий шум кси
sigma_a   = 10
S_ksi     = 2 * (sigma_a**2) * alpha * ((w0/c)**2)
sigma_ksi = math.sqrt((S_ksi)/(2*T))
ksi_list  = np.random.normal(loc = 0.0, scale = sigma_ksi * 1, size = M)
ksi_i     = np.array([[ksi_list[0]]])

# шум наблюдения
sigma_n   = 35.4
n_list    = np.random.normal(loc = 0.0, scale = sigma_n * 1, size = M)

"""---------------------------Инициализация--------------------------------"""
# начальные значения истинных параметров
a_true          = 1
phi_true        = math.pi/12
Omega_true      = 100
v_true          = 100
a_true_list     = []
phi_true_list   = []
Omega_true_list = []
v_true_list     = []

# начальные значения скорректированных оценок
a_corr                = 0.5
phi_corr              = 0
Omega_corr            = 0
v_corr                = 0
a_corr_list           = []
phi_corr_list         = []
Omega_corr_list       = [] 
v_corr_list           = [] 
sigma_Omega_corr_list = []
Eps_Omega_list        = []
D11_list              = []

# матрицы фильтра
X_true   = np.array([[a_true],\
                     [phi_true],\
                     [Omega_true],\
                     [v_true]])

# X_corr   = np.array([[a_corr],\
#                      [phi_corr],\
#                      [Omega_corr],\
#                      [v_corr]])                                 # 2x1
    
# I        = np.eye(4)                                                # 2x2

# H        = np.array([[1, 0]])                                       # 1x2

# D_x_corr = np.array([[(0.3)**2, 0, 0, 0],\
#                      [0, (math.pi)**2, 0, 0],\
#                      [0, 0, 34**2, 0],\
#                      [0, 0, 0, 340**2]])                              # 2x2
   
G        = np.array([[T,         0],\
                     [0,         0],\
                     [0,         0],\
                     [0, alpha * T]])                                  # 2x1

# D_ksi    = np.array([[sigma_ksi**2]])                               # 1x1

# D_n      = np.array([[sigma_n**2]])  

F        = np.array([[1, 0, 0,               0],\
                     [0, 1, T,               0],\
                     [0, 0, 1,               T],\
                     [0, 0, 0, (1 - alpha * T)]])                    # 4X4

    
psi_ksi  = np.array([[psi_i],\
                     [ksi_i]])    
    

# внешний цикл быстрый по Td
# внутренний должен быть медленным по T
# в моменты когда Td=T производим коррекцию    
i = 0
k = 0
while t_i < M:
    i+=1
    t_i = i*T_d
    """----------------------Входное воздействие----------------------------"""
    X_true  = F.dot(X_true) + G.dot(psi_ksi)
    a_true_list.append(X_true[0])
    phi_true_list.append(X_true[1])
    Omega_true_list.append(X_true[2])
    v_true_list.append(X_true[3])
    
    # ksi_k        = np.array([[ksi_list[k]]])

    # y            = H.dot(X_true) + n_list[k]
    
    while t_k < M/N:
        k+=1
        t_k = k*T
        if t_i == t_k:
        
    #     """------------------------------Фильтрация-----------------------------"""
        
    
    #     """----------------------------Экстраполяция----------------------------"""
    #     X_extr       = F.dot(X_corr)                                   # 2x1
        
    #     D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())            # 2x2
    
    #     D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())               # 2x2
    
    #     D_x_extr     = D_x_extr_pt1 + D_x_extr_pt2                     # 2x2
        
    #     """------------------------------Коррекция------------------------------"""
    #     K            = (D_x_extr.dot(H.transpose())).dot(inv(((H.dot(D_x_extr)).dot(H.transpose())) + D_n))  # 2x1
        
    #     D_x_corr     = (I - K.dot(H)).dot(D_x_extr)                    # 2x2        
        
    #     X_corr       = X_extr + K.dot(y - H.dot(X_extr))               # 2x1
        
    #     # мгновенная ошибка фильтрации 
    #     Eps_Omega    = X_corr[0] - X_true[0]
    #     Eps_Omega_list.append(Eps_Omega)
        
    #     Omega_corr   = X_corr[0]
    #     Omega_corr_list.append(Omega_corr)
        
    #     v_corr       = X_corr[1]
    #     v_corr_list.append(v_corr)    
    
    #     sigma_Omega_corr = math.sqrt(D_x_corr[0,0])
    #     sigma_Omega_corr_list.append(sigma_Omega_corr)
        
    #     D11 = D_x_corr[0,0]
    #     D11_list.append(D11)
    
# plt.figure(1)
# plt.plot(t_list[0::], Omega_true_list[0::], '.', color = 'mediumblue', linewidth = 1)
# plt.plot(t_list[0::], Omega_corr_list[0::], '.-', color = 'magenta', linewidth = 1)
# plt.xlabel('t, с')
# plt.ylabel('Omega_true(t), Omega_corr(t), рад/с')
# plt.legend(['Omega_true(t)', 'Omega_corr(t)'])
# plt.title('Зависимость истинной доплеровской частоты и ее оценки от времени')
# plt.grid()
# plt.show()  

# plt.figure(2)
# plt.plot(t_list[0::], sigma_Omega_corr_list[0::], '.-', color = 'blueviolet', linewidth = 1)
# plt.xlabel('t, с')
# plt.ylabel('σΩ_corr(t), рад/с')
# plt.title('Зависимость СКО ошибки оценивания доплеровской частоты от времени')
# plt.grid()
# plt.show() 

# plt.figure(3)
# plt.plot(t_list[0::], Eps_Omega_list[0::], '.-', color = 'orangered', linewidth = 1)
# plt.xlabel('t, с')
# plt.ylabel('Eps_Omega(t), рад/с')
# plt.legend(['Eps_Omega(t)'])
# plt.title('Зависимость мгновенной ошибки фильтрации частоты от времени')
# plt.grid()
# plt.show() 

# plt.figure(4)
# plt.plot(t_list[0::], D11_list[0::], '.-', color = 'blueviolet', linewidth = 1)
# plt.plot(t_list[0::], Eps_Omega_list[0::], '.-', color = 'orangered', linewidth = 1)
# plt.xlabel('t, с')
# plt.ylabel('D11(t), (рад/с)^2;  Eps_Omega(t), рад/с')
# plt.legend(['D11(t)','Eps_Omega(t)'])
# plt.grid()
# plt.show() 

# plt.figure(5)
# plt.plot(t_list[0::], Omega_true_list[0::], '.', color = 'mediumblue', linewidth = 1)
# plt.xlabel('t, с')
# plt.ylabel('Omega_true(t), рад/с')
# plt.title('Зависимость истинной доплеровской частоты от времени')
# plt.grid()
# plt.show()  
