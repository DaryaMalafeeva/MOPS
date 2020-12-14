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
T        = 10  * 1e-3 
T_d      = 0.2 * 1e-6
mod_time = 2
c        = 3* 1e8
w0       = 2 * math.pi * 1602* 1e6
w_p      = 2 * math.pi * 2   * 1e6
alpha    = 1
q_c_n0   = 10 ** (0.1 * 30)
t_list   = []
N        = int(T/T_d)

"""-----------------------Моделируем шум------------------------------------"""
# формирующий шум пси
sigma_psi = 0.5
psi_list  = np.random.normal(loc = 0.0, scale = sigma_psi * 1, size = N)

# формирующий шум кси
sigma_a   = 10
S_ksi     = 2 * (sigma_a**2) * alpha * ((w0/c)**2)
sigma_ksi = math.sqrt((S_ksi)/(2*T))
ksi_list  = np.random.normal(loc = 0.0, scale = sigma_ksi * 1, size = N)

# шум наблюдения
sigma_n   = 35.4
n_list    = np.random.normal(loc = 0.0, scale = sigma_n * 1, size = N)

"""---------------------------Инициализация--------------------------------"""
# начальные значения истинных параметров
a_true          = 1
phi_true        = math.pi/12
Omega_true      = 100
v_true          = 100

# начальные значения скорректированных оценок
a_corr          = 0.5
phi_corr        = 0
Omega_corr      = 0
v_corr          = 0

# матрицы фильтра
X_true   = np.array([[a_true],\
                     [phi_true],\
                     [Omega_true],\
                     [v_true]])

X_corr   = np.array([[a_corr],\
                      [phi_corr],\
                      [Omega_corr],\
                      [v_corr]])                                     # 4x1
    

C        = np.array([[1, 0, 0, 0],\
                     [0, 1, 0, 0]])                                       # 1x2

D_x_corr = np.array([[(0.3)**2, 0, 0, 0],\
                      [0, (math.pi)**2, 0, 0],\
                      [0, 0, 34**2, 0],\
                      [0, 0, 0, 340**2]])                             # 4x4
   
G        = np.array([[T,         0],\
                     [0,         0],\
                     [0,         0],\
                     [0, alpha * T]])                                 # 4x1

D_ksi    = np.array([[sigma_psi**2,            0],\
                     [           0, sigma_ksi**2]])                   # 2x2

    
psi_ksi  = np.array([[psi_list[0]],\
                     [ksi_list[0]]])       
    
    
D_n      = np.array([[sigma_n**2]])                                    #1x1

F        = np.array([[1, 0, 0,               0],\
                     [0, 1, T,               0],\
                     [0, 0, 1,               T],\
                     [0, 0, 0, (1 - alpha * T)]])                    # 4x4


W11      = (N)/(2 * sigma_n**2)
W12      = 0
W21      = 0
# W22 пересчитывается после экстраполяции
W22      = 0
W        = np.array([[W11, W12],\
                     [W21, W22]])
    
    
k   = 0
t_k = 0
while t_k < mod_time:
    t_k +=T
    
    """----------------------Входное воздействие----------------------------"""
    X_true  = F.dot(X_true) + G.dot(psi_ksi)
    
    psi_ksi = np.array([[psi_list[k]],\
                        [ksi_list[k]]]) 
        
    # амплитуда
    if t_k < 1:
        X_true[0][0] = 0.5
    elif t_k >= 1:
        X_true[0][0] = 1
    
    """----------------------------Экстраполяция----------------------------"""
    X_extr       = F.dot(X_corr)                                   # 4x1
    
    D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())            # 4x4

    D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())               # 4x4

    D_x_extr     = D_x_extr_pt1 + D_x_extr_pt2                     # 4x4

    W22          = (N *((X_extr[0][0])**2))/(2 * sigma_n**2)
    
    """--------------------------Коррелятор---------------------------------"""
    # фаза
    phi_p        = np.array(range(0, N, 1)) * T_d * w_p
    
    y            = X_true[0][0] * np.cos(phi_p + X_true[1][0]) + n_list[k]
    
    S_sin        = np.sin(phi_p + X_corr[1][0])
    
    S_cos        = np.cos(phi_p + X_corr[1][0])
    
    I            = np.sum(np.dot(y, S_cos))
    
    Q            = np.sum(np.dot(y, S_sin))
    
    U_1          = I * (1/D_n) - (X_extr[0][0] * N)/(2 * D_n)
    
    U_2          = Q * (X_extr[0][0])/(D_n)
    
    U            = np.array([U_1[0],\
                             U_2[0]])
        
    """------------------------------Коррекция------------------------------"""
    
    D_x_corr     = inv(inv(D_x_extr) + ((C.transpose().dot(W)).dot(C)))  # 4x4        
    
    X_corr       = X_extr + ((D_x_corr.dot(C.transpose())).dot(U))     # 4x1
    
    k+=1
    
    print('--------------')
    print('Шаг №' + str(k))
    print(X_corr)
    print('             ')
    
    

    
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
