#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:18:01 2020

@author: daryamalafeeva
"""
import numpy as np
import codecs
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv

"""--------------------------Параметры моделирования------------------------"""
f_s = 47.5*1e6
T   = 1/f_s
M   = 2048
k   = 0

"""----------------------------Моделирование шума---------------------------"""
sigma_n = 10
mu_n    = 0
n       = np.random.normal(mu_n, sigma_n, M)

# матрица дисперсии шума наблюдений
Dn = np.eye(5) * 1/(sigma_n**2)

# начальные значения параметров вектора lam_array
A1        = 7010
A2        = 3500
f         = 0.25*1e6
w         = 2 * math.pi * f
phi_0     = 150 * math.pi /180
delta_phi = 75 * math.pi / 180

# начальное значение для метода простой итерации
delta_phi_old = 0.1 * delta_phi

# вектор оцениваемых параметров сигнала
lam_array = np.array([A1, A2, w, phi_0, delta_phi])


y1_list = []
y2_list = []
y3_list = []
y4_list = []
S1_list = []
S2_list = []
S3_list = []
S4_list = []

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
        # метод простой итерации
 #       while abs(delta_phi - delta_phi_old)> 1e-8:
            # сигналы
            # S1 = A1 * math.cos(w * k * T + phi_0)
            # S1_list.append(S1)
            # S2 = A1 * math.sin(w * k * T + phi_0)
            # S2_list.append(S2)
            # S3 = A2 * math.cos(w * k * T + phi_0 + delta_phi)
            # S3_list.append(S3)
            # S4 = A2 * math.sin(w * k * T + phi_0 + delta_phi)
            # S4_list.append(S4)
            
            # первые производные функции правдоподобия
            # d_dA1 = 1/(sigma_n**2) * (sum(math.cos(w * k * T + phi_0) * y1_list +\
            #         math.sin(w * k * T + phi_0) * y2_list) - M * A1)
            
        
        #     #метод дискриминаторов
        #     lam_array_old = lam_array
        #     lam_array     = lam_array - L.transpose() * inv(H)
            
        # # нижняя граница Рао-Крамера
        # J = -inv(H)
        # # ошибка оценивания delta_phi
        # D_delta_phi = J[4,4]
    
        # шаг
#        k +=1        








# строим реализации
figure_1 = plt.figure(1)
plt.plot(range(0, len(y1_list)), y1_list, color = 'deeppink', linewidth = 2)
plt.plot(range(0, len(y2_list)), y2_list, color = 'orange', linewidth = 2)
plt.plot(range(0, len(y3_list)), y3_list, color = 'gold', linewidth = 2)
plt.plot(range(0, len(y4_list)), y4_list, color = 'limegreen', linewidth = 2)
plt.xlabel('k')
plt.ylabel('yn(k)')
plt.legend(['y1','y2','y3', 'y4'])
plt.title('Реализации на входе приёмника')
plt.grid()
plt.show()
