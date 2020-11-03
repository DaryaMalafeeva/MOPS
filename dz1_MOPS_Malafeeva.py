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

"""---------------------------Обработка файла-------------------------------"""
# инициализация списков
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

y_list = [y1_list, y2_list, y3_list, y4_list]

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

"""--------------------------Параметры моделирования------------------------"""
f_s     = 47.5*1e6
T       = 1/f_s
M       = 2047
sigma_n = 10

"""---------------------Начальные параметры алгоритма-----------------------"""
# начальные значения параметров вектора lam_array
A1        = 7020
A2        = 3500
f         = 1.56*1e6
w         = 2 * math.pi * f
phi_0     = -1.75
delta_phi = 2.27
lam_array = np.array([A1, A2, w, phi_0, delta_phi])

S1_list = []
S2_list = []
S3_list = []
S4_list = []

# списки под производные
d1_dA1_list              = []
d1_dA2_list              = []
d1_dw_list               = []
d1_dphi_0_list           = []
d1_ddelta_phi_list       = []
d2_dA1dw_list            = []
d2_dA1dphi_0_list        = []
d2_dA2dw_list            = []
d2_A2dphi_0_list         = []
d2_dA2ddelta_phi_list    = []
d2_dw_list               = []
d2_dwdphi_0_list         = []
d2_dwddelta_phi_list     = []
d2_dphi_0_list           = []
d2_dphi_0ddelta_phi_list = []
d2_ddelta_phi_list       = []

"""---------------------Алгоритм оценивания параметров----------------------"""
# начальное значение для метода Ньютона
delta_phi_old = 0.1 * delta_phi

while abs(delta_phi - delta_phi_old) > 1e-8:
    for n in range(len(y_list)):
        for k in range(len(y_list[n])):
            y1_k = y_list[0][k]
            y2_k = y_list[1][k]
            y3_k = y_list[2][k]
            y4_k = y_list[3][k]
            
            # первые производные функции правдоподобия
            d1_dA1_k = (math.cos(w * k * T + phi_0) * y1_k + math.sin(w * k * T + phi_0) * y2_k) - A1
            d1_dA1_list.append(d1_dA1_k)
            
            d1_dA2_k = (math.cos(w * k * T + phi_0 + delta_phi) * y3_k +\
                        math.sin(w * k * T + phi_0 + delta_phi) * y4_k) - A2
            d1_dA2_list.append(d1_dA2_k)
            
            d1_dw_k  = -A1 * math.sin(w * k * T + phi_0) * k * T * y1_k +\
                        A1 * math.cos(w * k * T + phi_0) * k * T * y2_k -\
                        A2 * math.sin(w * k * T + phi_0 + delta_phi) * k * T * y3_k +\
                        A2 * math.cos(w * k * T + phi_0 + delta_phi) * k * T * y4_k
            d1_dw_list.append(d1_dw_k)
            
            d1_dphi_0_k  = -A1 * math.sin(w * k * T + phi_0) * y1_k +\
                            A1 * math.cos(w * k * T + phi_0) * y2_k -\
                            A2 * math.sin(w * k * T + phi_0 + delta_phi) * y3_k +\
                            A2 * math.cos(w * k * T + phi_0 + delta_phi) * y4_k
            d1_dphi_0_list.append(d1_dphi_0_k)
            
            d1_ddelta_phi_k = -A2 * math.sin(w * k * T + phi_0 + delta_phi) * y3_k +\
                               A2 * math.cos(w * k * T + phi_0 + delta_phi) * y4_k
            d1_ddelta_phi_list.append(d1_ddelta_phi_k)
            
            # вторые и смешанные производные
            d2_dA1     = (-1)/(sigma_n**2)
            
            d2_dA1dA2 = 0
            
            d2_dA1dw_k  = -math.sin(w * k * T + phi_0) * k * T * y1_k +\
                           math.cos(w * k * T + phi_0) * k * T * y2_k
            d2_dA1dw_list.append(d2_dA1dw_k)
            
            d2_dA1dphi_0_k = -math.sin(w * k * T + phi_0) * y1_k +\
                              math.cos(w * k * T + phi_0) * y2_k
            d2_dA1dphi_0_list.append(d2_dA1dphi_0_k)
            
            d2_dA1ddelta_phi = 0
            
            d2_dA2    = (-1)/(sigma_n**2)
            
            d2_dA2dw_k = -math.sin(w * k * T + phi_0 + delta_phi) * k * T * y3_k +\
                          math.cos(w * k * T + phi_0 + delta_phi) * k * T * y4_k
            d2_dA2dw_list.append(d2_dA2dw_k)
            
            d2_A2dphi_0_k = -math.sin(w * k * T + phi_0 + delta_phi) * y3_k +\
                             math.cos(w * k * T + phi_0 + delta_phi) * y4_k
            d2_A2dphi_0_list.append(d2_A2dphi_0_k)
            
            d2_dA2ddelta_phi_k = -math.sin(w * k * T + phi_0 + delta_phi) * y3_k +\
                                  math.cos(w * k * T + phi_0 + delta_phi) * y4_k
            d2_dA2ddelta_phi_list.append(d2_dA2ddelta_phi_k)
            
            d2_dw_k = -A1 * math.cos(w * k * T + phi_0) * ((k * T)**2) * y1_k -\
                       A1 * math.sin(w * k * T + phi_0) * ((k * T)**2) * y2_k -\
                       A2 * math.cos(w * k * T + phi_0 + delta_phi) * ((k * T)**2) * y3_k -\
                       A2 * math.sin(w * k * T + phi_0 + delta_phi) * ((k * T)**2) * y4_k
            d2_dw_list.append(d2_dw_k)
            
            d2_dwdphi_0_k = -A1 * math.cos(w * k * T + phi_0) * k * T * y1_k -\
                             A1 * math.sin(w * k * T + phi_0) * k * T * y2_k -\
                             A2 * math.cos(w * k * T + phi_0 + delta_phi) * k * T * y3_k -\
                             A2 * math.sin(w * k * T + phi_0 + delta_phi) * k * T * y4_k
            d2_dwdphi_0_list.append(d2_dwdphi_0_k)
            
            d2_dwddelta_phi_k = -A2 * math.cos(w * k * T + phi_0 + delta_phi) * k * T * y3_k -\
                                 A2 * math.sin(w * k * T + phi_0 + delta_phi) * k * T * y4_k
            d2_dwddelta_phi_list.append(d2_dwddelta_phi_k)
            
            d2_dphi_0_k = -A1 * math.cos(w * k * T + phi_0) * y1_k -\
                           A1 * math.sin(w * k * T + phi_0) * y2_k -\
                           A2 * math.cos(w * k * T + phi_0 + delta_phi) * y3_k -\
                           A2 * math.sin(w * k * T + phi_0 + delta_phi) * y4_k
            d2_dphi_0_list.append(d2_dphi_0_k)
            
            d2_dphi_0ddelta_phi_k = -A2 * math.cos(w * k * T + phi_0 + delta_phi) * y3_k -\
                                     A2 * math.sin(w * k * T + phi_0 + delta_phi) * y4_k
            d2_dphi_0ddelta_phi_list.append(d2_dphi_0ddelta_phi_k)
            
            d2_ddelta_phi_k = -A2 * math.cos(w * k * T + phi_0 + delta_phi) * y3_k -\
                               A2 * math.sin(w * k * T + phi_0 + delta_phi) * y4_k
            d2_ddelta_phi_list.append(d2_ddelta_phi_k)

    # считаем суммы
    d1_dA1              = 1/(sigma_n**2) * sum(d1_dA1_list)
    d1_dA2              = 1/(sigma_n**2) * sum(d1_dA2_list)
    d1_dw               = 1/(sigma_n**2) * sum(d1_dw_list)
    d1_dphi_0           = 1/(sigma_n**2) * sum(d1_dphi_0_list)
    d1_ddelta_phi       = 1/(sigma_n**2) * sum(d1_ddelta_phi_list)
    d2_dA1dw            = 1/(sigma_n**2) * sum(d2_dA1dw_list)
    d2_dA1dphi_0        = 1/(sigma_n**2) * sum(d2_dA1dphi_0_list)
    d2_dA2dw            = 1/(sigma_n**2) * sum(d2_dA2dw_list)
    d2_A2dphi_0         = 1/(sigma_n**2) * sum(d2_A2dphi_0_list)
    d2_dA2ddelta_phi    = 1/(sigma_n**2) * sum(d2_dA2ddelta_phi_list)
    d2_dw               = 1/(sigma_n**2) * sum(d2_dw_list)
    d2_dwdphi_0         = 1/(sigma_n**2) * sum(d2_dwdphi_0_list)
    d2_dwddelta_phi     = 1/(sigma_n**2) * sum(d2_dwddelta_phi_list)
    d2_dphi_0           = 1/(sigma_n**2) * sum(d2_dphi_0_list)
    d2_dphi_0ddelta_phi = 1/(sigma_n**2) * sum(d2_dphi_0ddelta_phi_list)
    d2_ddelta_phi       = 1/(sigma_n**2) * sum(d2_ddelta_phi_list)
    
    # собираем первые производные в вектор
    L = np.array([d1_dA1, d1_dA2, d1_dw, d1_dphi_0, d1_ddelta_phi])
    
    # собираем вторые и смешанные производные в матрицу
    H = np.array([[d2_dA1,           d2_dA1dA2,        d2_dA1dw,        d2_dA1dphi_0,        d2_dA1ddelta_phi],
                  [d2_dA1dA2,        d2_dA2,           d2_dA2dw,        d2_A2dphi_0,         d2_dA2ddelta_phi],
                  [d2_dA1dw,         d2_dA2dw,         d2_dw,           d2_dwdphi_0,         d2_dwddelta_phi],
                  [d2_dA1dphi_0,     d2_A2dphi_0,      d2_dwdphi_0,     d2_dphi_0,           d2_dphi_0ddelta_phi],
                  [d2_dA1ddelta_phi, d2_dA2ddelta_phi, d2_dwddelta_phi, d2_dphi_0ddelta_phi, d2_ddelta_phi]])
    
    # метод дискриминаторов
    lam_array_old = lam_array
    lam_array     = lam_array - np.dot(L,H)
    
    # обновляем параметры
    A1            = lam_array[0]
    A2            = lam_array[1]
    w             = lam_array[2]
    pho_0         = lam_array[3]
    delta_phi_old = delta_phi
    delta_phi     = lam_array[4]
            
# ошибка оценивания delta_phi
J = -H
D_lambda = inv(J)
D_delta_phi = D_lambda[4,4]

    
# # сигналы с оценками параметров
# S1 = A1 * math.cos(w * k * T + phi_0)
# S1_list.append(S1)
# S2 = A1 * math.sin(w * k * T + phi_0)
# S2_list.append(S2)
# S3 = A2 * math.cos(w * k * T + phi_0 + delta_phi)
# S3_list.append(S3)
# S4 = A2 * math.sin(w * k * T + phi_0 + delta_phi)
# S4_list.append(S4)
    
# # figure_2 = plt.figure(2)
# # plt.plot(range(0, len(S1_list)), S1_list, color = 'deeppink', linewidth = 2)
# # plt.plot(range(0, len(S2_list)), S2_list, color = 'orange', linewidth = 2)
# # plt.plot(range(0, len(S3_list)), S3_list, color = 'gold', linewidth = 2)
# # plt.plot(range(0, len(S4_list)), S4_list, color = 'limegreen', linewidth = 2)
# # plt.xlabel('k')
# # plt.ylabel('Sn(k)')
# # plt.legend(['S1','S2','S3', 'S4'])
# # plt.title('Сигналы с оценками параметров ')
# # plt.grid()
# # plt.show()   




