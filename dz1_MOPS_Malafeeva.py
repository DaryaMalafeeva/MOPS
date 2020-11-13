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
import os

# если однажды создали файл - то удаляем его чтобы не писать в него же
os.remove('D:\Malafeeva_DD\MOPS\out_parameters.txt')
os.remove('D:\Malafeeva_DD\MOPS\out_derivatives.txt')

"""---------------------------Обработка файла-------------------------------"""
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
M       = 2048
sigma_n = 10

"""---------------------Начальные параметры алгоритма-----------------------"""
# начальные значения параметров вектора lam_array
A1                  = 7010
A2                  = 3505
f                   = f_s/(234.6-44.93)
w                   = 2 * math.pi * f
phi_0               = math.radians(-60)
delta_phi           = math.radians(130)
lam_array           = np.array([A1, A2, w, phi_0, delta_phi])
delta_phi_old       = 0.1 * delta_phi

S1_list             = []
S2_list             = []
S3_list             = []
S4_list             = []
d1_dA1              = 0
d1_dA2              = 0
d1_dw               = 0
d1_dphi_0           = 0 
d1_ddelta_phi       = 0
d2_dA1dw            = 0
d2_dA1dphi_0        = 0
d2_dA2dw            = 0
d2_A2dphi_0         = 0
d2_dA2ddelta_phi    = 0
d2_dw               = 0
d2_dwdphi_0         = 0
d2_dwddelta_phi     = 0
d2_dphi_0           = 0
d2_dphi_0ddelta_phi = 0
d2_ddelta_phi       = 0
while_count         = 0 



"""---------------------Алгоритм оценивания параметров----------------------"""
while(abs(delta_phi - delta_phi_old)>1e-8):
    while_count+=1
    for k in range(0, M, 1):
        y1_k = y1_list[k]
        y2_k = y2_list[k]
        y3_k = y3_list[k]
        y4_k = y4_list[k]
         
        # синусы и косинусы
        sin_A1 = math.sin(w * k * T + phi_0)
        sin_A2 = math.sin(w * k * T + phi_0 + delta_phi)
        cos_A1 = math.cos(w * k * T + phi_0)
        cos_A2 = math.cos(w * k * T + phi_0 + delta_phi)
        
        # первые производные функции правдоподобия
        d1_dA1_k = cos_A1 * y1_k + sin_A1 * y2_k - A1
        d1_dA1 += d1_dA1_k
        
        d1_dA2_k = cos_A2 * y3_k + sin_A2 * y4_k - A2
        d1_dA2 += d1_dA2_k
        
        d1_dw_k  = -A1 * sin_A1 * k * T * y1_k + A1 * cos_A1 * k * T * y2_k -\
                    A2 * sin_A2 * k * T * y3_k + A2 * cos_A2 * k * T * y4_k
        d1_dw += d1_dw_k
        
        d1_dphi_0_k  = -A1 * sin_A1 * y1_k + A1 * cos_A1 * y2_k -\
                        A2 * sin_A2 * y3_k + A2 * cos_A2 * y4_k
        d1_dphi_0 += d1_dphi_0_k
        
        d1_ddelta_phi_k = -A2 * sin_A2 * y3_k + A2 * cos_A2 * y4_k
        d1_ddelta_phi += d1_ddelta_phi_k
        
        # вторые и смешанные производные
        d2_dA1dw_k = -sin_A1 * k * T * y1_k + cos_A1 * k * T * y2_k
        d2_dA1dw += d2_dA1dw_k
        
        d2_dA1dphi_0_k = -sin_A1 * y1_k + cos_A1 * y2_k
        d2_dA1dphi_0 += d2_dA1dphi_0_k
        
        d2_dA2dw_k = -sin_A2 * k * T * y3_k + cos_A2 * k * T * y4_k
        d2_dA2dw += d2_dA2dw_k
        
        d2_A2dphi_0_k = -sin_A2 * y3_k + cos_A2 * y4_k
        d2_A2dphi_0 += d2_A2dphi_0_k
        
        d2_dA2ddelta_phi_k = -sin_A2 * y3_k + cos_A2 * y4_k
        d2_dA2ddelta_phi += d2_dA2ddelta_phi_k
        
        d2_dw_k = -A1 * cos_A1 * k**2 * T**2 * y1_k - A1 * sin_A1 * k**2 * T**2 * y2_k -\
                   A2 * cos_A2 * k**2 * T**2 * y3_k - A2 * sin_A2 * k**2 * T**2 * y4_k
        d2_dw += d2_dw_k
        
        d2_dwdphi_0_k = -A1 * cos_A1 * k * T * y1_k - A1 * sin_A1 * k * T * y2_k -\
                         A2 * cos_A2 * k * T * y3_k - A2 * sin_A2 * k * T * y4_k
        d2_dwdphi_0 += d2_dwdphi_0_k
        
        d2_dwddelta_phi_k = -A2 * cos_A2 * k * T * y3_k - A2 * sin_A2 * k * T * y4_k
        d2_dwddelta_phi += d2_dwddelta_phi_k
        
        d2_dphi_0_k = -A1 * cos_A1 * y1_k - A1 * sin_A1 * y2_k -\
                       A2 * cos_A2 * y3_k - A2 * sin_A2 * y4_k
        d2_dphi_0 += d2_dphi_0_k
        
        d2_dphi_0ddelta_phi_k = -A2 * cos_A2 * y3_k - A2 * sin_A2 * y4_k
        d2_dphi_0ddelta_phi += d2_dphi_0ddelta_phi_k
        
        d2_ddelta_phi_k = -A2 * cos_A2 * y3_k - A2 * sin_A2 * y4_k
        d2_ddelta_phi += d2_ddelta_phi_k
    
    d1_dA1              *= 1/(sigma_n**2) 
    d1_dA2              *= 1/(sigma_n**2) 
    d1_dw               *= 1/(sigma_n**2) 
    d1_dphi_0           *= 1/(sigma_n**2)
    d1_ddelta_phi       *= 1/(sigma_n**2) 
    
    d2_dA1               =-M/(sigma_n**2)
    d2_dA1dA2            = 0
    d2_dA1dw            *= 1/(sigma_n**2) 
    d2_dA1dphi_0        *= 1/(sigma_n**2) 
    d2_dA2dw            *= 1/(sigma_n**2)
    d2_dA1ddelta_phi     = 0
    d2_dA2               =-M/(sigma_n**2)
    d2_A2dphi_0         *= 1/(sigma_n**2) 
    d2_dA2ddelta_phi    *= 1/(sigma_n**2) 
    d2_dw               *= 1/(sigma_n**2) 
    d2_dwdphi_0         *= 1/(sigma_n**2) 
    d2_dwddelta_phi     *= 1/(sigma_n**2) 
    d2_dphi_0           *= 1/(sigma_n**2) 
    d2_dphi_0ddelta_phi *= 1/(sigma_n**2) 
    d2_ddelta_phi       *= 1/(sigma_n**2) 
    
    
    # пишем в файл производные
    file_drv = open('out_derivatives.txt', 'a')
    file_drv.write('Итерация №' + str(while_count))
    file_drv.write('\n')
    file_drv.write('Значения производных:')
    file_drv.write('\n')
    file_drv.write('d1_dA1=' + str(d1_dA1))
    file_drv.write('\n')
    file_drv.write('d1_dA2=' + str(d1_dA2))
    file_drv.write('\n')
    file_drv.write('d1_dw=' + str(d1_dw))
    file_drv.write('\n')
    file_drv.write('d1_dphi_0=' + str(d1_dphi_0))
    file_drv.write('\n')
    file_drv.write('d1_dphi_0=' + str(d1_dphi_0))
    file_drv.write('\n')
    file_drv.write('d1_ddelta_phi=' + str(d1_ddelta_phi))
    file_drv.write('\n')
    file_drv.write('d2_dA1=' + str(d2_dA1))
    file_drv.write('\n')
    file_drv.write('d2_dA1dA2=' + str(d2_dA1dA2))                                                                                
    file_drv.write('\n')
    file_drv.write('d2_dA1dw=' + str(d2_dA1dw))
    file_drv.write('\n')
    file_drv.write('d2_dA1dphi_0=' + str(d2_dA1dphi_0)) 
    file_drv.write('\n')
    file_drv.write('d2_dA2dw=' + str(d2_dA2dw)) 
    file_drv.write('\n')
    file_drv.write('d2_dA1ddelta_phi=' + str(d2_dA1ddelta_phi))    
    file_drv.write('\n')
    file_drv.write('d2_dA2=' + str(d2_dA2))    
    file_drv.write('\n')
    file_drv.write('d2_A2dphi_0=' + str(d2_A2dphi_0)) 
    file_drv.write('\n')
    file_drv.write('d2_dA2ddelta_phi=' + str(d2_dA2ddelta_phi)) 
    file_drv.write('\n')
    file_drv.write('d2_dw=' + str(d2_dw))     
    file_drv.write('\n')
    file_drv.write('d2_dwdphi_0=' + str(d2_dwdphi_0))         
    file_drv.write('\n')
    file_drv.write('d2_dwddelta_phi=' + str(d2_dwddelta_phi))      
    file_drv.write('\n')
    file_drv.write('d2_dphi_0=' + str(d2_dphi_0))        
    file_drv.write('\n')
    file_drv.write('d2_dphi_0ddelta_phi=' + str(d2_dphi_0ddelta_phi))  
    file_drv.write('\n')
    file_drv.write('d2_ddelta_phi=' + str(d2_ddelta_phi)) 
    file_drv.write('\n')
    file_drv.write('-------------------------------------------------------------')
    file_drv.write('\n')
    file_drv.close()

    L = np.array([d1_dA1, d1_dA2, d1_dw, d1_dphi_0, d1_ddelta_phi])
    
    H = np.array([[d2_dA1,           d2_dA1dA2,        d2_dA1dw,        d2_dA1dphi_0,        d2_dA1ddelta_phi],
                  [d2_dA1dA2,        d2_dA2,           d2_dA2dw,        d2_A2dphi_0,         d2_dA2ddelta_phi],
                  [d2_dA1dw,         d2_dA2dw,         d2_dw,           d2_dwdphi_0,         d2_dwddelta_phi],
                  [d2_dA1dphi_0,     d2_A2dphi_0,      d2_dwdphi_0,     d2_dphi_0,           d2_dphi_0ddelta_phi],
                  [d2_dA1ddelta_phi, d2_dA2ddelta_phi, d2_dwddelta_phi, d2_dphi_0ddelta_phi, d2_ddelta_phi]])
    
    lam_array     = lam_array - np.dot(L,H)
        
    # обновляем параметры
    # (вне цикла for потому что в нем параметры сигналов постоянны)
    A1            = lam_array[0]
    A2            = lam_array[1]
    w             = lam_array[2]
    phi_0         = lam_array[3]
    delta_phi_old = delta_phi
    delta_phi     = lam_array[4]
    
    # пишем в файл оценки параметров сигнала
    file = open('out_parameters.txt', 'a')
    file.write('Итерация №' + str(while_count))
    file.write('\n')
    file.write('Оценки параметров сигнала:')
    file.write('\n')
    file.write('A1=' + str(A1))
    file.write('\n')
    file.write('A2=' + str(A2))
    file.write('\n')
    file.write('w=' + str(w))
    file.write('\n')
    file.write('phi_0=' + str(phi_0))
    file.write('\n')
    file.write('delta_phi=' + str(delta_phi))
    file.write('\n')
    file.write('-------------------------------------------------------------')
    file.write('\n')
    file.close()
    
    # # сигналы с оценками параметров
    # S1 = A1 * math.cos(w * k * T + phi_0)
    # S1_list.append(S1)
    # S2 = A1 * math.sin(w * k * T + phi_0)
    # S2_list.append(S2)
    # S3 = A2 * math.cos(w * k * T + phi_0 + delta_phi)
    # S3_list.append(S3)
    # S4 = A2 * math.sin(w * k * T + phi_0 + delta_phi)
    # S4_list.append(S4)
    
    # очищаем счетчики перед следующим заходом в while
    d1_dA1              = 0
    d1_dA2              = 0
    d1_dw               = 0
    d1_dphi_0           = 0 
    d1_ddelta_phi       = 0
    d2_dA1dw            = 0
    d2_dA1dphi_0        = 0
    d2_dA2dw            = 0
    d2_A2dphi_0         = 0
    d2_dA2ddelta_phi    = 0
    d2_dw               = 0
    d2_dwdphi_0         = 0
    d2_dwddelta_phi     = 0
    d2_dphi_0           = 0
    d2_dphi_0ddelta_phi = 0
    d2_ddelta_phi       = 0
    
J = -H
D_lambda = inv(J)
D_delta_phi = D_lambda[4,4]
        
# figure_2 = plt.figure(2)
# plt.plot(range(0, len(S1_list)), S1_list, color = 'deeppink', linewidth = 2)
# plt.plot(range(0, len(S2_list)), S2_list, color = 'orange', linewidth = 2)
# plt.plot(range(0, len(S3_list)), S3_list, color = 'gold', linewidth = 2)
# plt.plot(range(0, len(S4_list)), S4_list, color = 'limegreen', linewidth = 2)
# plt.xlabel('k')
# plt.ylabel('Sn(k)')
# plt.legend(['S1','S2','S3', 'S4'])
# plt.title('Сигналы с оценками параметров ')
# plt.grid()
# plt.show()   
