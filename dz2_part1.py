import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np


# константы
c         = 3e8
w0        = 2 * math.pi * 1602e6
alpha     = 1
T         = 10e-3

q_c_n0_list           = []
N0_list               = []
sigma_11_list         = []
sigma_11_sigma_a_list = []
Ky_Omega_p_list       = []
delta_F_list          = []
delta_F_list_sigma_a  = []

"""------------------------------Меняется с/ш-------------------------------"""
sigma_a   = 10
S_ksi     = 2 * (sigma_a**2) * alpha * ((w0/c)**2) 

for k in range(14, 51, 1):
    q_c_n0 = (10**(0.1*k))
    q_c_n0_list.append(q_c_n0)
    
    N0 = (2/(q_c_n0 * (T**2))) * (1 + (1/(2 * q_c_n0 * T)))
    N0_list.append(N0)
    
    D11_pt2 = (math.sqrt(1 + 2 * math.sqrt((S_ksi)/((alpha**2) * N0 ))) - 1)
    
    D11 = ((alpha * N0)/ 2) * (D11_pt2)
    
    sigma_11 = math.sqrt(D11)
    sigma_11_list.append(sigma_11)
    
    # попытка расчета полосы
    K1 = alpha * (math.sqrt(1 + 2 * math.sqrt((S_ksi)/((alpha**2) * N0))) - 1)
    
    K2 = (K1**2)/2    
    
    def integrand(w):
        p         = (0 + 1j) * w
        K_f       = 1/p * (K1 + K2/(p + alpha))
        K_y_Omega = K_f/(1 + K_f)
        return ((abs(K_y_Omega))**2)
    # считаем интеграл - quad(integrand - интегрируемая функция, 0 - нижний предел, np.inf - верхний беспредел)
    delta_F_pt2 = quad(integrand, 0, np.inf)[0]
    
    delta_F = (1/(2*math.pi)) * delta_F_pt2
    delta_F_list.append(delta_F)

plt.figure(1)
plt.plot(range(14, 51, 1), sigma_11_list, '.-', color = 'deeppink', linewidth = 2)
plt.xlabel('q_c_no, дБГц')
plt.ylabel('σΩ (q_c_no), рад/с')
plt.title('Зависимость среднеквадратической ошибки фильтрации частоты от с/ш')
plt.grid()
plt.show()  

plt.figure(2)
plt.plot(range(14, 51, 1), delta_F_list, '.-', color = 'gold', linewidth = 2)
plt.xlabel('q_c_no, дБГц')
plt.ylabel('ΔF_ЧАП(q_c_no), Гц')
plt.title('Зависимость оптимальной полосы ЧАП от с/ш')
plt.grid()
plt.show()  

"""---------------------------Меняется СКО ускорения------------------------"""
q_c_no_sigma_a = 10**(0.1*30)
N0_sigma_a     = (2 / (q_c_no_sigma_a * (T**2))) * (1 + (1/(2 * q_c_no_sigma_a * T)))

for sigma_a in range(1, 31, 1):
    S_ksi_sigma_a = 2 * (sigma_a**2) * alpha * ((w0/c)**2)
    
    D11_sigma_a_pt2 = (math.sqrt(1 + 2 * math.sqrt((S_ksi_sigma_a)/((alpha**2)*N0_sigma_a))) - 1)
    D11_sigma_a = ((alpha * N0_sigma_a)/2) * D11_sigma_a_pt2
    
    sigma_11_sigma_a  = math.sqrt(D11_sigma_a)
    sigma_11_sigma_a_list.append(sigma_11_sigma_a)
    
    # попытка расчета полосы
    K1_sigma_a = alpha * (math.sqrt(1 + 2 * math.sqrt((S_ksi_sigma_a)/((alpha**2) * N0_sigma_a))) - 1)
    
    K2_sigma_a = (K1_sigma_a**2)/2    
    
    def integrand_sigma_a(w):
        p         = (0 + 1j) * w
        K_f       = 1/p * (K1_sigma_a + K2_sigma_a/(p + alpha))
        K_y_Omega = K_f/(1 + K_f)
        return ((abs(K_y_Omega))**2)
    # считаем интеграл - quad(integrand - интегрируемая функция, 0 - нижний предел, np.inf - верхний беспредел)
    delta_F_pt2_sigma_a = quad(integrand_sigma_a, 0, np.inf)[0]
    
    delta_F_sigma_a = 1/(2*math.pi) * delta_F_pt2_sigma_a
    delta_F_list_sigma_a.append(delta_F_sigma_a)
    
plt.figure(3)
plt.plot(range(1, 31, 1), sigma_11_sigma_a_list, '.-', color = 'darkmagenta', linewidth = 2)
plt.xlabel('σa, м/с^2')
plt.ylabel('σΩ(σa), рад/с')
plt.title('Зависимость среднеквадратической ошибки фильтрации частоты от σa')
plt.grid()
plt.show()    

plt.figure(4)
plt.plot(range(1, 31, 1), delta_F_list_sigma_a, '.-', color = 'c', linewidth = 2)
plt.xlabel('σa, м/с^2')
plt.ylabel('ΔF_ЧАП(σa), Гц')
plt.title('Зависимость оптимальной полосы ЧАП от σa')
plt.grid()
plt.show()    
