import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random
#import dz3_filter as filter_no_ins

"""-----------------------------Параметры моделирования---------------------"""
T        = 10  * 1e-3 
T_d      = 0.2 * 1e-6
mod_time = 20               # в секундах
c        = 3 * 1e8
w0       = 2 * math.pi * 1602 * 1e6
w_p      = 2 * math.pi * 2    * 1e6
q_c_n0   = 10 ** (0.1 * 30)
t_list   = []
N        = int(T/T_d)

"""-----------------------Моделируем шум------------------------------------"""
# формирующий шум пси
sigma_psi = 0.5

# формирующий шум хи
sigma_delta = 1
alpha_delta = 1
S_hi        = 2 * (sigma_delta**2) * alpha_delta 
sigma_hi    = math.sqrt((S_hi)/(2*T))

# шум наблюдения
sigma_n   = 35.4

"""---------------------------Инициализация--------------------------------"""
# начальные значения истинных параметров
a_true          = 1
phi_true        = math.pi/12
Omega_true      = 100
delta_true      = 1

# начальные значения скорректированных оценок
a_corr          = 0.5
phi_corr        = 0
Omega_corr      = 0
delta_corr      = 0

# матрицы фильтра
X_true   = np.array([[a_true],\
                     [phi_true],\
                     [Omega_true],\
                     [delta_true]])

X_corr   = np.array([[a_corr],\
                      [phi_corr],\
                      [Omega_corr],\
                      [delta_corr]])                                     # 4x1
    

C        = np.array([[1, 0, 0, 0],\
                     [0, 1, 0, 0]])                                  # 1x2

D_x_corr = np.array([[(0.3)**2, 0, 0, 0],\
                     [0, (math.pi)**2, 0, 0],\
                     [0, 0, 34**2, 0],\
                     [0, 0, 0, 1**2]])                            # 4x4
   
G        = np.array([[T,         0],\
                     [0,         0],\
                     [0,         0],\
                     [0, alpha_delta * T]])                                # 4x2

D_ksi    = np.array([[sigma_psi**2,            0],\
                     [           0, sigma_hi**2]])                  # 2x2

    
D_n      = np.array([[sigma_n**2]])                                  # 1x1

F        = np.array([[1, 0, 0,                     0],\
                     [0, 1, T,                     0],\
                     [0, 0, 1,         (-(w0)/(c)*T)],\
                     [0, 0, 0, (1 - alpha_delta * T)]])                # 4x4

W12      = 0
W21      = 0
    
k   = 0
t_k = 0
t_k_list         = []
Epsilon_phi_list = []
D_phi_max_list   = []
D_phi_min_list   = []
S_sin_list       = []
phi_p_list       = []
delta_corr_list  = []
a_rad_true_list  = []
D_phi_ins_list   = []

gamma = 0

while t_k < mod_time:
    t_k +=T
    t_k_list.append(t_k)
    
    """----------------------Входное воздействие----------------------------"""
    psi_hi = np.array([[random.normalvariate(0,sigma_psi)],\
                       [random.normalvariate(0,sigma_hi)]]) 
    
    X_true  = F.dot(X_true) + G.dot(psi_hi)
    
    # амплитуда
    if t_k < 5:
        X_true[0][0] = 1
    elif t_k >= 5:
        X_true[0][0] = 0.5 
    
    nu_true    = gamma - ((w0)/(c)) * X_true[3][0]
    
    a_rad_true = nu_true * ((c)/(w0))
    
    a_rad_true_list.append(a_rad_true)
        

    """----------------------------Экстраполяция----------------------------"""
    X_extr       = F.dot(X_corr)                                   # 4x1
    
    D_x_extr_pt1 = (F.dot(D_x_corr)).dot(F.transpose())            # 4x4

    D_x_extr_pt2 = (G.dot(D_ksi)).dot(G.transpose())               # 4x4

    D_x_extr     = D_x_extr_pt1 + D_x_extr_pt2                     # 4x4

    W11          = (N)/(2 * sigma_n**2)
    
    W22          = (N *((X_extr[0][0])**2))/(2 * sigma_n**2)
    
    W            = np.array([[W11, W12],\
                             [W21, W22]])
            
    """--------------------------Коррелятор---------------------------------"""
    # фаза на w_p
    phi_p        = np.array(range(0, N, 1)) * T_d * w_p
    
    n_list  = np.random.normal(loc = 0.0, scale = sigma_n * 1, size = N)
    
    y            = X_true[0][0] * np.cos(phi_p + X_true[1][0]) + n_list
    
    S_sin        = np.sin(phi_p + X_extr[1][0])
    
    S_cos        = np.cos(phi_p + X_extr[1][0])
    
    I            = np.sum(np.dot(y, S_cos))
    
    Q            = np.sum(np.dot(y, S_sin))
    
    U_1          = I * (1/D_n) - ((X_extr[0][0] * N)/(2 * D_n))
    
    U_2          = Q * ((-X_extr[0][0])/(D_n))
    
    U            = np.array([U_1[0],\
                             U_2[0]])
        
    """------------------------------Коррекция------------------------------"""
    
    D_x_corr     = inv(inv(D_x_extr) + ((C.transpose().dot(W)).dot(C)))  # 4x4        
    
    X_corr       = X_extr + ((D_x_corr.dot(C.transpose())).dot(U))       # 4x1
    
    k+=1
    
    # мгновенная ошибка фильтрации фазы
    Epsilon_phi = math.degrees(X_corr[1][0] - X_true[1][0])
    Epsilon_phi_list.append(Epsilon_phi)
    # предельные границы ошибок фильтрации фазы по уровню 3 сигма
    D_phi_max =  math.degrees(3 * math.sqrt(D_x_corr[1][1]))
    D_phi_max_list.append(D_phi_max)
    D_phi_min = math.degrees(-3 * math.sqrt(D_x_corr[1][1]))
    D_phi_min_list.append(D_phi_min)
    
    delta_corr_list.append(X_corr[3][0])
    
    
    
    
    print('--------------')
    print('Шаг №' + str(k))
    print('X_corr = ' + str(X_corr))
    print('             ')
    

"""----------------------Сохранение и вывод результатов---------------------"""

# # изменение дисперсии ошибки фазы

# D_phi_before = 15.1505
# D_phi_after  = 29.2124

# ins_gain_before = filter_no_ins.D_phi_before/D_phi_before
# ins_gain_after  = filter_no_ins.D_phi_after/D_phi_after


# # выигрыш в помехоустойчивости (по последней оценке дисперсии ошибки)
# delta_gain_dB = 10 * math.log10(filter_no_ins.D_x_corr[1][1]/D_x_corr[1][1])


plt.figure(1)
plt.plot(t_k_list, Epsilon_phi_list, '.-', color = 'blueviolet', linewidth = 1)
plt.plot(t_k_list, D_phi_max_list, '.-', color = 'red', linewidth = 1)
plt.plot(t_k_list, D_phi_min_list, '.-', color = 'red', linewidth = 1)
plt.xlabel('t, с')
plt.ylabel('Epsilon_phi(t), град/с')
plt.title('')
plt.grid()
plt.show() 

plt.figure(2)
plt.plot(t_k_list, delta_corr_list, '.-', color = 'blueviolet', linewidth = 1)
plt.plot(t_k_list, a_rad_true_list, '.-', color = 'hotpink', linewidth = 1)
plt.xlabel('t, с')
plt.ylabel('δ(t), ν(t) * c/w0, м/с^2')
plt.title('Погрешность измерения рад. ускорения от ИНС и истинное рад. ускорение')
plt.legend(['δ(t)','ν(t) * c/w0'])
plt.grid()
plt.show() 

