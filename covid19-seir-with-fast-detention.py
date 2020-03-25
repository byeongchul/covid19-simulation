# SEIR model    byeongchul@gmail.com
# semi-implicit Euler method
# refer: https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
# REMARK: The following is an implementation from Christian Hubbs' article.
#         Most of codes are explained by Christian himself.
#         My work is just to write them.
#
import numpy as np
import matplotlib.pyplot as plt

def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

def seir_model_with_fast_detention(init_vals, params, t):
    P_0, S_0, E_0, I_0, Q_0, R_0, D_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    P, Q, D = [P_0], [Q_0], [D_0]
    alpha, beta, gamma, rho, nu, delta, lamda, kappa = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_P = P[-1] + (nu*S[-1])*dt
        next_S = S[-1] - (rho*beta*S[-1]*I[-1] - nu*S[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - (1 - delta)*alpha*E[-1] - delta)*dt  #### <--- here I have to code
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
rho = 0.8 # 1.0 0.2
params = alpha, beta, gamma, rho
# Run simulation
results = seir_model_with_soc_dist(init_vals, params, t)

# Plot results
plt.figure(figsize=(12,8))
plt.plot(results)
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.xlabel('Time Steps')
plt.title(r'SEIR Model with Social Distancing ($\alpha={p[0]}, \beta={p[1]}, \gamma={p[2]}, and \rho={p[3]}$)'.format(p=params))
plt.show()
