import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D

def plot_V_and_Bellman_residuals(environment, s0, Q2, Q, SA):
    # Value of an initial state across iterations
    Qs0a = []
    for a in range(environment.action_space.n):
        s0a = np.append(s0, a).reshape(1, -1)
        Qs0a.append(Q2.predict(s0a))
    Vs0 = np.max(Qs0a)

    # Bellman residual
    residual = np.mean((Q2.predict(SA) - Q.predict(SA)) ** 2)

    return Vs0, residual


def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, environment, disable_tqdm=False):
    s0, _ = environment.reset() #for evaluation
    Vs0_list = []
    residuals_list = []

    nb_samples = S.shape[0]
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
            Q = RandomForestRegressor(n_jobs=-1)
            Q.fit(SA, value)
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Q.predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2

        if iter >= 1:
            Q_ = RandomForestRegressor(n_jobs=-1)
            Q_.fit(SA, value)

            Vs0, residual = plot_V_and_Bellman_residuals(environment, s0, Q_, Q, SA)
            Vs0_list.append(Vs0)
            residuals_list.append(residual)
            
            Q = Q_

    plt.figure()
    plt.plot(Vs0_list)
    plt.figure()
    plt.plot(residuals_list)

    return Q