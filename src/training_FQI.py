#!/usr/bin/env python
# coding: utf-8

# In[1]:


from FQI_agent import collect_samples, rf_fqi
from env_hiv import HIVPatient
from utils import greedy_action_FQI


# In[2]:


environment = HIVPatient()


# ## 1

# In[5]:


S,A,R,S2,D = collect_samples(environment, int(2e5))
print("nb of collected samples:", S.shape[0])
for i in range(3):
    print("sample", i, "\n  state:", S[i], "\n  action:", A[i], "\n  reward:", R[i], "\n  next state:", S2[i], "\n terminal?", D[i])


# In[4]:


from sklearn.ensemble import RandomForestRegressor
import numpy as np

SA = np.append(S,A,axis=1)
value = R.copy()

Q1 = RandomForestRegressor()
Q1.fit(SA,value);


# In[5]:


print("training MSE:", np.mean((value-Q1.predict(SA))**2))


# ## 2

# In[ ]:


gamma = .9
nb_iter = 150
nb_actions = environment.action_space.n
Qfunctions = rf_fqi(S, A, R, S2, D, nb_iter, nb_actions, gamma)


# In[102]:


import matplotlib.pyplot as plt

# Value of an initial state across iterations
s0,_ = environment.reset()
Vs0 = np.zeros(nb_iter)
for i in range(nb_iter):
    Qs0a = []
    for a in range(environment.action_space.n):
        s0a = np.append(s0,a).reshape(1, -1)
        Qs0a.append(Qfunctions[i].predict(s0a))
    Vs0[i] = np.max(Qs0a)
plt.plot(Vs0)

# Bellman residual
residual = []
for i in range(1,nb_iter):
    residual.append(np.mean((Qfunctions[i].predict(SA)-Qfunctions[i-1].predict(SA))**2))
plt.figure()
plt.plot(residual);


# In[83]:


Qfunctions[0]


# In[103]:


import pickle
import gzip

print(Qfunctions)
with gzip.open('FQI_Q_functions.pkl.gz', 'wb') as file:
    pickle.dump(Qfunctions[40], file)


# In[104]:


with gzip.open('FQI_Q_functions.pkl.gz', 'rb') as f:
    QfunctionsLoaded = pickle.load(f)
print(QfunctionsLoaded)


# In[105]:


s,_ = environment.reset()
for t in range(500):
    a = greedy_action_FQI(QfunctionsLoaded,s,environment.action_space.n)
    s2,r,d,trunc,_ = environment.step(a)
    s = s2
    if d:
        break

