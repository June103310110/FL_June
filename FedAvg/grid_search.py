#!/usr/bin/env python
# coding: utf-8

# # grid search 
# https://blog.gtwang.org/programming/tensorflow-object-detection-api-multiple-gpu-parallel-training/

# In[1]:


rounds = 100
C = [1, 0.5, 0.1]
K = [12, 50, 100, 1000]
E = [1,3,7]
mu = [1.0,0.0]


# In[ ]:


import pandas as pd
import os

directory = './data'



lis = []
for c_ in C:
    for k_ in K:
        for e_ in E:
            for mu_ in mu:
                cmd = 'python E_FedProx.py --rounds '+str(rounds)+' --k '+str(k_)+' --c '+str(c_)+' --e '+str(e_)+' --mu '+str(mu_)
        #         cmd = 'python E_FedProx.py -h'
                os.system(cmd)
    #                 print(cmd)
                df_loc = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
                lis.append([c_, k_, rounds, e_, mu,  df_loc])


# In[4]:


import numpy as np
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

df = pd.DataFrame()
a = np.transpose([lis])

to_list = lambda a, i: np.squeeze(a[i], axis = 1)
to_list(a, 0)
df['C'] = to_list(a, 0)
df['K'] = to_list(a, 1)
df['rounds'] = to_list(a, 2)
df['epochs'] = to_list(a, 3)
df['mu'] = to_list(a,4)
df['df_loc'] = to_list(a, 5)

df.to_csv('grid_search/grid_search_'+timestr+'.csv', index=False)


# In[ ]:




