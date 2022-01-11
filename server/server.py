#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@param {type:"boolean"}


# In[1]:


# gitURL = 'https://gitlab.aiacademy.tw/junew/federated_aia_test.git'
# account = 'at102091:12345678'

with open('../../FL_June/account.cfg', 'r') as f:
    r = f.read()
    
    dic = {}
    for i in r.splitlines():
        i = [item.strip() for item in i.split('=')]
#         print(i.split('='))
        dic[i[0]] = i[1]
    print(dic)
gitURL = dic['gitURL']
account = dic['account']
repo_name = 'june-federated-server'


# In[2]:


import os
if os.getcwd().split('/')[-2:] == ['FL_June', 'server']:
    os.popen('git clone https://{account}@{gitURL}'.format(account=account,
                                                           gitURL=gitURL.split('//')[-1])).read()
    print('clone to ',os.getcwd())
else:
    print(os.getcwd())


# In[3]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Activation,
                                     BatchNormalization, Flatten,
                                     Conv2D, MaxPooling2D)
import tensorflow as tf
import numpy as np
import sys
from datetime import date
sys.path.append('june-federated-server')
from utils import compressed_cpickle, decompress_cpickle


# In[ ]:





# ### control_key (optional)

# In[4]:


contro_key = {}
contro_key['new_model'] = False # default to False


# ## 移動到federated_aia_test floder
# 如果不存在，請先執行最上面的git clone

# In[5]:


print(os.getcwd())
if os.getcwd().split('/')[-2:] == ['server', 'june-federated-servert']:
    pass
else:
    os.chdir('../../FL_June/server/june-federated-server')
    os.getcwd()


# ## 建立新的初始化global model 
# > 只有當模型不存在、或者你更新了架構、打算重新訓練的時候

# In[6]:


def simplecnn():
    # 選擇 Keras 的 API 寫法
    inputs = Input(shape=(28,28,1))
#     inputs = inputs
    # 第一層
    # 建立卷積層，設定32個3*3的filters
    # 設定ReLU為激活函數。
    x = Conv2D(32, (3, 3), activation='relu')(inputs)

    # 第二層 - 卷積層 + 池化層
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 第三層 - 卷積層
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # 第四層 - 卷積層 + 池化層
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 建立分類模型 (MLP) : 平坦層 + 輸出層 (10)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    cnn_model = Model(inputs=inputs, outputs=outputs)
    return cnn_model


# In[7]:


from datetime import date
import pathlib

today = date.today()
print(today)

path = '../saved_model'
pathlib.Path(f'{path}').mkdir(parents=True, exist_ok=True)     
lis = os.listdir(path)
lis = [i for i in lis if i.__contains__('global_model')]
print(lis)


# In[8]:


if contro_key['new_model'] == True:
    import os
    import shutil
    from datetime import date
    
    # 保留舊模型到server本機 (saved_model被登錄在.gitignore)
    os.makedirs('../saved_model', exist_ok=True)
    if os.listdir().__contains__('global_model.pbz2'):
        today = date.today()
        lis = os.listdir('../saved_model')
        lis = [i for i in lis if i.__contains__('global_model')]
        
        new_model_name = 'global_model_{ver}_{today}.pbz2'.format(today=today, ver=len(lis))
        shutil.move('./global_model.pbz2', '../saved_model/'+new_model_name)
        print('Move global model to FL_June/server/saved_model')
    else:
        pass
    model = simplecnn()

    # weight、架構 (json) to pickle

    model_attri = {'weights':model.get_weights(), 'json':model.to_json()}
    
    print('create new model')
    compressed_cpickle('./global_model', model_attri)

    print('update gitrepo global_model.pbz2')
    run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]
    cmd_lis = '''git add .
    git commit -m'global model complete aggregate and update to g'
    git push https://{account}@{gitURL}
    '''.format(account=account, gitURL=gitURL.split('//')[-1])
    
    
    run_cmd(cmd_lis)


# ## 下載各個branch中的模型壓縮檔

# In[9]:


r = os.popen('git pull').read()
print(r)

os.popen('git remote update origin --prune')
lis = os.popen('git branch -r').read().split('\n')[:-1]


all_client_branch = [i for i in  lis if not i.__contains__('master')]

print(all_client_branch)

# 判斷各個分支是否有更新
if len(all_client_branch) <= 0:
    raise ValueError('June: No clients appear')


# In[10]:


run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]

for i in all_client_branch:
    origin_branch_name = i.lstrip()
    filename = origin_branch_name.split('/')[-1]+'.pbz2'
    
    print(origin_branch_name, filename)
    
    
    cmd_lis = '''git checkout remotes/{branch_name} {model_attri_pbz2}
    '''.format(branch_name = origin_branch_name, model_attri_pbz2=filename)
    result = os.popen(cmd_lis).read()
    print(result)


# In[11]:


get_ipython().system('git branch -a')


# ## 聚合並更新global model

# In[12]:


model_attri = decompress_cpickle('./global_model.pbz2')
global_model = tf.keras.models.model_from_json(model_attri['json'])

lis = [i for i in os.listdir() if i.__contains__('pbz2')]
lis.remove('global_model.pbz2')
print(lis)


# In[13]:


weights = []
for i in lis:
    model_attri = decompress_cpickle(i)
    weights.append(model_attri['weights'])
print(np.shape(weights))


# In[14]:


new_weights = list()
if len(weights) == 0:
    print('no new client to aggregate')
    pass
elif len(weights) == 1:
    print('only single participant')
    new_weights = weights[0]
else:
    for i in zip(*weights):
        new_weights.append(tf.reduce_sum(i, axis=0))
        
global_model.set_weights(model_attri['weights'])


# In[15]:


model_attri = {'weights':global_model.get_weights(), 'json':global_model.to_json()}

compressed_cpickle('./global_model', model_attri)


# In[16]:


import os

run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]

cmd_lis = '''mv ./global_model.pbz2 ../
rm *.pbz2
mv ../global_model.pbz2 ./
git add .
git commit -m'global model complete aggregate and update to Gmodel'
git push https://{account}@{gitURL}
'''.format(account=account, gitURL=gitURL.split('//')[-1])


run_cmd(cmd_lis)
branch = os.popen('git branch -a').read()
print(branch)


# In[ ]:




