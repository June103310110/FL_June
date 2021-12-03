#!/usr/bin/env python
# coding: utf-8

# In[1]:


gitURL = 'https://gitlab.aiacademy.tw/junew/federated_aia_test.git'
account = 'at102091:12345678'


# In[2]:


import os
if os.getcwd().split('/')[-2:] == ['FL_June', 'server']:
    os.popen('git clone -b master https://{account}@{gitURL}'.format(account=account,
                                                           gitURL=gitURL.split('//')[-1])).read()

    print(gitURL.split('//')[-1], 'cloning to ',os.getcwd())
else:
    print(os.getcwd())


# In[3]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Activation,
                                     BatchNormalization, Flatten,
                                     Conv2D, MaxPooling2D)
import tensorflow as tf
import numpy as np

from federated_aia_test.utils import compressed_cpickle, decompress_cpickle


# In[4]:


from IPython import get_ipython
import os
import argparse

print(get_ipython().__class__.__name__)
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    pass
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_model", help="if initialize a new model or not, [default]:False",
                       default = False, required=False)
    args = parser.parse_args()
    print(vars(args))


# ### control_key (optional)

# In[5]:


contro_key = {}
if 'args' in locals():
    for i in vars(args).keys():
        contro_key[i] = vars(args)[i]
else:
    contro_key['new_model'] = False
    pass


# ## 移動到federated_aia_test floder
# 如果不存在，請先執行最上面的git clone

# In[6]:


print(os.getcwd())
if os.getcwd().split('/')[-2:] == ['server', 'federated_aia_test']:
    pass
else:
    os.chdir('../../FL_June/server/federated_aia_test')
    os.getcwd()
print(os.getcwd())


# ## 建立新的初始化global model 
# > 只有當模型不存在、或者你更新了架構、打算重新訓練的時候

# In[7]:


print(contro_key['new_model'])


# In[8]:


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


# In[9]:


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


# ## 選擇本輪參與的參與者，並更新round和training.cfg

# In[10]:


contro_key['C'] = 1


# In[11]:


import random
C = contro_key['C']

r = os.popen('git pull').read()
print(r)

os.popen('git remote update origin --prune')
lis = os.popen('git branch -r').read().split('\n')[:-1]


all_client_branch = [i.strip().split('/')[-1] for i in lis if not i.__contains__('master')]

print(all_client_branch)

participate_branch = random.sample(all_client_branch, int(C*len(all_client_branch)))
print(participate_branch)


# In[12]:


with open('./training.cfg', mode='r', encoding='UTF-8') as f:
    r = f.readlines()
    item_dict = {}
    for item in r:
        key, value = item.split('=')
        item_dict[key] = value.split('\n')[0]

item_dict


# In[13]:


rounds = int(item_dict['round'])+1
print(rounds)
print(participate_branch)


# In[14]:


with open('./training.cfg', mode='w+', encoding='UTF-8') as f:
    f.writelines('round={rounds}\n'.format(rounds=rounds))
    f.writelines('trainKey={participate_branch}\n'.format(participate_branch='/'.join(all_client_branch)))


# ## 下載各個branch中的模型壓縮檔

# In[15]:


r = os.popen('git pull').read()
print(r)

os.popen('git remote update origin --prune')
lis = os.popen('git branch -r').read().split('\n')[:-1]


all_client_branch = [i for i in  lis if not i.__contains__('master')]

print(all_client_branch)

# 判斷各個分支是否有更新
if len(all_client_branch) <= 0:
    raise ValueError('June: No clients appear')


# In[16]:


run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]

for i in all_client_branch:
    origin_branch_name = i.lstrip()
    filename = origin_branch_name.split('/')[-1]+'.pbz2'
    
    print(origin_branch_name, filename)
    
    
    cmd_lis = '''git checkout remotes/{branch_name} {model_attri_pbz2}
    '''.format(branch_name = origin_branch_name, model_attri_pbz2=filename)
    result = os.popen(cmd_lis).read()
    print(result)


# ## 聚合並更新global model

# In[17]:


model_attri = decompress_cpickle('./global_model.pbz2')
global_model = tf.keras.models.model_from_json(model_attri['json'])

lis = [i for i in os.listdir() if i.__contains__('pbz2')]
lis.remove('global_model.pbz2')
print(lis)


# In[18]:


weights = []
for i in lis:
    model_attri = decompress_cpickle(i)
    weights.append(model_attri['weights'])
print(np.shape(weights))


# In[19]:


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


# In[20]:


model_attri = {'weights':global_model.get_weights(), 'json':global_model.to_json()}

compressed_cpickle('./global_model', model_attri)


# In[21]:


import os

run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]

cmd_lis = '''mv ./global_model.pbz2 ../
rm *.pbz2
mv ../global_model.pbz2 ./
git add .
git commit -m'global model complete aggregate and update to Gmodel'
git push https://{account}@{gitURL}
'''.format(account=account, gitURL=gitURL.split('//')[-1])


result = run_cmd(cmd_lis)
for i in result:
    print(i)
branch = os.popen('git branch -a').read()
print(branch)


# In[22]:


from IPython import get_ipython
import os

print(get_ipython().__class__.__name__)
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    os.popen('ipython nbconvert --to script ../server.ipynb')


# In[ ]:




