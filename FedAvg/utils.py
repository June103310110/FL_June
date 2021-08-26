import numpy as np
import matplotlib.pyplot as plt

def view_clientDict(subset):
    print('>>> 分割給 %d'%len(subset.keys()),'個client')

    _ = list(subset.keys())[0]
    print('>>> subset',_,'資料維度')
    print('-- data shape',np.shape(subset[_][0]), '--label shape',np.shape(subset[_][1]))

    _ = list(subset.keys())[-1]
    print('>>> subset',_,'資料維度')
    print('-- data shape',np.shape(subset[_][0]), '--label shape',np.shape(subset[_][1]))
    
    # plot 
    lis = []
    for _ in subset.keys():
        a = np.shape(subset[_][1])[0]
        lis.append(a)
    plt.figure(figsize=(12, 6))
    
    tick = [key+'_'+str(np.unique(subset[key][1])) for key in list(subset.keys())]
    
    plt.bar(x = tick, height =lis)
    
    plt.xticks(tick, rotation = 45)
    plt.xlabel('client subset datapoints amount')
    plt.show()
    
def byClasses_inbal_split(data, label, K):
    
    if K < len(np.unique(label)):
        raise ValueError('本函式確保同一個subset只有一種類別，clients數量(K)不得小於label classes數量.')
        
    repeat = K//len(np.unique(label))
    left = K%len(np.unique(label))
    
    subset = dict()
    unique = np.unique(label)

    choice = np.random.choice(unique, size = left)
    lis = []
    for i in range(len(unique)):
        cond = np.where(label==i)
        if np.isin(i, choice):

            for _ in np.array_split(cond[0], repeat+np.bincount(choice)[i]):  # 不等分割
                lis.append([data[_], label[_]])
        else:
            for _ in np.array_split(cond[0], repeat):  # 不等分割
                lis.append([data[_], label[_]])
    for _ in lis:
        a = str(len(subset))
        subset['client'+a] = _
    return subset

def bySample_bal_split(data, label, K):
    nums = len(label)//K
    lis = list(range(0,len(label),nums))

    subset = dict()

    for i in range(K-1):
        a, b = lis[i], lis[i+1]
        subset['client'+str(i)] = [data[a:b], label[a:b]]

    a = K-1
    subset['client'+str(a)] = [data[lis[a]:], label[lis[a]:]]
    return subset
