# The Role of Cross-Silo Federated Learning in Facilitating Data

source: https://arxiv.org/abs/2104.07468
## 問題設定/背景
農業用，基於電腦視覺(衛星影像)，進行指定區域內的收穫量預測。
- 收穫前的衛星串列影像預測黃豆產量
- 基於參與者的角度用一些方法保護隱私
    - both imaging (remote sensing) [13] and tabular (weather and soil data)
- As our aim is to propose technological solutions to facilitate and subsequently begin to build confidence in data sharing

### dataset 
- Nasa衛星影像 DAAC2015
    - `J. You, X. Li, M. Low, D. Lobell, S. Ermon, Deep gaussian process for crop yield predic-tion based on remote sensing data, in: Proceedings of the AAAI Conference on Arti cial Intelligence, Vol. 31, 2017.` 
- 表格型?
    - `S. Khaki, L. Wang, S. V. Archontoulis, A cnn-rnn framework for crop yield prediction, Fron-tiers in Plant Science 10 (2020) 1750. `


## 創意/貢獻/差異化
- 這個論文跟13差異在於，模型經由多個獨立的silo dataset訓練。資料分割的方法是把資料集D依據美國各州切分，藉此產生11(N)個silos。(彼此不互通)

### 貢獻
1. We demonstrate the applicability of federated and model sharing machine learning method-ologies to enable training of distributed datasets in the settings relevant to the agri-food domain. 
    - 在農業領域實現聯邦學習
2. We show that the necessary privacy and security concerns prevalent in the agri-food sector can be appropriately overcome via privacy preserving methods, in our case differential privacy. 
    - d在農業部門視為必須保密隱私的考量可以被避免，基於隱私保護方法，在我們的方法中是透過differential privacy技巧。
3. We argue for the potential adoption of the proposed technological methods to facilitate data sharing, and give key example use cases where such facilitation would benefit all participants. 
    - 資料共享的可能，創造範例表示共享對參與者全員都有好處。


### 做了一些use-case example
-  Production optimization for collaborative federations. 
-  Analysis of client production from a distribution source. 
-  Regulatory analysis of data from a central governing body. 

## 技巧
### 衛星影像
- 影像轉成影像直方圖
    - 影像是光譜，在這邊把影像依據band分成9個，然後分別做成直方圖。最後concatenated起來。

### differential privacy
- 抵擋惡意攻擊，例如inference attack，where the aim is to extract raw data or sensitive information from the shared/communicated models
    - Our proposition mainly focuses on the increasingly important method of differential privacy [48] to combat these attacks at train time. 
- differential privacy在shared model內加入不確定性的概念，並藉此掩蓋掉來自獨立參與者的貢獻，並基於此保證從參數中還原的資訊受到限制。
    - 使用DP-SGD
    - [差分隱私 hackmd](https://hackmd.io/KhCskHrqSUivzcRaAlU1bA?view)
    - [tensorflow 原生工具](https://github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial.py)
    - Section 7 regarding optimal parameters of the differential privacy mechanism including


### SiloUpdate (include DP)
![](https://i.imgur.com/W1JGX8i.png =240x)
- 實作上可能是在梯度上加上高斯雜訊。 `DP-SGD`

### FedBN(federated batch normalization)
- https://arxiv.org/abs/2102.07623
    - We propose an efficient and effective learning strategy denoted FedBN. Similar to FedAvg, FedBN performs local updates and averages local models. However, FedBN assumes local models have BN layers and excludes their parameters from the averaging step.
    - 假設模型裡面有BN層，在aggregation的時候因為BN依賴於local dataset，所以不要參與聚合，讓每個local client能夠差異化以符合資料分布。
