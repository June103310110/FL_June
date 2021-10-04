# FederatedLearning_JuneToolbox
聯邦學習工具包


## Tensorflow聯邦學習入門簡介

在這一系列專文中，我們會逐步的從問題設定，經典方法以及實作方法等層面慢慢深入聯邦學習的世界。另外，實作儘管Tensorflow上但我不會馬上分享Tensorflow Federated(TFF)的API，會盡可能使用Tensorflow基本框架以及python、numpy實現演算法，實現通用性。
這邊暫且整理了聯邦學習的關鍵字，如果需要自修或練習的朋友可以從關鍵字列表中做一些調查。
`Federated Learning; Non-IID data; Model aggregation; Communication cost; Security and issues; Homomorphic encryption; Data silo; FedAvg;`

![image](https://user-images.githubusercontent.com/32012425/135801492-bf5391e4-6236-4f2a-a28d-c772c3e9825d.png)
> source: Google-AI-Blog: Your phone personalizes the model locally, based on your usage (A). Many users’ updates are aggregated (B) to form a consensus change © to the shared model, after which the procedure is repeated.

## 聯邦學習的問題設定
聯邦學習(Federated Learning)的原始發想情境(源自Google)是手機上的模型訓練。但歐盟有隱私協定(GDPR)，其他國家也對行動裝置上的資料收集有規範，因此廠商不能橫刀奪取用戶資料來訓練模型。儘管有政策限制，但大型企業因為資料隱私而被告上法院的案例仍時有所聞(FB被美國告、Google加州訴訟)。
那有沒有辦法資料不出手機，但還是能訓練一個模型呢？可以的，我把模型架構和預訓練權重給你，你自己在local端微調(Fine-tune)重新訓練你自己的模型。
但有沒有辦法你自己訓練好後，把你的經驗分享給大家呢？可以的，這就是聯邦學習。這讓我們可以建立更強健(Robust)並通用(General)的模型。
你只需要把經驗傳給大家，不需要把資料上傳，這些經驗也許會經過加密或其他手段保障你的資訊/資料不會從經驗中被反向萃取出來。
這邊提到被傳送的「經驗」可能是深度學習中的梯度或者你訓練後的權重，又或者在機器學習中以決策樹家族(e.g. XGBoost, AdaBoost)CART機制下的Gini-table。
這些經驗被傳送真的是安全的嗎?
在大部分的情況下，基本上是安全的，尤其是在加入防範機制，例如加入一些雜訊或差分隱私方法(Differential privacy)進行訓練(DP-SGD)或者乾脆進行同態加密後，更是難以反向攻擊。
當然也有研究者在討論關於惡意參與者也參與聯邦學習的話，會造成什麼影響並提出防範方法。可以參考以下論文：
[1] [Deep Leakage from Gradients , MIT Song Han et.al , 2019](https://arxiv.org/abs/1906.08935)
[2] [How To Backdoor Federated Learning, Cornell University Eugene Bagdasaryan et.al ,2018](https://arxiv.org/pdf/1807.00459.pdf)

## 不同使用者的資料大不相同，為什麼能訓練起來?
首先要先理解資料大不相同在這裡的意義，精確得來說，這是關於不同的裝置中的資料(或稱Data silos)之間Non-IID的問題：
Non-IID的意思是非獨立同分布，在聯邦學習的條件下主要是非同分布。以Youtube頻道的瀏覽紀錄舉例，我的瀏覽紀錄可能充滿JoJo和鯊鯊，但你可能沒看過任何日本動畫。這時候拿我們的資料來訓練模型，通常是會壞掉的。
根據Communication-Efficient Learning of Deep Networks from Decentralized Data這篇聯邦學習創始論文提到的狀況為例，聯邦學習遇到的Non-IID狀況主要分成4種；
使用者資料的非獨立同分布：特定的用戶數據(相當於被抽樣的子資料集)不能代表整體(母體)
使用者資料量的不平衡： 用戶端的資料量數量有差異，有的人資料多有的人資料少，比如我每天看Youtube 30分鐘，你整整看了23小時。這種情況下，直接訓練資料量多的人會統治(dominate)整個模型訓練。
用戶數量遠大於用戶的平均資料量： 雖然只有在手機上會有這種情境，但他表示的是參與聯邦學習的用戶數量，遠大於用戶們平均擁有的資料量。
客戶端設備通訊有限制：客戶端的設備可能不適合參與長時間訓練。以手機為例，你可能只有在wifi環境下才能訓練，還有可能突然沒電。要逼你參與長時間訓練成本很昂貴(甚至不可能)。同時也包括你的資料量太大(高解析度串流影像、或高頻率收集的工具機感測器數據)，要全部上雲非常困難。Communication Cost

## 在這些限制下，為什麼能訓練起來呢
在不同應用會考慮不同的Non-IID情況，為了克服上面四點，聯邦學習領域不斷推出新的方法，這邊解釋以一個最簡單的方法流程：
初始化全域權重。
將全域權重分配給每個用戶端，讓他們有相同的初始權重。
隨機抽選一部分(e.g. 50%)的用戶端參與這一輪的聯邦學習。
用戶端各自訓練幾個Epochs後，把梯度(或權重)上傳。
Server端依據被抽選的用戶端各自的資料數量，對上傳的梯度(或權重)加權平均。平均後得到新的全域權重。
重複以上流程，直到基於全域權重的模型達到停止訓練標準。
一步一步來看：
步驟2，是確保所有客戶端共用相同起點，這是聚合後模型有意義的基礎。否則起點不同，得到的梯度聚合起來沒有意義。
步驟3，是基於用戶數量遠大於用戶的平均資料量的條件下，減少使用者資料的非獨立同分布、使用者資料量的不平衡帶來的影響。比如說軍營裏面有1000個人，大家每天都只吃1碗飯，只有我一天吃12碗飯，那如果長官每天抽20%的人檢查飯量，我被抽到的機率也只有20%，也就是極端值參與到訓練內的機率只有20%。藉此我的存在就不太明顯，保證了模型的通用性，並且不會因為我而產生Over-fitting。同時50%就能盡可能掌握大多數用戶端的資料分布(盡可能接近母體)。
步驟4，是減少通訊成本。不需要每個Epochs都聚合一次。
步驟5，資料多的用戶端還是應該受重視，所以如果我被抽到，他就會發現我是個一天吃12碗飯的人，這個發現讓長官很震驚，相較其他只吃1碗飯的人，會更大程度影響了他的決策(模型)。
上面這些就是起源論文Communication-Efficient Learning of Deep Networks from Decentralized Data的演算法內容，可以從下面連結去參考，之後我會發一篇文章讓各位能夠實作。
[3] [Communication-Efficient Learning of Deep Networks from Decentralized Data, HB McMahan et.al, 2016](https://arxiv.org/abs/1602.05629)


### ref: 
[1] [Deep Leakage from Gradients , MIT Song Han et.al , 2019](https://arxiv.org/abs/1906.08935)
[2] [How To Backdoor Federated Learning, Cornell University Eugene Bagdasaryan et.al ,2018](https://arxiv.org/pdf/1807.00459.pdf)
[3] [Communication-Efficient Learning of Deep Networks from Decentralized Data, HB McMahan et.al, 2016](https://arxiv.org/abs/1602.05629)
