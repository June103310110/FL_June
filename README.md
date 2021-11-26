# FL system teaching_sample
聯邦學習工具包

## Abstract 
在Teaching sample的branch中，我利用台灣人工智慧學校的Gitlab(`aia_federated_test`)作為梯度/模型的中轉站，並將執行程序分成`server`、`client`兩個路徑分別對Gitlab進行存取。
- 在server資料夾內，你可以將來自不同branch的模型權重聚合，並更新`global_model.pbz2`並設定`training.cfg`用來控制client的工作狀態與網路拓樸。
- 在client資料夾內，你需要先執行`setup.py`建立環境並下載`aia_federated_test`的repo，`setup.py`會幫你新建一個金鑰以及建立與金鑰相同命名的git branch並幫你checkout過去，接著你可以用`client_train.ipynb`的程式碼訓練模型，模型的架構與權重來自`master` branch的`global_model.pbz2`，資料則是從MNIST中隨機採樣的1000筆資料(應該只包含1~2個類別)，完成訓練後會產生新的模型壓縮檔，檔名為`金鑰.pbz2`，並在Jupyter Notebook的最後會幫你上傳到你的branch中，server端會主動到各個branch中尋找最新的模型權重。

### 本教學不完善之處
- 通常應該為每一個參與者設定一個gitlab帳號，我這邊為了方便大家練習先使用一個共用的測試帳號。
- 這個教學提供一個聯邦系統的架構，但目前還沒有實現自動化的訓練與聚合。
- 有很多新的演算法可以應用在模型訓練和聚合上，在模型訓練上，通常可以客製化optimizer或loss function，模型聚合則可以用一些技巧來更精準的加權平均所有資料。但本範例中沒有太多細節，你可以參考本github `main` branch下的example folder，裡面有一些能夠參考的實現

> 大家一起來聯邦學習模型吧
