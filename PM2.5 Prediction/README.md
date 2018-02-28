給定前9個小時的PM2.5的觀測數據，我們想要預測第10小時的PM2.5濃度。
label為前9個小時的PM2.5的濃度，格式為numpy array。
data為第10小時的PM2.5的濃度，格式為numpy array。
使用keras配合tensorflow來訓練

gradient_descent.py為傳統機器學習
使用一次線性函數為模型下去訓練

DNN.py為使用神經網路的作法
使用32維→16維→1維的方式

因為是類比資料，因此透過mean squared error來當作loss function是最合適
earlystopping與checkpoint用來驗證validation data避免overfitting並儲存訓練模型
最後使用predict來預測資料
