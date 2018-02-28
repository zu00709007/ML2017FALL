給定一個人基本個資，我們想要預測這個人的年收入是否超過50000。
個人資料中有age、sex、capital_gain、capital_loss、workclass、education、marital_status、occupation、relationship、race、native_country。
不過為了其中有些欄位是文字，而非數值
因此我們需要將資料使用one hot encoding的方式儲存
'age', 'sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia', '?_native_country'
label為one hot encoding的個人資料，排序方式如上，格式為numpy array。
data為年收入是否大於50000，使用0與1來表示，格式為numpy array。
使用keras配合tensorflow來訓練

logistic_regression.py為傳統機器學習
因為輸出為0與1的二元分類
因此在輸出的最後
我們必須添加sigmoid函數，將數值歸化到0與1之間
因為是分類問題，我們必須以binary_crossentropy來做loss才有意義
最後predict資料時，我們須將大於0.5的值填入1，小於0.5填入0才能完成











