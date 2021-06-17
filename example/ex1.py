# 출력을 원하실 경우 print() 활용
# 예) print(df.head())

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)

import pandas as pd
import sklearn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

X_train = pd.read_csv('data/X_train.csv')
X_train = X_train.iloc[:,1:]
X_train['환불금액'] = X_train['환불금액'].fillna(0)

y_train = pd.read_csv('data/y_train.csv')
y_train = y_train.iloc[:,-1]

X_test = pd.read_csv('data/X_test.csv')
X_test = X_test.iloc[:,1:]
X_test['환불금액'] = X_test['환불금액'].fillna(0)

label = pd.concat([X_train['주구매상품'],X_test['주구매상품']])
label = label.unique()
enc = LabelEncoder()
enc.fit(label)
X_train['주구매상품'] = enc.transform(X_train['주구매상품'])
X_test['주구매상품'] = enc.transform(X_test['주구매상품'])

label = pd.concat([X_train['주구매지점'],X_test['주구매지점']])
label = label.unique()
enc = LabelEncoder()
enc.fit(label)
X_train['주구매지점'] = enc.transform(X_train['주구매지점'])
X_test['주구매지점'] = enc.transform(X_test['주구매지점'])

mlp = MLPClassifier(hidden_layer_sizes = (10, ), solver = 'adam', activation = 'relu', learning_rate_init = 0.001, max_iter =500)
mlp.fit(X_train, y_train)

print("ROCAUC Score:", roc_auc_score(y_train,pd.DataFrame(mlp.predict_proba(X_train)).iloc[:,1]))



