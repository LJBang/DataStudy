# 출력을 원하실 경우 print() 활용
# 예) print(df.head())

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)
import pandas as pd
import sklearn 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

X_train_path = 'data/X_train.csv'
X_test_path = 'data/X_test.csv'
y_train_path = 'data/y_train.csv'
submit_path = '0000.csv'

X_train = pd.read_csv(X_train_path)
X_train = X_train.iloc[:,1:]
X_train['환불금액'] = X_train['환불금액'].fillna(0)

X_test = pd.read_csv(X_test_path)
X_test['환불금액'] = X_test['환불금액'].fillna(0)
X_test_enc = X_test.iloc[:,1:]

y_train = pd.read_csv(y_train_path)
y_train = y_train.iloc[:,1:]

train_df = pd.concat([X_train, y_train],axis=1)
X_enc = pd.get_dummies(X_train)
X_test_enc = pd.get_dummies(X_test_enc)

lack_col = set(X_enc.columns) - set(X_test_enc.columns)
remain_col = set(X_test_enc.columns) - set(X_enc.columns)

for col in lack_col:
	X_test_enc[col]=0

for col in remain_col:
	X_test_enc.drop(col,axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_enc,y_train, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_valid=scaler.transform(X_valid)
X_test_enc=scaler.transform(X_test_enc)

RF=RandomForestClassifier()
RF.fit(X_train,y_train)

predict_proba_rf = RF.predict_proba(X_test_enc)
submit = pd.DataFrame(predict_proba_rf[:,1],columns=['gender'])
submit['cust_id']=X_test['cust_id']
submit = submit[['cust_id','gender']]
submit.to_csv(submit_path, index=False)

print(submit)