## 빅분기 실기 준비를 위한 공부

- 데이터 전처리 공부
- 모델은 분류, 회귀를 위주로 공부 (+클러스터링까지?)  
- 데이터는 csv로 제공 -> pd.read_csv()  
- 라이브러리 함수 모두 알아야 함(탭 안되서 철자까지 알아야함)  
- 이론문제도 10문제 나오니까 소홀히 할 수 없음  

[예제문제](./example/ex1.py)

### 전략
부분점수만 받자!  
분류, 회귀별 두개 정도만 준비하자  
배깅준비해서 값 비교 후 더 좋은 것 선정 -> 베스트  
텐서플로는 없으니까 사이킷런, 넘파이, pd 정도 +matplotlib?  

---
**함수 정리**  
전처리 중에서 할 수 있는건 함수를 안써도 됨!  
`df.fillna(0)` 같은 것들  

### 데이터 전처리
fit -> transfrom 과정을 거침  
보통은 train set에서 scaler를 먼저 fit 하고  
train, test set 모두에 transform을 적용  

라벨인코딩의 경우 OOV문제가 발생할 수 있기 때문에  
train, test를 합쳐서 fit한 후에 각각의 데이터에 transform하는 것이 좋아보임.  

`fit_transform()`함수는 fit과 transform을 한번에 순서대로 진행  
OR
```python
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(y_train)
```  
위와 같은 방법으로 해도 된다.  

- Standard Scaler  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```  

- MinMaxScaler  
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```  

- imputation  
```python


```    

- Label Encoding  
```python
from sklearn.preprocessing import LabelEncoder

X_train.loc[:,['주구매상품','주구매지점']] = X_train.loc[:,['주구매상품','주구매지점']].apply(LabelEncoder().fit())
```   

### 분류
- 로지스틱 회귀  
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
```  

- 결정트리 -> 랜덤 포레스트  
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = '나무개수', max_depth = '깊이', random_state = '랜덤시드')
rf.fit(X_train, y_train)
```  

- MLP
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes = (10, ), solver = 'adam', activation = 'relu', learning_rate_init = 0.001, max_iter =500)
mlp.fit(X_train, y_train)
```  

### 클러스터링
- K-means  
```python


```  

### 회귀
- 릿지 회귀  
```python
from sklearn.linear_model import Ridge
rid = Ridge(alpha=.5)
rid.fit(X_train, y_train)

rid.coef_ # X의 class별 coef
rid.intercept_ # 절편
```  

- 라쏘 회귀  
```python
from sklearn.linear_model import Lasso
las = Lasso(alpha=.5)
las.fit(X_train, y_train)

las.coef_ # X의 class별 coef
las.intercept_ # 절편
```  

### 앙상블  
- 배깅  
```python


```  

### AUC평가  
```python
from sklearn.metrics import roc_auc_score
'''
 모델 학습 구간
'''
print("ROCAUC Score:", roc_auc_score(y_train,pd.DataFrame(model.predict_proba(X_train)).iloc[:,1]))
```  

### 예측 및 마무리
```python
# 그냥 예측
y_pred = model.predict(X_test)
# 확률형으로 예측 (예제)
y_pred = model.predict_proba(X_test)

# 예측 결과 df로
y_pred = pd.DataFrame(y_pred)

# 예측 결과 csv 형태로 제출
y_pred.to_csv(y_test_path, index = False)
```