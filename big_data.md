## 빅분기 실기 준비를 위한 공부

- 데이터 전처리 공부
- 모델은 분류, 회귀를 위주로 공부 (+클러스터링까지?)  
- 데이터는 csv로 제공 -> pd.read_csv()  
- 라이브러리 함수 모두 알아야 함(탭 안되서 철자까지 알아야함)  
- 이론문제도 10문제 나오니까 소홀히 할 수 없음  

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
- 표준화  
```python


```  

- 정규화  
```python


```  

- imputation  
```python


```    

- encoding  
```python


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

- 나이브 베이지안
```python


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
 모델 학습 
'''
print("ROCAUC Score:", roc_auc_score(y_train,pd.DataFrame(model.predict_proba(X_train))))
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