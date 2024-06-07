
## Title
### Forecast demand for bicycles based on specific climates
--- 
Members: 
```
김민석, 컴퓨터학부, kminseok5167@gmail.com
엄규현, 해양융합공학과, gheom24@gmail.com
김현민, 산업경영공학과, jihiyo7@hanyang.ac.kr
최준영, 전자공학부, jychoihy0507@naver.com
```

## I. Proposal 

**Why are you doing this?**

마이크로 모빌리티는 최근 친환경 교통 수단으로 주목받고 있으며, 많은 기업과 도시들이 이에 대한 관심을 높이고 있다. 자전거는 전통적인 마이크로 모빌리티 수단으로서 환경 보호와 도시 교통 관리에 중요한 역할을 한다. 자전거 사용은 탄소 배출을 줄이고 교통 혼잡을 완화하며 에너지를 절약하는 데 기여한다. 그러나 이러한 혜택을 최대한 누리기 위해서는 자전거의 적절한 배치와 관리가 필요하다. 이를 위해서는 정확한 수요 예측이 필수적이다. 정확한 수요 예측은 자전거 대여 시스템의 효율성을 높이고 사용자 경험을 개선하는 데 중요한 역할을 한다.

**What do you want to see at the end?**

목표는 자전거 수요 예측 모델을 개발하여 도시 교통 관리와 환경 보호에 기여하는 것이다. 이 예측 모델은 자전거 사용 패턴을 분석하고 미래의 수요를 예측하여 자전거 인프라 운영과 관리를 최적화할 수 있게 한다. 이를 통해 자전거가 필요한 위치에 적시에 배치되어 사용자가 편리하게 이용할 수 있도록 지원한다. 자전거 사용을 촉진하여 자동차 사용을 줄임으로써 탄소 배출을 감소시키고, 도로의 차량 수를 줄여 교통 혼잡을 완화하며, 자전거의 높은 에너지 효율로 전체 에너지 소비를 줄이는 데 기여한다. 이를 통해 지속 가능한 도시 교통을 실현하고 환경 보호에 이바지하며, 더 깨끗하고 살기 좋은 도시를 만드는 데 일조할 수 있다.

--- 
## II. Datasets

이 데이터셋은 자전거 대여 시스템의 시간별 기록을 나타낸다. 각 열은 특정 시간에 대해 다양한 특성 및 대여 기록을 포함하고 있다.

```
1. datetime: 자전거 대여 기록이 발생한 날짜와 시간.(예: 2011-01-01 00:00:00)
2. season: 계절 (1: 겨울, 2: 봄, 3: 여름, 4: 가을)
3. holiday: 공휴일 여부 (0: 공휴일 아님, 1: 공휴일)
4. workingday: 평일 여부 (0: 평일 아님, 1: 평일)
5. weather: 날씨 상황
    - 1: 맑음, 약간 흐림, 부분적으로 흐림
    - 2: 안개, 안개 + 흐림, 안개 + 부분적 흐림
    - 3: 가벼운 눈, 가벼운 비 + 천둥
    - 4: 심한 비, 얼어붙는 비, 천둥, 눈 + 안개
6. temp: 실제 기온(섭씨)
7. atemp: 체감 기온(섭씨)
8. humidity: 습도(%)
9. windspeed: 풍속 (mph)
10. casual: 등록되지 않은 사용자가 대여한 자전거 수
11. registered: 등록된 사용자가 대여한 자전거 수
12. count: 총 대여된 자전거 수
```

### 데이터 전처리 
datetime을 year, month, day, hour, minute, second 열로 분리 시킨다. 
월이나 시간 정보가 결과 값에 영향을 미칠 수 있다고 생각하여 아래와 같이 분리 했다. 

```python
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train.shape
```
<img width="1078" alt="스크린샷 2024-05-31 오후 1 34 25" src="https://github.com/msk226/Bike-Sharing-Demand/assets/77945998/9cd3dc0d-c0da-4de0-ab72-d807a386e227">



--- 


## III. Methodology 

이번 프로젝트는 주어진 시계열 데이터와 특정 기후 상황을 기반으로 자전거 대여량을 예측하는 회귀(Regression)문제이다. 트레이닝 데이터셋에는 실제 자전거 대여량(count)이 포함되어 있어 이를 통해 모델을 학습시키고, 학습된 모델을 사용하여 테스트 데이터셋의 대여량을 예측한다. 따라서, 본 프로젝트는 지도 학습(Supervised Learning) 접근 방식을 사용한다.

자전거 대여 수요 예측은 여러 특성 간의 복잡한 상호작용을 포함한다. 다층 퍼셉트론 신경망은 이러한 복잡하고 비선형적인 관계를 효과적으로 모델링할 수 있다. 우리는 PyTorch를 이용하여 다층 퍼셉트론 신경망 모델을 구축하였으며, 이 모델은 다음과 같은 특성들을 입력으로 받아 자전거 대여 수요를 예측한다:

1. 시계열 특성: 연도, 요일, 시간
2. 기후 특성: 계절, 공휴일 여부, 근무일 여부, 날씨 상태, 실제 온도, 체감 온도, 습도, 풍속

이 특성들은 자전거 대여 패턴에 중요한 영향을 미치는 요소들이며, 이를 통해 모델의 예측 성능을 향상시키기 위해 선택되었다.

---

## IV. Evaluation & Analysis

### 연관 관계 파악 
연/월/일 별 대여량을 파악하기 위해 다음과 같이 그래프로 나타냈다. 
``` python
figure.set_size_inches(18,8)

sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)
sns.barplot(data=train, x="minute", y="count", ax=ax5)
sns.barplot(data=train, x="second", y="count", ax=ax6)

ax1.set(ylabel='Count',title="Rental volume by year")
ax2.set(xlabel='month',title="Rental volume by month")
ax3.set(xlabel='day', title="Rental volume by day")
```

---<img width="919" alt="스크린샷 2024-05-31 오후 1 40 37" src="https://github.com/msk226/Bike-Sharing-Demand/assets/77945998/47c37667-2095-4b24-9334-251faac7e6a7">

2012년의 대여량이 2011년보다 더 많은 이유는 자전거 대여 시스템이 시간이 지남에 따라 점점 더 인기를 끌었기 때문이라고 추측할 수 있다. 

월별 대여량을 살펴보면, 6월에 대여량이 가장 많으며 7월에서 10월 사이에도 대여량이 높은 편인데, 이는 날씨가 따뜻하고 자전거 타기 좋은 계절이기 때문이다. 반면, 1월에는 날씨가 추워 자전거 이용이 적어 대여량이 가장 낮다. 

일별 대여량 데이터는 1일부터 19일까지의 정보만 포함되어 있고, 나머지 날짜의 데이터는 test.csv 파일에 있기 때문에 이 데이터를 분석에 사용하면 정확한 예측을 할 수 없다. 

시간대별 대여량을 보면 출퇴근 시간에 대여량이 증가하는 경향이 있지만, 주말과 비교해서 분석할 필요가 있는데, 이는 주중과 주말의 생활 패턴이 다르기 때문이라고 유추할 수 있다. 또한, 분과 초의 값은 모두 0으로 설정되어 있어 예측에 사용하기 어렵다고 생각 된다. 

---

근무일 여부와 대여량의 관계이다. 
```python
# 시간대별 대여량 그래프
plt.figure(figsize=(12, 6))
sns.pointplot(data=train, x='hour', y='count', hue='day_of_week')
plt.title('Hourly Rental Count by Day of the Week')
plt.xlabel('Hour of the Day')
plt.ylabel('Rental Count')
plt.legend(title='Day of the Week', loc='upper left')
plt.xticks(rotation=45)
plt.show()

```
![output](https://github.com/msk226/Bike-Sharing-Demand/assets/80662948/c5cd2ec0-84ec-4260-8056-2357dd5de618)


**근무일(주중)**에는 출퇴근 시간대에 대여량이 두드러지게 증가하는 패턴이 나타난다. 특히 오전 8시와 오후 5시~6시에 대여량이 급격히 증가하는데, 이는 많은 사람들이 출근과 퇴근 시에 자전거를 이용하는 경향이 있기 때문이다.
**근무하지 않는 날(주말)**에는 이러한 시간적 피크가 없고, 대여량이 하루 종일 비교적 균일하게 분포되어 있다. 이는 사람들이 시간적 제약 없이 자전거를 이용하기 때문이다.
평균 대여량을 비교해보면, 근무일의 전체 평균 대여량이 근무하지 않는 날보다 높으며, 특히 출퇴근 시간대의 대여량 증가가 주요 원인이다.

또한, 각 시간대별 대여량의 평균과 표준편차를 분석하면, 근무일의 특정 시간대(출퇴근 시간대)에 대여량의 변동성이 더 크다. 이는 많은 사람들이 특정 시간대에 집중적으로 자전거를 이용하기 때문이다.

그래프의 범례를 보면, 색상으로 요일을 구분하고 있으며, 각각의 색상은 다음을 나타낸다:

0: 일요일
1: 월요일
2: 화요일
3: 수요일
4: 목요일
5: 금요일
6: 토요일

이 분석을 통해, 근무일과 근무하지 않는 날의 자전거 대여 패턴의 차이를 명확히 이해할 수 있으며, 이를 기반으로 자전거 대여 시스템의 운영을 최적화할 수 있다.

---

다음은 온도 & 체감 온도가 대여량에 어떤 영향을 미치는지 알아보기 위해 시각화 한 것이다. 
온도와 체감 온도를 반올림 해서 시각화를 진행 했다. 

``` python
# 온도와 체감온도 반올림
train['temp_rounded'] = train['temp'].round()
train['atemp_rounded'] = train['atemp'].round()

# 반올림된 온도와 체감온도의 평균 대여량 계산
temp_atemp_mean = train.groupby(['temp_rounded', 'atemp_rounded'])['count'].mean().unstack()

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(temp_atemp_mean, cmap="YlGnBu", annot=True, fmt=".1f")
plt.title('Heatmap of Rounded Temperature and Feels-like Temperature vs Rental Count')
plt.xlabel('Feels-like Temperature (Rounded)')
plt.ylabel('Temperature (Rounded)')
plt.show()
```

<img width="824" alt="스크린샷 2024-05-31 오후 1 57 10" src="https://github.com/msk226/Bike-Sharing-Demand/assets/77945998/7f64db80-0afd-4697-af59-8d9065cdfff5">

다음 그래프를 통해 비교적 온도가 높은 날에 대여량이 더 높아지는 것을 알 수 있다. 겨울보다는 여름에 대여량이 많다는 것도 연관 관계가 있음을 알 수 있다. 

--- 

마지막으로, 풍속과 대여량의 관계에 대해서도 그래프로 표현 했다. 

더욱 관계를 쉽게 파악하기 위해 풍속을 구간별로 나누어 시각화를 진행 했다. 

```python

# 풍속을 구간별로 나누기
train['windspeed_interval'] = pd.cut(train['windspeed'], bins=np.arange(0, train['windspeed'].max() + 5, 10), right=False)

# 각 구간별 대여량의 평균 계산
windspeed_count_mean = train.groupby('windspeed_interval')['count'].mean()

# 바 그래프로 시각화
plt.figure(figsize=(10, 6))
windspeed_count_mean.plot(kind='bar', color='skyblue')
plt.title('Average Rental Count by Windspeed Interval')
plt.xlabel('Windspeed Interval')
plt.ylabel('Average Rental Count')
plt.xticks(rotation=45)
plt.show()
```
<img width="862" alt="스크린샷 2024-05-31 오후 2 03 55" src="https://github.com/msk226/Bike-Sharing-Demand/assets/77945998/99b210aa-feb9-4430-8383-d1050f780a84">

바람이 너무 강하거나 너무 약할 때보다 적당한 바람이 부는 날에 자전거 대여량이 상대적으로 높아짐을 알 수 있다.

## 모델 학습 코드 

### 데이터 준비
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

train = pd.read_csv(f'train.csv', parse_dates=["datetime"])
test = pd.read_csv(f'test.csv', parse_dates=["datetime"])
submission = pd.read_csv(f'sampleSubmission.csv')
```
필요한 모듈을 import 하고, 학습 및 테스트 데이터 셋을 불러온다. 

```python
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
```
데이터 전처리를 위해 datetime 열에서 날짜와 시간 관련 특징들을 추출한다. 
```python
cols = test.columns[1:]
```
테스트 데이터의 열 중 첫 번째 열을 제외한 나머지 열들을 cols로 정의한다. 

```python
# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X_train_scaled, np.log1p(train["count"]), test_size=0.2, random_state=42)

# 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.values.reshape(-1, 1))
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid.values.reshape(-1, 1))
X_test = torch.FloatTensor(X_test_scaled)

```
학습 데이터를 학습용 데이터와 검증용 데이터로 분할하고 파이토치 모델에서 사용할 수 있도록 데이터를 텐서로 변환한다.
- train_test_split을 사용해 훈련 데이터를 훈련 세트와 검증 세트로 분할한다.
- test_size=0.2는 20%의 데이터를 검증 세트로 사용함을 의미한다.
- random_state=42는 분할을 재현 가능하게 한다.
- np.log1p를 사용해 target 값 (count)을 로그로 변환한다. 이는 target 값의 분포를 정규화하여 모델의 성능을 향상시키기 위함이다.

### 모델 정의 
```python
class MultivariateLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

```
위 클래스는 입력층, 은닉층 (ReLU 활성화 함수 포함) 및 출력층을 포함하는 신경망 모델을 정의한다.
각 층의 노드 수는 256, 256, 128이다.

```python
model = ImprovedModel(input_size=X_train.shape[1], output_size=1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
모델 객체를 생성하고 손실 함수와 옵티마이저를 설정한다

```python
# 모델 학습
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X_train)
    
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}]')

```
모델을 학습시키는 루프다. 총 2000 에포크 동안 학습을 진행하고, 매 100 에포크마다 현재 손실을 출력한다. 

```python
with torch.no_grad():
    model.eval()
    y_pred_valid = model(X_valid)
    print(f'Test Loss: {loss_fn(y_pred_valid, y_valid).item():.4f}')


```
검증 데이터로 모델의 성능을 평가한다. 

```python
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

y_true = y_valid.numpy().squeeze()
y_pred = y_pred_valid.numpy().squeeze()
rmsle_score = rmsle(np.expm1(y_true), np.expm1(y_pred))
print(f'RMSLE: {rmsle_score:.4f}')
```
RMSLE (Root Mean Squared Logarithmic Error)를 계산하여 모델의 예측 성능을 측정한다. 

```python
with torch.no_grad():
    outputs = model(X_test)
    y_predict = outputs.squeeze().numpy()

submission["count"] = np.expm1(y_predict)
file_name = f"submit_improved_{rmsle_score:.5f}.csv"
submission.to_csv(file_name, index=False)
print(pd.read_csv(file_name).head(2))

```
테스트 데이터셋에 대해 예측을 수행하고, 결과를 submission 데이터프레임에 저장하여 CSV 파일로 생성한다. RMSLE 점수를 파일 이름에 포함시킨다.


## 결과 분석 

### 1. 손실 값 
```
Epoch [100/2000], Loss: 1.2554]
Epoch [200/2000], Loss: 0.5998]
Epoch [300/2000], Loss: 0.2701]
Epoch [400/2000], Loss: 0.1512]
Epoch [500/2000], Loss: 0.1169]
Epoch [600/2000], Loss: 0.0972]
Epoch [700/2000], Loss: 0.0819]
Epoch [800/2000], Loss: 0.0735]
Epoch [900/2000], Loss: 0.0663]
Epoch [1000/2000], Loss: 0.0627]
Epoch [1100/2000], Loss: 0.0574]
Epoch [1200/2000], Loss: 0.0542]
Epoch [1300/2000], Loss: 0.0517]
Epoch [1400/2000], Loss: 0.0493]
Epoch [1500/2000], Loss: 0.0469]
Epoch [1600/2000], Loss: 0.0452]
Epoch [1700/2000], Loss: 0.0469]
Epoch [1800/2000], Loss: 0.0426]
Epoch [1900/2000], Loss: 0.0402]
Epoch [2000/2000], Loss: 0.0405]
```
각 에포크(epoch)에서 출력된 손실 값이 점진적으로 감소하고 있다.  모델이 학습을 통해 성능이 향상되고 있다는 것을 알 수 있다. 

### 2. 검증 손실 
검증 데이터에 대한 손실 값을 확인하여 모델이 과적합(overfitting)되지 않았는지 판단한다. 
```
Test Loss: 0.1311
```
학습이 완료된 후 검증 데이터셋에서의 손실 값은 0.1311로 계산된다. 이는 모델이 훈련되지 않은 데이터에 대해 얼마나 잘 예측하는지를 보여준다.

```
RMSLE: 0.3621
```
최종 RMSLE 값은 0.3621이다. RMSLE는 예측 값과 실제 값 사이의 차이를 로그 스케일에서 측정한 지표로, 값이 낮을수록 모델의 예측 성능이 좋음을 의미한다. 

Kaggle의 "Bike Sharing Demand" 대회 리더보드와 비교했을 때, 상위 5% 이내에 속하는 결과를 얻었다. 

---

## V. Related Work (e.g., existing studies)

Dateset : https://www.kaggle.com/c/bike-sharing-demand/overview

Reference code 
1. https://github.com/corazzon/KaggleStruggle/blob/master/bike-sharing-demand/bike-sharing-demand-EDA.ipynb
2. https://github.com/corazzon/KaggleStruggle/blob/master/bike-sharing-demand/bike-pytorch.ipynb
3. https://velog.io/@danniel1025/%EB%8B%A4%EC%B8%B5%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0MLP



---

## VI. Conclusion: Discussion
우리가 만든 모델은 자전거 수요 예측을 위한 도구로서 중요한 역할을 할 수 있을 것으로 보인다. 과거 자전거 대여 패턴과 주어진 환경 조건(온도, 바람 속도 등)을 기반으로 미래의 자전거 대여량을 예측할 수 있고, 이러한 예측은 도시의 자전거 인프라 운영과 관리를 최적화하는 데 도움을 준다. 

미래의 자전거 수요를 예측할 수 있음으로써, 도시는 자전거를 사용하는 시민들이 필요로 하는 곳에 자원을 효율적으로 할당할 수 있다. 예를 들어, 예측된 대여량이 높은 지역에는 더 많은 자전거를 배치하여 사용자들이 필요할 때 언제든지 이용할 수 있도록 한다. 반면에 예측된 대여량이 낮은 지역에서는 자원을 다른 곳에 더 효율적으로 이동시켜 자원 낭비를 최소화할 수 있다.

또한, 날씨 뿐 만 아니라 지역의 특징 같은 정보를 FEATURE로 추가 한다면 더 좋은 성능의 모델을 개발할 수 있을 것으로 추측된다. 

## VII. Role 
```
김민석 : 코드 작성, 데이터 분석, 보고서 작성, 깃허브 관리
엄규현 : 코드 작성, 데이터 분석, 보고서 작성
최준영 : 보고서 작성
김현민 : 발표 영상 촬영

```
