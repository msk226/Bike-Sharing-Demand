
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

마이크로 모빌리티는 최근 친환경적인 교통 수단으로 인식되며 많은 기업과 도시가 이에 대한 관심을 높이고 있다. 자전거는 전통적인 마이크로 모빌리티 수단으로서, 환경 보호와 도시 교통 관리에 중요한 역할을 할 수 있다. 자전거 사용은 탄소 배출을 줄이고, 교통 혼잡을 완화하며, 에너지 절약에 기여한다. 그러나 이러한 혜택을 극대화하려면 자전거의 적절한 배치와 관리가 필수적이다. 이를 위해서는 정확한 수요 예측이 필요하다. 이를 통해 자전거 대여 시스템의 효율성을 높이고, 사용자 경험을 향상시킬 수 있다.

**What do you want to see at the end?**

목표는 자전거 수요 예측 모델을 개발하여 도시 교통 관리와 환경 보호에 기여하는 것이다. 이 예측 모델은 자전거 사용 패턴을 분석하고 미래의 수요를 예측하여 자전거 인프라의 운영과 관리를 최적화하는 데 도움을 줄 것이다. 이를 통해 자전거가 필요할 때 적절한 위치에 배치되어 사용자들이 편리하게 이용할 수 있도록 하고, 자전거 사용을 촉진하여 자동차 사용을 줄임으로써 탄소 배출을 감소시키며, 도로의 차량 수를 줄여 교통 혼잡 문제를 완화하고, 자전거의 높은 에너지 효율로 전체 에너지 소비를 줄이는 데 기여할 수 있다. 지속 가능한 도시 교통을 실현하여 환경 보호에 기여하고, 더 깨끗하고 살기 좋은 도시를 만드는 데 일조할 수 있을 것이다.

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
sns.pointplot(data=train, x='hour', y='count', hue='workingday')
plt.title('Hourly Rental Count by Working Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Rental Count')
plt.legend(title='Working Day', loc='upper left')
plt.xticks(rotation=45)
plt.show()
```
<img width="899" alt="스크린샷 2024-05-31 오후 2 05 49" src="https://github.com/msk226/Bike-Sharing-Demand/assets/77945998/cee9c9e3-65f7-48e0-b967-469710b57847">

근무하는 날과 근무하지 않는 날의 자전거 대여량이 명확히 차이가 나타난다. 근무일에는 출퇴근 시간에 많은 대여량이 발생하는 반면, 근무하지 않는 날에는 이러한 시간적 패턴이 뚜렷하지 않다.

근무일에는 주로 출퇴근 시간대에 대여량이 높은데, 이는 아마도 근무하는 사람들이 출근하고 퇴근할 때 자전거를 이용하는 경향이 있기 때문일 것으로 추측된다. 반면 근무하지 않는 날에는 이러한 시간적 제약이 없으므로, 대여량의 분포가 더 균일해 보인다.

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

또한, 풍속과 대여량의 관계에 대해서도 그래프로 표현 했다. 

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

풍속이 아주 낮거나, 높은 날에 비해서 적당한 바람이 부는 날에 자전거 대여량이 비교적 높아짐을 알 수 있다. 


--- 


## III. Methodology 

이번 프로젝트는 시계열 데이터와 특정 기후 상황이 주어졌을 때 자전거 대여량을 예측하는 문제로, 이는 회귀(Regression) 문제에 해당한다. 트레이닝 데이터셋에는 실제 자전거 대여량(count)이 포함되어 있어 이를 통해 모델을 학습시키고, 학습된 모델을 사용하여 테스트 데이터셋의 대여량을 예측한다. 따라서, 본 프로젝트는 지도 학습(Supervised Learning) 접근 방식을 사용한다. 

자전거 대여 수요 예측은 여러 피처 간의 복잡한 상호작용을 포함한다. 다층 퍼셉트론 신경망은 이러한 복잡하고 비선형적인 관계를 효과적으로 모델링할 수 있다. PyTorch를 이용하여 다층 퍼셉트론 신경망 모델을 구축하였으며, 이 모델은 연도, 요일, 시간 등의 시계열 피처와 계절, 공휴일, 근무일 여부, 날씨 상태, 온도, 체감 온도, 습도, 풍속 등의 기후 피처를 입력으로 받아 자전거 대여 수요를 예측한다. 이는 자전거 대여 패턴에 중요한 영향을 미치는 요소들이며, 모델의 예측 성능을 향상시키기 위해 선택 했다.

---

## IV. Evaluation & Analysis

-Graphs, tables, any statistics (if any)

---

## V. Related Work (e.g., existing studies)

-Tools, libraries, blogs, or any documentation that you have used to do this project.

---

## VI. Conclusion: Discussion
