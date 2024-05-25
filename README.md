
## Title: Bike sharing demand

Members: 
```
김민석, 컴퓨터학부, kminseok5167@gmail.com

엄규현, 해양융합공학과, gheom24@gmail.com
김현민 , 산업경영공학과, jihiyo7@hanyang.ac.kr
Name3, Department, Email ...
```

## I. Proposal 

**Why are you doing this?**

마이크로 모빌리티는 최근 친환경적인 교통 수단으로 인식되며 많은 기업과 도시가 이에 대한 관심을 높이고 있다. 자전거는 전통적인 마이크로 모빌리티 수단으로서, 환경 보호와 도시 교통 관리에 중요한 역할을 할 수 있다. 자전거 사용은 탄소 배출을 줄이고, 교통 혼잡을 완화하며, 에너지 절약에 기여한다. 그러나 이러한 혜택을 극대화하려면 자전거의 적절한 배치와 관리가 필수적이다. 이를 위해서는 정확한 수요 예측이 필요하다. 이를 통해 자전거 대여 시스템의 효율성을 높이고, 사용자 경험을 향상시킬 수 있다.

**What do you want to see at the end?**

목표는 자전거 수요 예측 모델을 개발하여 도시 교통 관리와 환경 보호에 기여하는 것이다. 이 예측 모델은 자전거 사용 패턴을 분석하고 미래의 수요를 예측하여 자전거 인프라의 운영과 관리를 최적화하는 데 도움을 줄 것이다. 이를 통해 자전거가 필요할 때 적절한 위치에 배치되어 사용자들이 편리하게 이용할 수 있도록 하고, 자전거 사용을 촉진하여 자동차 사용을 줄임으로써 탄소 배출을 감소시키며, 도로의 차량 수를 줄여 교통 혼잡 문제를 완화하고, 자전거의 높은 에너지 효율로 전체 에너지 소비를 줄이는 데 기여할 수 있다. 지속 가능한 도시 교통을 실현하여 환경 보호에 기여하고, 더 깨끗하고 살기 좋은 도시를 만드는 데 일조할 수 있을 것이다.

--- 
## II. Datasets

-Describing your dataset 

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

---
## III. Methodology 

-Explaining your choice of algorithms (methods)
-Explaining features (if any)

---

## IV. Evaluation & Analysis

-Graphs, tables, any statistics (if any)

---

## V. Related Work (e.g., existing studies)

-Tools, libraries, blogs, or any documentation that you have used to do this project.

---

## VI. Conclusion: Discussion
