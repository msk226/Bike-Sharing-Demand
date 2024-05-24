
## Title: Bike sharing demand

Members: 
```
김민석, 컴퓨터학부, kminseok5167@gmail.com

엄규현, 해양융합공학과, gheom24@gmail.com

Name3, Department, Email ...
```

## I. Proposal (Option 1or 2)

–This should be filled bysometime in early May.

-Motivation: Why are you doing this?

-What do you want to see at the end?


마이크로 모빌리티가 친환경적인 교통 수단으로 인식 되고 있어 많은 기업은 마이크로 모빌리티 시장에 관심을 가지고 있습니다. 자전거 역시 다른 마이크로 모빌리티와 마찬가지로 환경 보호 측면과 도시 교통 관리에 매우 중요한 역할을 할 수 있습니다. 자전거는 탄소 배출을 줄이고, 교통 혼잡을 감소시키며, 에너지를 절약하는 교통수단 입니다. 그러나 자전거의 적절한 배치와 관리에는 정확한 수요와 예측이 필요합니다. 저희는 자전거 수요 예측 모델을 개발하여 도시 교통 관리와 환경 보호에 기여하고자 합니다. 자전거 수요 예측 모델은 자전거 사용 패턴을 분석하고 예측하여 효율적인 자전거 인프라의 운영과 관리에 도움을 줄 것 이고, 이를 통해 자전거 사용을 촉진하고, 도시의 교통 혼잡 문제 및 환경 오염 문제를 해결하는데 기여할 수 있을 것 입니다. 

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


## III. Methodology 

-Explaining your choice of algorithms (methods)
-Explaining features (if any)

## IV. Evaluation & Analysis

-Graphs, tables, any statistics (if any)

## V. Related Work (e.g., existing studies)

-Tools, libraries, blogs, or any documentation that you have used to do this project.

## VI. Conclusion: Discussion
