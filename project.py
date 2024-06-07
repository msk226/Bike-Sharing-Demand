import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
train = pd.read_csv('data/train.csv', parse_dates=["datetime"])
test = pd.read_csv('data/test.csv', parse_dates=["datetime"])
submission = pd.read_csv('data/sampleSubmission.csv')

# 특징 추출
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

# 특징 열 정의
cols = test.columns[1:]


# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[cols])
X_test_scaled = scaler.transform(test[cols])

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X_train_scaled, np.log1p(train["count"]), test_size=0.2, random_state=42)

# 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.values.reshape(-1, 1))
X_valid = torch.FloatTensor(X_valid)
y_valid = torch.FloatTensor(y_valid.values.reshape(-1, 1))
X_test = torch.FloatTensor(X_test_scaled)

# 모델 정의
class ImprovedModel(nn.Module):
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

# 모델, 손실 함수, 옵티마이저 설정
model = ImprovedModel(input_size=X_train.shape[1], output_size=1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# 검증 데이터로 성능 평가
with torch.no_grad():
    model.eval()
    y_pred_valid = model(X_valid)
    print(f'Test Loss: {loss_fn(y_pred_valid, y_valid).item():.4f}')

# RMSLE 계산 함수
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

# RMSLE 계산
y_true = y_valid.numpy().squeeze()  # 실제 값
y_pred = y_pred_valid.numpy().squeeze()  # 모델의 예측 값
rmsle_score = rmsle(np.expm1(y_true), np.expm1(y_pred))
print(f'RMSLE: {rmsle_score:.4f}')

# 테스트 데이터 예측
with torch.no_grad():
    outputs = model(X_test)
    y_predict = outputs.squeeze().numpy()

# 제출 파일 생성
submission["count"] = np.expm1(y_predict)
file_name = f"submit_improved_{rmsle_score:.5f}.csv"
submission.to_csv(file_name, index=False)
print(pd.read_csv(file_name).head(2))
