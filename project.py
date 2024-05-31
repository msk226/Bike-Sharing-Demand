import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split



train = pd.read_csv(f'train.csv', parse_dates=["datetime"])
test = pd.read_csv(f'test.csv', parse_dates=["datetime"])
submission = pd.read_csv(f'sampleSubmission.csv')

print(train.head(1))

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


cols = test.columns[1:]
X, y = train[cols], np.log1p(train["count"])
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=42)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values.reshape(-1, 1))
X_valid = torch.FloatTensor(X_valid.values)
y_valid = torch.FloatTensor(y_valid.values.reshape(-1, 1))
X_test = torch.FloatTensor(test[cols].values)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)


class MultivariateLinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultivariateLinearRegression, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


model = MultivariateLinearRegression(input_size=X_train.shape[1], output_size=1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X_train)
    
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}]')


with torch.no_grad():
    y_pred_valid = model(X_valid)
    print(f'Test Loss: {loss_fn(y_pred_valid, y_valid).item():.4f}')

print(y_valid.shape, y_pred_valid.shape)
valid_score = ((y_valid - y_pred_valid) ** 2).mean() ** 0.5
print(valid_score)


with torch.no_grad():
    outputs = model(X_test)
    y_predict = outputs.squeeze().numpy()
print(y_predict[:5])

submission["count"] = np.expm1(y_predict)
print(submission.head(2))

file_name = f"submit_pytorch_{valid_score:.5f}.csv"
submission.to_csv(file_name, index=False)
print(pd.read_csv(file_name).head(2))
