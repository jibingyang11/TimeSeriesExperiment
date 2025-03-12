import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 数据加载
data = pd.read_csv("electricity.csv")  # 替换为你的数据集
data = data.values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)
train_data = data_scaled[:int(0.7 * len(data))]
test_data = data_scaled[int(0.7 * len(data)):]
# 超参数
input_size = data.shape[1]  # 输入特征维度
hidden_size = 64
output_size = data.shape[1]  # 输出特征维度
num_layers = 2
seq_length = 15  # 时间步长
batch_size = 32
learning_rate = 0.001
num_epochs = 1000
# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 数据预处理
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 模型初始化
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
# 计算MSE
y_pred = test_outputs.numpy()
y_true = y_test.numpy()
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
# 计算 MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
# 计算 F1 分数（异常检测）
# 假设异常检测的阈值为 2 倍标准差
threshold = 2 * np.std(y_true - y_pred)
y_true_anomaly = (y_true - y_pred) > threshold
y_pred_anomaly = (y_true - y_pred) > threshold
f1 = f1_score(y_true_anomaly, y_pred_anomaly, average='macro')  # 使用 macro 平均
print(f"F1 Score (Anomaly Detection): {f1:.4f}")