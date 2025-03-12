import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR  # 使用 VAR 模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from scipy.stats import entropy

# 数据加载
data = pd.read_csv("electricity.csv")  # 替换为你的数据集
data = data.values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)
train_data = data_scaled[:int(0.7 * len(data_scaled))]
test_data = data_scaled[int(0.7 * len(data_scaled)):]

# 超参数
input_size = data_scaled.shape[1]  # 输入特征维度
hidden_size = 64
output_size = data_scaled.shape[1]  # 输出特征维度
seq_length = 15  # 时间步长
batch_size = 32
learning_rate = 0.001
num_epochs = 1000

# VAR模型（多变量）
def var_predict(data):
    try:
        # 使用 VAR 模型
        model = VAR(data)
        model_fit = model.fit(maxlags=seq_length)  # 设置最大滞后阶数
        return model_fit.forecast(data, steps=seq_length)  # 预测未来 seq_length 步
    except Exception as e:
        print(f"VAR failed: {e}")
        return np.zeros((seq_length, data.shape[1]))  # 如果失败，返回零值

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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# VAR预测（多变量）
var_pred = var_predict(train_data)  # 对整个训练数据进行预测
var_pred = np.tile(var_pred, (X_train.shape[0], 1, 1))  # 扩展为与 X_train 相同的形状
var_pred = torch.tensor(var_pred, dtype=torch.float32)

# LSTM模型初始化
model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# 计算MSE
y_pred = test_outputs.numpy()
y_true = y_test_tensor.numpy()
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# 计算MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

# 计算F1分数（假设二值化）
threshold = np.mean(y_true)  # 使用均值作为阈值
y_true_binary = (y_true > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)
f1 = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), average='macro')  # 使用macro平均
print(f"F1 Score: {f1:.4f}")

# 计算KL散度（假设y_true和y_pred是概率分布）
def calculate_kl_divergence(y_true, y_pred):
    # 将数据归一化为概率分布
    y_true_normalized = y_true / np.sum(y_true, axis=1, keepdims=True)
    y_pred_normalized = y_pred / np.sum(y_pred, axis=1, keepdims=True)
    # 计算KL散度
    kl_div = np.mean([entropy(y_true_normalized[i], y_pred_normalized[i]) for i in range(len(y_true))])
    return kl_div

kl_divergence = calculate_kl_divergence(y_true, y_pred)
print(f"KL Divergence: {kl_divergence:.4f}")