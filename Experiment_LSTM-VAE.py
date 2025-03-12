import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import MinMaxScaler

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
latent_size = 16
seq_length = 15  # 时间步长
batch_size = 32
learning_rate = 0.001
num_epochs = 1000

# LSTM-VAE模型
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super(LSTMAE, self).__init__()
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.lstm_decoder = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.fc_decoder = nn.Linear(hidden_size, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码器
        _, (h_n, _) = self.lstm_encoder(x)
        h_n = h_n[-1]  # 取最后一层的隐藏状态
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)

        # 解码器
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        output, _ = self.lstm_decoder(z)
        recon_x = self.fc_decoder(output)
        return recon_x, mu, logvar

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
model = LSTMAE(input_size, hidden_size, latent_size, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    recon_x, mu, logvar = model(X_train)
    recon_loss = criterion(recon_x, X_train)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_div
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}")

# 测试
model.eval()
with torch.no_grad():
    recon_x, mu, logvar = model(X_test)
    recon_loss = criterion(recon_x, X_test)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    test_loss = recon_loss + kl_div
    print(f"Test Loss: {test_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}")

# 计算 MSE 和 MAE
y_pred = recon_x.numpy()
y_true = X_test.numpy()

# 将三维数据展平为二维
y_true_flat = y_true.reshape(-1, input_size)
y_pred_flat = y_pred.reshape(-1, input_size)

mse = mean_squared_error(y_true_flat, y_pred_flat)
mae = mean_absolute_error(y_true_flat, y_pred_flat)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# 计算 F1 分数（异常检测）
# 假设异常检测的阈值为 2 倍标准差
threshold = 2 * np.std(y_true_flat - y_pred_flat)
y_true_anomaly = (y_true_flat - y_pred_flat) > threshold
y_pred_anomaly = (y_true_flat - y_pred_flat) > threshold
f1 = f1_score(y_true_anomaly.flatten(), y_pred_anomaly.flatten(), average='macro')  # 使用 macro 平均
print(f"F1 Score (Anomaly Detection): {f1:.4f}")