import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 数据加载
data = pd.read_csv("electricity.csv")  # 替换为你的数据集
data = data.values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)
train_data = data_scaled[:int(0.7 * len(data))]
test_data = data_scaled[int(0.7 * len(data)):]

# 超参数
input_size = data.shape[1]  # 输入特征维度
hidden_size = 64  # 减少隐藏层大小
latent_size = 16
seq_length = 15  # 时间步长
batch_size = 64  # 增加批量大小
learning_rate = 0.001
num_epochs = 100
dropout_prob = 0.2  # 添加 Dropout

# 生成器（LSTM-based）
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        output, _ = self.lstm(z)
        return self.fc(output)

# 判别器（LSTM-based）
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.sigmoid(self.fc(output[:, -1, :]))

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
generator = Generator(latent_size, hidden_size, input_size, num_layers=2)
discriminator = Discriminator(input_size, hidden_size, num_layers=2)
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 学习率调度器（移除 verbose 参数）
scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)
scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)

# 训练
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    for i in range(0, len(X_train), batch_size):
        real_data = X_train[i:i + batch_size]
        current_batch_size = real_data.size(0)  # 当前批次的样本数
        z = torch.randn(current_batch_size, latent_size)
        fake_data = generator(z)

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_data), torch.ones(current_batch_size, 1))
        fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(current_batch_size, 1))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_data), torch.ones(current_batch_size, 1))
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer_G.step()

    # 更新学习率
    scheduler_G.step(g_loss)
    scheduler_D.step(d_loss)

    # 手动打印学习率
    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D = optimizer_D.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
          f"LR Generator: {current_lr_G:.6f}, LR Discriminator: {current_lr_D:.6f}")
# 测试
generator.eval()
with torch.no_grad():
    z = torch.randn(len(X_test), latent_size)
    fake_data = generator(z)
    y_pred = fake_data.numpy()
    y_true = X_test.numpy()

    # 将三维数据展平为二维
    y_true_flat = y_true.reshape(-1, input_size)
    y_pred_flat = y_pred.reshape(-1, input_size)

    # 计算 MSE
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    print(f"MSE: {mse:.4f}")

    # 计算 MAE
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    print(f"MAE: {mae:.4f}")

    # 计算 F1 分数（异常检测）
    threshold = 2 * np.std(y_true_flat - y_pred_flat)
    y_true_anomaly = (y_true_flat - y_pred_flat) > threshold
    y_pred_anomaly = (y_true_flat - y_pred_flat) > threshold
    f1 = f1_score(y_true_anomaly.flatten(), y_pred_anomaly.flatten(), average='macro')
    print(f"F1 Score (Anomaly Detection): {f1:.4f}")

    # 计算 KL 散度
    def kl_divergence(p, q):
        eps = 1e-10  # 添加一个极小值避免除零
        return np.sum(p * np.log((p + eps) / (q + eps)))

    # 将数据归一化为概率分布
    p = (np.abs(y_true_flat) + 1e-10) / (np.sum(np.abs(y_true_flat)) + 1e-10)
    q = (np.abs(y_pred_flat) + 1e-10) / (np.sum(np.abs(y_pred_flat)) + 1e-10)
    kl_div = kl_divergence(p, q)
    print(f"KL Divergence: {kl_div:.4f}")