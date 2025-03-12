import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import MinMaxScaler

# 数据加载
data = pd.read_csv("electricity.csv")  # 替换为你的数据集
data = data.values
scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一化到 [0, 1] 范围
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

# VAE编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size * seq_length, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = torch.relu(self.fc1(x.view(-1, input_size * seq_length)))
        return self.fc_mu(h), self.fc_logvar(h)

# VAE解码器
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * seq_length)
        self.output_size = output_size  # 添加 output_size 属性

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h)).view(-1, seq_length, self.output_size)  # 使用 Sigmoid 确保输出在 [0, 1] 范围内

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size * seq_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # 确保输出在 [0, 1] 范围内

    def forward(self, x):
        h = torch.relu(self.fc1(x.view(-1, input_size * seq_length)))
        return self.sigmoid(self.fc2(h))

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
encoder = Encoder(input_size, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_size)
discriminator = Discriminator(input_size, hidden_size)
criterion = nn.BCELoss()
optimizer_VAE = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        real_data = X_train[i:i + batch_size]
        current_batch_size = real_data.size(0)  # 当前批次的样本数

        # 训练判别器
        optimizer_D.zero_grad()
        real_labels = torch.ones(current_batch_size, 1)  # 真实数据标签为 1
        fake_labels = torch.zeros(current_batch_size, 1)  # 生成数据标签为 0

        # 真实数据的判别器损失
        real_loss = criterion(discriminator(real_data), real_labels)

        # 生成数据
        mu, logvar = encoder(real_data)
        z = encoder.reparameterize(mu, logvar)
        fake_data = decoder(z)

        # 生成数据的判别器损失
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)

        # 判别器总损失
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # 训练VAE
        optimizer_VAE.zero_grad()
        recon_loss = criterion(fake_data, real_data)  # 重构损失
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL 散度
        vae_loss = recon_loss + kl_div
        vae_loss.backward()
        optimizer_VAE.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, VAE Loss: {vae_loss.item():.4f}")

# 测试
encoder.eval()
decoder.eval()
with torch.no_grad():
    mu, logvar = encoder(X_test)
    z = encoder.reparameterize(mu, logvar)
    fake_data = decoder(z)
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
    threshold = 2 * np.std(y_true_flat - y_pred_flat)  # 假设异常检测的阈值为 2 倍标准差
    y_true_anomaly = (y_true_flat - y_pred_flat) > threshold
    y_pred_anomaly = (y_true_flat - y_pred_flat) > threshold
    f1 = f1_score(y_true_anomaly.flatten(), y_pred_anomaly.flatten(), average='macro')  # 使用 macro 平均
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