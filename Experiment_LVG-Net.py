import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
import time
import os

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 数据加载与预处理
def load_data(file_path, seq_length=15, stride=1):
    print("Loading data...")
    start_time = time.time()

    data = pd.read_csv(file_path).values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    print(f"Data shape: {data_scaled.shape} | Time: {time.time() - start_time:.2f}s")

    # 划分训练集和测试集
    train_size = int(0.7 * len(data_scaled))
    train_raw = data_scaled[:train_size]
    test_raw = data_scaled[train_size:]

    # 创建时序样本
    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i + seq_len])
            ys.append(data[i + seq_len])
        return np.array(xs), np.array(ys)

    print("Creating training sequences...")
    X_train, y_train = create_sequences(train_raw, seq_length)
    print("Creating test sequences...")
    X_test, y_test = create_sequences(test_raw, seq_length)

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Data loading completed. Total time: {time.time() - start_time:.2f}s")

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler
    )


class LVGNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, latent_size=16, num_layers=2, seq_length=15):
        super(LVGNet, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.input_size = input_size

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder_init = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.discriminator_fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        _, (h_n, _) = self.encoder(x)
        h_n = h_n[-1]
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)

        # Decoding
        decoder_h0 = self.decoder_init(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_c0 = torch.zeros_like(decoder_h0)
        decoder_input = torch.zeros(x.size(0), self.seq_length, self.hidden_size).to(x.device)
        recon_seq, _ = self.decoder(decoder_input, (decoder_h0, decoder_c0))
        recon_x = self.fc_out(recon_seq)

        # 获取预测结果（最后一个时间步）
        pred = recon_x[:, -1, :]

        # Discrimination
        d_out, _ = self.discriminator_lstm(recon_x.detach())  # LSTM输出
        validity = self.discriminator_fc(d_out[:, -1, :])  # 只取最后一个时间步的输出

        return pred, mu, logvar, validity

    def save(self, path):
        """保存模型和超参数"""
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'latent_size': self.latent_size,
            'num_layers': self.num_layers,
            'input_size': self.input_size,
            'seq_length': self.seq_length
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """加载模型和超参数"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            latent_size=checkpoint['latent_size'],
            num_layers=checkpoint['num_layers'],
            seq_length=checkpoint['seq_length']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


def evaluate_model(model, X_test_tensor, y_test_tensor, scaler):
    """
    评估模型性能
    :param model: 训练好的模型
    :param X_test_tensor: 测试集输入数据 (torch.Tensor)
    :param y_test_tensor: 测试集真实标签 (torch.Tensor)
    :param scaler: 数据归一化器 (用于反归一化)
    :return: 包含评估指标的字典
    """
    model.eval()
    with torch.no_grad():
        # 模型预测（只取第一个返回值 pred）
        pred, _, _, _ = model(X_test_tensor.to(device))
        test_outputs = pred.cpu()  # 将预测结果移动到 CPU

    # 反归一化
    y_pred = scaler.inverse_transform(test_outputs.numpy())
    y_true = scaler.inverse_transform(y_test_tensor.numpy())

    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # F1 Score（使用相同阈值策略）
    threshold = np.mean(y_true)
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    f1 = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), average='macro')

    # KL散度
    def calculate_kl_divergence(true, pred):
        true_normalized = true / (np.sum(true, axis=1, keepdims=True) + 1e-8)
        pred_normalized = pred / (np.sum(pred, axis=1, keepdims=True) + 1e-8)
        return np.mean([entropy(true_normalized[i], pred_normalized[i]) for i in range(len(true))])

    kl_div = calculate_kl_divergence(y_true, y_pred)

    return {
        "MSE": mse,
        "MAE": mae,
        "F1": f1,
        "KL_Divergence": kl_div
    }


class Trainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, device=device):
        print("\nInitializing trainer...")
        self.model = model.to(device)
        self.device = device

        # 数据加载（添加pin_memory加速）
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=32,
                                       shuffle=True, pin_memory=True)
        test_dataset = TensorDataset(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=32,
                                      pin_memory=True)

        # 优化器（添加权重衰减）
        self.opt_vae = torch.optim.Adam(
            [p for n, p in model.named_parameters() if 'discriminator' not in n],
            lr=0.001, weight_decay=1e-5
        )
        self.opt_disc = torch.optim.Adam(
            list(model.discriminator_lstm.parameters()) + list(model.discriminator_fc.parameters()),
            lr=0.0001, weight_decay=1e-5
        )
        self.scheduler_vae = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_vae, patience=5, factor=0.5
        )

        # 损失函数
        self.recon_criterion = nn.MSELoss()
        self.adv_criterion = nn.BCELoss()
        print("Trainer initialized.\n")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            # 训练判别器
            self.opt_disc.zero_grad()
            with torch.no_grad():
                recon_pred, _, _, _ = self.model(x)
                # 生成完整序列用于判别
                fake_seq = torch.cat([x[:, :-1, :], recon_pred.unsqueeze(1)], dim=1)

            # 真实数据判别
            d_out_real, _ = self.model.discriminator_lstm(x)
            real_pred = self.model.discriminator_fc(d_out_real[:, -1, :])
            d_real_loss = self.adv_criterion(real_pred, torch.ones_like(real_pred))

            # 生成数据判别
            d_out_fake, _ = self.model.discriminator_lstm(fake_seq.detach())
            fake_pred = self.model.discriminator_fc(d_out_fake[:, -1, :])
            d_fake_loss = self.adv_criterion(fake_pred, torch.zeros_like(fake_pred))

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator_lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.discriminator_fc.parameters(), 1.0)
            self.opt_disc.step()

            # 训练生成器
            self.opt_vae.zero_grad()
            recon_pred, mu, logvar, validity = self.model(x)

            # 重构损失
            recon_loss = self.recon_criterion(recon_pred, y)
            # KL散度
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # 对抗损失
            adv_loss = self.adv_criterion(validity, torch.ones_like(validity))

            total_vae_loss = recon_loss + 0.1 * kl_loss + adv_loss
            total_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt_vae.step()

            total_loss += total_vae_loss.item()

            # 打印batch进度
            if (batch_idx + 1) % 50 == 0:
                speed = (time.time() - start_time) / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Avg time/batch: {speed:.3f}s | "
                      f"Current loss: {total_vae_loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                recon_pred, _, _, _ = self.model(x)
                recon_loss = self.recon_criterion(recon_pred, y)
                total_loss += recon_loss.item()
        return total_loss / len(self.test_loader)


# 主程序
if __name__ == "__main__":
    # 超参数对齐
    SEQ_LENGTH = 15
    INPUT_SIZE = 321  # 根据实际数据维度调整
    HIDDEN_SIZE = 64  # 与VARIMA-LSTM的hidden_size=64对齐
    LATENT_SIZE = 16
    NUM_LAYERS = 2
    EPOCHS = 1000  # 与VARIMA-LSTM的num_epochs=1000对齐

    # 数据加载
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler = load_data(
        "electricity.csv", seq_length=SEQ_LENGTH
    )

    # 模型初始化
    model = LVGNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        num_layers=NUM_LAYERS,
        seq_length=SEQ_LENGTH
    )

    # 训练
    trainer = Trainer(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = trainer.train_epoch()
        test_loss = trainer.evaluate()
        trainer.scheduler_vae.step(test_loss)

        if (epoch + 1) % 50 == 0:
            print("\nRunning validation...")
            metrics = evaluate_model(model, X_test_tensor, y_test_tensor, scaler)
            print("Current metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            model.save("best_model.pth")

    # 最终评估
    print("\nTraining completed. Final evaluation...")
    model = LVGNet.load("best_model.pth", device=device)
    metrics = evaluate_model(model, X_test_tensor, y_test_tensor, scaler)
    print("\nFinal Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")