"""
训练器模块
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    """LSTM训练器"""
    
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate=0.001, device='cuda', patience=15):
        """
        初始化训练器
        
        Args:
            model: LSTM模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            device: 设备('cuda' 或 'cpu')
            patience: early stopping耐心值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        print(f"训练设备: {self.device}")
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        
        for sequences, targets in self.train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss
    
    def train(self, epochs):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            
        Returns:
            train_losses, val_losses: 训练和验证损失历史
        """
        print(f"\n开始训练，共 {epochs} 个epochs")
        print("=" * 70)
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                # 恢复最佳模型
                self.model.load_state_dict(self.best_model_state)
                break
        
        print("=" * 70)
        print(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses
    
    def save_model(self, save_path):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path)
        
        print(f"模型已保存至: {save_path}")
    
    def load_model(self, load_path):
        """
        加载模型
        
        Args:
            load_path: 模型路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"模型已从 {load_path} 加载")


class Evaluator:
    """评估器"""
    
    def __init__(self, model, data_loader, data_processor, device='cuda'):
        """
        初始化评估器
        
        Args:
            model: 训练好的模型
            data_loader: 数据加载器
            data_processor: 数据处理器
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.data_processor = data_processor
        
    def predict(self):
        """
        对数据集进行预测
        
        Returns:
            y_true, y_pred: 真实值和预测值（原始尺度）
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in self.data_loader:
                sequences = sequences.to(self.device)
                
                predictions = self.model(sequences)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # 合并所有批次
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 反标准化
        y_pred = self.data_processor.inverse_transform_target(all_predictions)
        y_true = self.data_processor.inverse_transform_target(all_targets)
        
        return y_true, y_pred
