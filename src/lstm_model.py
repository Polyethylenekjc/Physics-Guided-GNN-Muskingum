"""
LSTM模型定义
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM模型用于时间序列预测"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.2, bidirectional=False):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: dropout比例
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测输出 (batch_size, output_size)
        """
        # LSTM层
        # lstm_out: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        # last_output: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output
    
    def predict(self, x, scaler=None):
        """
        预测并可选地反标准化
        
        Args:
            x: 输入张量
            scaler: 标准化器(可选)
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            
        return predictions
