import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .Encoder import CNNEncoder

class DRN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_blocks=3, num_layers=2):
        super(DRN, self).__init__()

        # 嵌入层
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(embed_dim, num_layers) for _ in range(num_blocks)
        ])

        # 最终输出层
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, user_ids, item_ids):
        # 嵌入用户和项目ID
        user_embed = self.user_embed(user_ids)
        item_embed = self.item_embed(item_ids)

        # 连接用户和项目嵌入
        x = torch.cat([user_embed, item_embed], dim=1)

        # 通过残差块
        x = self.residual_blocks(x)

        # 最终输出层
        x = self.output_layer(x)

        return x.squeeze()


class ResidualBlock(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        return x + self.layers(x)


class DRN_CNN(nn.Module):
    def __init__(self, num_items, embedding_dim, num_filters, filter_sizes, hidden_dim, dropout_prob):
        super(DRN_CNN, self).__init__()
        
        # 嵌入层
        self.embedding = CNNEncoder(num_items, embedding_dim, num_filters, filter_sizes)
        
        # 注意力层
        self.attention = nn.Linear(num_filters * len(filter_sizes), 1, bias=False)
        
        # 隐藏层
        self.hidden_layer = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, user_history, candidate_items):
        # 计算用户历史项目的CNN嵌入向量
        history_embeddings = self.embedding(user_history)
        
        # 计算候选项目的CNN嵌入向量
        candidate_embeddings = self.embedding(candidate_items)
        
        # 计算用户历史项目向量的注意力分数
        attention_scores = self.attention(history_embeddings)

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权平均历史项目向量
        weighted_history_embeddings = attention_weights * history_embeddings
        weighted_history_embeddings = torch.sum(weighted_history_embeddings, dim=1)
        
        # 计算候选项目与用户历史项目的余弦相似度
        candidate_scores = torch.matmul(candidate_embeddings, weighted_history_embeddings.unsqueeze(-1))
        candidate_scores = candidate_scores.squeeze(-1)
        
        # 计算隐藏层输出
        hidden_output = F.relu(self.hidden_layer(candidate_scores))
        hidden_output = self.dropout(hidden_output)
        
        # 计算最终得分
        final_scores = self.output_layer(hidden_output)
        
        return final_scores



class DRN_RNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, num_layers, dropout_prob):
        super(DRN_RNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # 定义用户和项目的嵌入层
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # 定义RNN嵌入层和注意力层
        self.rnn_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, dropout=dropout_prob)
        self.attention = nn.Linear(hidden_dim, 1)

        # 定义隐藏层和输出层
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # 定义Dropout层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, user_ids, item_ids, history_ids):
        # 计算用户、候选和历史项目的嵌入向量
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        history_embeddings = self.item_embedding(history_ids)

        # 计算用户历史项目的RNN嵌入向量
        rnn_input = history_embeddings.view(history_embeddings.size(0), -1, self.embedding_dim)
        _, (rnn_output, _) = self.rnn_layer(rnn_input)
        history_embeddings = rnn_output[-1, :, :]

        # 计算用户历史项目向量的注意力分数
        attention_scores = self.attention(history_embeddings)
        attention_scores = attention_scores.squeeze(-1)

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)

        # 加权平均历史项目向量
        weighted_history_embeddings = attention_weights.unsqueeze(-1) * history_embeddings
        weighted_history_embeddings = torch.sum(weighted_history_embeddings, dim=1)

        # 计算候选项目与用户历史项目的余弦相似度
        candidate_scores = torch.sum(item_embeddings * weighted_history_embeddings, dim=1)

        # 计算隐藏层输出
        hidden_output = F.relu(self.hidden_layer(candidate_scores))
        hidden_output = self.dropout(hidden_output)

        # 计算最终得分
        final_scores = self.output_layer(hidden_output)

        return final_scores


"""
在前向方法中，我们首先计算用户、候选和历史项目的嵌入向量，然后将历史项目的嵌入向量输入到RNN嵌入层中计算RNN嵌入向量。
接下来，我们使用注意力层计算用户历史项目向量的注意力分数，然后使用softmax函数计算注意力权重。
我们将注意力权重应用于用户历史项目向量的加权平均值，得到加权历史项目向量。
最后，我们计算候选项目与加权历史项目向量之间的余弦相似度，然后通过隐藏层和输出层计算最终得分。
"""