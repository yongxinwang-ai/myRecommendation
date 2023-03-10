import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim, num_filters, filter_sizes):
        super(CNNEncoder, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        # 卷积层
        self.conv_layers = nn.ModuleList()
        for filter_size in filter_sizes:
            conv_layer = nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            self.conv_layers.append(conv_layer)
        
        # 池化层
        self.pooling_layer = nn.AdaptiveMaxPool2d((1, num_filters))
        
    def forward(self, item_ids):
        # 计算项目嵌入向量
        item_embeddings = self.embedding(item_ids)
        
        # 增加一个维度
        item_embeddings = item_embeddings.unsqueeze(1)
        
        # 卷积层
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(item_embeddings)
            conv_output = F.relu(conv_output)
            conv_output = self.pooling_layer(conv_output)
            conv_output = conv_output.squeeze()
            conv_outputs.append(conv_output)
        
        # 将卷积输出拼接起来
        cnn_embeddings = torch.cat(conv_outputs, dim=1)
        
        return cnn_embeddings
