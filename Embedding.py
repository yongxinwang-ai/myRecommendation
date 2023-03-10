import torch
import torch.nn as nn


#注意力机制嵌入
class AttentionEmbedding(nn.Module):
    def __init__(self, num_items, embedding_dim, attention_dim):
        super(AttentionEmbedding, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, item_ids):
        # 计算嵌入向量
        item_embeddings = self.embedding(item_ids)
        
        # 计算注意力权重
        attention_scores = self.attention(item_embeddings)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 加权平均嵌入向量
        weighted_embeddings = attention_weights * item_embeddings
        embeddings_sum = weighted_embeddings.sum(dim=1)
        
        return embeddings_sum


"""
在上面的代码中，我们定义了一个AttentionEmbedding类，它接受三个参数：
num_items表示项目的数量
embedding_dim表示嵌入向量的维度
attention_dim表示注意力层的隐藏维度
在类的初始化函数中，我们创建了一个嵌入层和一个注意力层。
在前向传播函数中，我们首先计算输入项目的嵌入向量，然后使用注意力层计算不同项目之间的注意力权重。
最后，我们使用这些注意力权重对嵌入向量进行加权平均，并返回加权平均向量作为最终的项目表示。
"""


#时间感知嵌入
class TimeAwareEmbedding(nn.Module):
    def __init__(self, num_items, embedding_dim, time_dim):
        super(TimeAwareEmbedding, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        # 时间嵌入层
        self.time_embedding = nn.Embedding(24*7, time_dim)
        
    def forward(self, item_ids, hour_of_day):
        # 计算项目嵌入向量
        item_embeddings = self.embedding(item_ids)
        
        # 计算时间嵌入向量
        time_embeddings = self.time_embedding(hour_of_day)
        
        # 将时间嵌入向量添加到项目嵌入向量中
        embeddings = item_embeddings + time_embeddings
        
        return embeddings

"""
在上面的代码中，我们定义了一个TimeAwareEmbedding类，它接受三个参数：
num_items表示项目的数量
embedding_dim表示项目嵌入向量的维度
time_dim表示时间嵌入向量的维度
在类的初始化函数中，我们创建了一个项目嵌入层和一个时间嵌入层。在前向传播函数中，我们首先计算输入项目的嵌入向量，然后使用时间嵌入层计算对应时间的时间嵌入向量
"""


class DeepEmbedding(nn.Module):
    def __init__(self, num_items, num_features, embedding_dim, hidden_dim):
        super(DeepEmbedding, self).__init__()
        
        # 项目嵌入层
        self.embedding = nn.Embedding(num_items, embedding_dim)
        
        # 深度嵌入网络
        self.network = nn.Sequential(
            nn.Linear(num_features * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, item_ids, features):
        # 计算项目嵌入向量
        item_embeddings = self.embedding(item_ids)
        
        # 展开特征矩阵
        features = features.view(features.size(0), -1)
        
        # 将项目嵌入向量和特征矩阵拼接在一起
        embeddings = torch.cat((item_embeddings, features), dim=1)
        
        # 计算深度嵌入向量
        embeddings = self.network(embeddings)
        
        return embeddings


"""
在上面的代码中，我们定义了一个DeepEmbedding类，它接受四个参数：
num_items表示项目的数量
num_features表示每个项目的特征数量
embedding_dim表示项目嵌入向量的维度
hidden_dim表示深度嵌入网络的隐藏层维度
在类的初始化函数中，我们创建了一个项目嵌入层和一个深度嵌入网络。
在前向传播函数中，我们首先计算输入项目的嵌入向量，然后将特征矩阵展开成一维张量，并将项目嵌入向量和特征矩阵拼接在一起
"""