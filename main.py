

import torch
import torch.nn as nn
import torch.optim as optim



def train(model, train_dataset, test_dataset, batch_size=128, lr=1e-3, num_epochs=10):
    # 损失函数和优化器
    #loss_fn = nn.MarginRankingLoss(margin=0.1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # 训练循环
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        for user_ids, item_ids, ratings in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
            # 前向传递
            predictions = model(user_ids, item_ids)

            # 计算损失
            loss = loss_fn(predictions, ratings)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算RMSE
            train_rmse += torch.sqrt(loss).item() * len(user_ids)
            train_loss += loss.item() * len(user_ids)

    
    model.eval()


    test_loss = 0.0
    test_rmse = 0.0

    with torch.no_grad():
        for user_ids, item_ids, ratings in DataLoader(test_dataset, batch_size=batch_size):
            # 前向传递
            predictions = model(user_ids, item_ids)

            # 计算损失
            loss = loss_fn(predictions, ratings)

            # 计算RMSE
            test_rmse += torch.sqrt(loss).item() * len(user_ids)
            test_loss += loss.item() * len(user_ids)

    # 计算平均损失和RMSE
    train_loss /= len(train_dataset)
    train_rmse /= len(train_dataset)
    test_loss /= len(test_dataset)
    test_rmse /= len(test_dataset)


     # 打印训练进度
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train RMSE: {train_rmse:.4f} - Test Loss: {test_loss:.4f} - Test RMSE: {test_rmse:.4f}")