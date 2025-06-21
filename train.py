
import torch
from model import AddiDecomposer
from data import sample_data, validset
import matplotlib.pyplot as plt

import os
import pickle

name = 'adder1'

lr = 1e-3
batch_size = 32

num_epoch = 100
num_valid_cycle = 100

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_one_iter(adder,num_update_cycle=32,lr=1e-3):
    """
    训练一个迭代周期，累积 num_update_cycle 次的梯度后更新一次模型参数。
    """
    global device
    total_loss = 0
    optimizer = torch.optim.Adam(adder.parameters(),lr=lr)  # 定义优化器
    # 累积梯度
    optimizer.zero_grad()  # 清空初始梯度
    for i in range(num_update_cycle):
        # 采样数据
        data = sample_data()
        x1, x2 = data['data']
        # 确保 x1 >= x2
        if x1 < x2:
            x1, x2 = x2, x1
        # 计算损失
        loss = adder.get_loss(x1, x2)
        total_loss += loss.item()
        # 反向传播（累积梯度）
        loss.backward()
    # 计算平均损失
    avg_loss = total_loss / num_update_cycle
    # 平均梯度并更新参数
    for param in adder.parameters():
        if param.grad is not None:
            param.grad /= num_update_cycle  # 平均梯度
    optimizer.step()  # 更新参数
    return avg_loss


def valid(adder,validset):
    """
    在验证集上评估模型性能。
    """
    global device
    total_loss = 0
    adder.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for data in validset:
            x1, x2 = data['data']
            # 确保 x1 >= x2
            if x1 < x2:
                x1, x2 = x2, x1
            # 计算损失
            loss = adder.get_loss(x1, x2)
            total_loss += loss.item()
    # 计算平均损失
    avg_loss = total_loss / len(validset)
    return avg_loss


def save_loss_fig(loss_train_all,loss_valid_all,output_path):
    # 创建图像
    plt.figure(figsize=(8, 5))  # 设置图像大小
    # 绘制训练损失曲线（蓝色）
    plt.plot(loss_train_all, label='Training Loss', color='blue', marker='o')
    # 绘制验证损失曲线（红色）
    plt.plot(loss_valid_all, label='Validation Loss', color='red', marker='x')
    # 添加标题和标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # 显示图例
    # 保存为 PDF
    save_path = os.path.join(output_path,'{}_loss.pdf'.format(name))
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    # 显示图像（可选）
    plt.close()



ckp_path = '../params/{}.pkl'.format(name)
output_path = '../output'


loss_train_all = []
loss_valid_all = []
adder = AddiDecomposer(n_bits=64).to(device)  # 初始化模型

need_load = os.path.exists(ckp_path)
if need_load:
    with open(ckp_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    ckp = loaded_dict['params']
    adder.load_state_dict(ckp)
    loss_train_all,loss_valid_all = loaded_dict['loss']


for j in range(num_epoch):
    total_loss_train = 0
    for i in range(num_valid_cycle):
        loss_train = train_one_iter(adder=adder,num_update_cycle=batch_size,lr=lr)
        print('epoch:{} iter:{} Trainloss:{}'.format(j,i,loss_train))
        total_loss_train += loss_train / num_valid_cycle
    loss_train_all.append(total_loss_train)
    loss_valid = valid(adder=adder, validset=validset)
    loss_valid_all.append(loss_valid)
    print('epoch:{} Validloss:{}'.format(loss_valid))
    save_dict = {'params':adder.state_dict(),'loss':(loss_train_all,loss_valid_all)}
    with open(ckp_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print('参数保存完成')
    save_loss_fig(loss_train_all,loss_valid_all,output_path)
    print('loss绘制完成')


'''
x = x1 + x2
x1_pred,x2_pred = adder.infer(x)
'''

