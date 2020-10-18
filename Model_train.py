'''
seefun . Aug 2020.
Modified by Naive . Sep 2020.
1. add L1-norm regularization on the NMSE loss
2. adjust the number of kernel
3. use different initilazations
4. learning rate adjustment according to reference
'''

import numpy as np
import h5py
import torch
import os
import math
import torch.nn as nn
import random
from torch.nn import init

from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss

# Parameters for training
gpu_list = '1, 2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=258):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
SEED = 258
seed_everything(SEED) 

batch_size = 256
epochs = 150
learning_rate = 2e-3  # bigger to train faster
learning_rate_init = 2e-3
learning_rate_final = 5e-6
num_workers = 4
print_freq = 500  
train_test_ratio = 0.8
L1_weight = 5e-5
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2


# Model construction
model = AutoEncoder(feedback_bits)

model.encoder.quantization = False
model.decoder.quantization = False

# initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform(m.weight, mode='fan_in', nonlinearity='leaky_relu')


if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()

criterion = NMSELoss(reduction='mean')  # nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Data loading
mat = h5py.File('./data/Hdata.mat', 'r')
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(80%) and validation(20%)
np.random.shuffle(data)
start = int(data.shape[0] * train_test_ratio)
x_train, x_test = data[:start], data[start:]

# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)



best_loss = 200
for epoch in range(epochs):
    print('========================')
    print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
    # model training
    model.train()
    if epoch < epochs//10:
        try:
            model.encoder.quantization = False
            model.decoder.quantization = False
        except:
            model.module.encoder.quantization = False
            model.module.decoder.quantization = False
    else:
        try:
            model.encoder.quantization = True
            model.decoder.quantization = True
        except:
            model.module.encoder.quantization = True
            model.module.decoder.quantization = True
        
    # lr modified according to cosine
    optimizer.param_groups[0]['lr'] = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(
        1+math.cos((epoch*3.14)/epochs)
    )
    
    for i, input in enumerate(train_loader):
            
        input = input.cuda()
        output = model(input)
        
        loss = criterion(output, input)
        # 使用L1 正则化项强制使网络学习稀疏特征
        L1_loss = 0
        for param in model.parameters():
            L1_loss += torch.sum(torch.abs(param))
        L1_loss = 0  # 该问题似乎并不需要正则化
        loss = L1_loss * L1_weight + loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))
    model.eval()
    try:
        model.encoder.quantization = True
        model.decoder.quantization = True
    except:
        model.module.encoder.quantization = True
        model.module.decoder.quantization = True
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion_test(output, input).item()
        average_loss = total_loss / len(test_dataset)
        print('NMSE %.4f'%average_loss)
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
            print('Model saved!')
            best_loss = average_loss

