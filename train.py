import time
import torch
import numpy as np

from core.loss import TotalLoss
from core.model import FairMOT as build_model
from core.dataset import MOTDatset
from train_utils import setup_seed, CosineDecayLR, save_model_weights

setup_seed(2022)

print('CUDA : ', torch.cuda.is_available())
if torch.cuda.is_available():
    device = 'cuda:0'

dataset_path = '/home/hunmin/project/dataset/MOT20_train'
class_file = './data/mot.names'

epochs = 300
input_img_size = 512
output_size = 128
batch_size = 4
epochs = 300
lr = 1e-3

save_root = './checkpoints'
log_root = './logs'
checkpoint_dir = time.strftime('%y%m%d_%H%M')

train_dataset = MOTDatset(dataset_path, class_file, input_img_size, output_size)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True
                                            )

train_dataset_id_num = train_dataset.total_id_nums
class_num = train_dataset.class_num
print(len(train_dataset), len(train_loader))

model = build_model(1, train_dataset_id_num)
model.to(device)
model.train()
loss_func = TotalLoss()
len_train = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
warmup = 3
lr_end = lr / 100
scheduler = CosineDecayLR(optimizer,
                            T_max=epochs * len_train,
                            lr_init=lr,
                            lr_min=lr_end,
                            warmup=warmup * len_train)


for epoch in range(epochs):
    loss_temp = []
    for i, data in enumerate(train_loader):
        reduced_lr = scheduler.step(len_train / (batch_size) * epoch + i)
        img = data[0].to(device)
        heat_map = data[1].to(device)
        offset_wh = data[2].to(device)
        offset = offset_wh[:, 2:, ...]
        wh = offset_wh[:, :2, ...]
        id_info = data[3].to(device)
        pred = model(img)
        loss, temp = loss_func(heat_map, offset, wh, id_info, pred[0], pred[1], pred[2], pred[3])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.cpu().detach().item()
        losses = [loss_np]

        for l in temp:
            losses.append(l.cpu().detach().item())
        loss_temp.append(losses)
        if i % 10 == 0:
            loss_temp = np.array(loss_temp)
            loss_temp = np.mean(loss_temp, 0)
            loss_temp = np.around(loss_temp, 3).astype(str)
            print('{}, {} '.format(epoch, i) + ', '.join(loss_temp.tolist()), reduced_lr)
            loss_temp = []
    if epoch % 5 == 0:
        save_model_weights(model, optimizer, epoch, save_root, checkpoint_dir)