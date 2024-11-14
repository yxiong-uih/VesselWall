# This is the main file for PyTorch implementation of the algorithm.
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from datetime import datetime
from VesseSeg.nets.vnet import Net

# 训练函数, epoch为训练轮数
def train(model, train_loader, optimizer, loss_function, epoch, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    model.train()
    for i, arr_dict in enumerate(train_loader):
        image = arr_dict['image']
        if torch.any(torch.isnan(image)): continue
        mask = arr_dict['mask']
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output, mask)
        loss.backward()
        optimizer.step()
        plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_image.png'), image.cpu().numpy()[0, 0, 50, :, :], cmap='gray')
        plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_logit.png'), output.detach().cpu().numpy()[0, 0, 50, :, :], cmap='gray')
        plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_color.png'), create_mask_color(np.max(torch.argmax(output, dim=1).detach().cpu().numpy()[0], axis=0)))
        if i % 50 == 0: save_model(model, epoch)
        _, dice_results_str = calculate_dice_per_class(output, mask)
        __stdout__ = '%s - Epoch: %d, Iter: %d, Loss: %.4f, %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, i, loss.item(), dice_results_str)
        print(__stdout__, end='')
        with open(os.path.join(run_dir, 'log_train.txt'), 'a') as file: file.write(__stdout__)
        # assert False

# 训练函数，使用AMP混合精度和梯度累计训练
def train_amp_accumulate(model, train_loader, optimizer, loss_function, epoch, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    accumulate_step = 8
    model.train()
    optimizer.zero_grad()
    for i, arr_dict in enumerate(train_loader):
        image = arr_dict['image']
        if torch.any(torch.isnan(image)): continue
        mask = arr_dict['mask']
        image = image.to(device)
        mask = mask.to(device)
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = loss_function(output, mask)
            pass
        scaler.scale(loss).backward()
        if (i + 1) % accumulate_step == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        try:
            plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_image.png'), image.cpu().numpy()[0, 0, 50, :, :], cmap='gray')
            plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_logit.png'), output.detach().cpu().numpy()[0, 0, 50, :, :], cmap='gray')
            plt.imsave(os.path.join(run_dir, f'img_train_{epoch}_color.png'), create_mask_color(np.max(torch.argmax(output, dim=1).detach().cpu().numpy()[0], axis=0)))
        except Exception as e:
            print("plt.imsave", e)

        if i % 50 == 0: save_model(model, epoch)
        _, dice_results_str = calculate_dice_per_class(output, mask)
        __stdout__ = '%s - Epoch: %d, Iter: %d, Loss: %.4f, %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, i, loss.item(), dice_results_str)
        print(__stdout__, end='')
        with open(os.path.join(run_dir, 'log_train.txt'), 'a') as file: file.write(__stdout__)
        # assert False


# 测试函数
def val(model, test_loader, loss_function, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # # 测试模型
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, arr_dict in enumerate(test_loader):
            image = arr_dict['image']
            mask = arr_dict['mask']
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = loss_function(output, mask.long())

            total_loss += loss.item()
            _, dice_results_str = calculate_dice_per_class(output, mask)
            __stdout__ = '%s - Iter: %d, Loss: %.4f, %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i, loss.item(), dice_results_str) 
            print(__stdout__, end='')
            with open(os.path.join(run_dir, 'log_val.txt'), 'a') as file: file.write(__stdout__)

    avg_loss = total_loss / len(test_loader)
    print('Test Loss: %.4f' % (avg_loss)) 
    return avg_loss


# 预测函数
def predict(model, image, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        print(image.shape, output.shape, output.min(), output.max(), output.std(), output.mean())
        output_argmax = torch.argmax(output, dim=1)
    return output.cpu().numpy(), output_argmax.cpu().numpy()


class DiceLoss(object):
    def __init__(self, weight=None, apply_nonlin=None, ignore_label=False) -> None:
        self.weight = weight
        self.apply_nonlin = apply_nonlin
        self.ignore_label = ignore_label

    def __call__(self, x, y, epsilon=1e-6, mode='mean'):
        if self.apply_nonlin is not None: x = self.apply_nonlin(x)
        n, c = x.shape[:2]
        if self.weight: assert len(self.weight) == c, "check weight's length!"
        x = x.view(n, c, -1).permute(0, 2, 1).contiguous().view(-1, c)          # XC
        y = torch.eye(c, device=x.device)[y.view(n, -1).long()].view(-1, c)     # XC
        tp = torch.sum(x * y, dim=0)    # XC
        fp = torch.sum(x, dim=0)        # XC
        fn = torch.sum(y, dim=0)        # XC
        dice = (2. * tp + epsilon) / (fp + fn + epsilon)
        dice_loss = 1. - dice
        if self.weight:
            weight = torch.tensor(self.weight, device=x.device) / max(self.weight)
            dice_loss = weight * dice_loss  # [N, C]
        if self.ignore_label:
            dice_loss = dice_loss[::, 1:]
        if mode!="mean": 
            dice_loss = torch.sum(dice_loss)
        else:
            dice_loss = torch.mean(dice_loss)
        return dice_loss


if __name__=="__main__":
    # 定义数据集
    if len(train_path_list)!=0:
        train_dataset = SegmentationDataset(train_path_list, crop_size=crop_size, transform=get_transform())
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 定义数据集
    if len(val_path_list)!=0:
        val_dataset = SegmentationDataset(val_path_list, crop_size=crop_size, transform=get_transform(train=False))
        val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    # 定义模型
    model = Net(in_channels=in_channels, out_channels=out_channels).to(device)
    # 定义优化器
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # 定义损失函数
    loss_function = DiceLoss() # nn.CrossEntropyLoss()
    # 加载模型
    start_epoch = load_model(model, start_epoch)
    print('Start epoch: %d' % (start_epoch))
    # 训练模型
    for epoch in range(start_epoch+1, num_epochs):
        # 当前Epoch
        print('Epoch: %d' % (epoch))
        # 训练模型
        if train_path_list.__len__()!=0:
            train_amp_accumulate(model, train_loader, optimizer, loss_function, epoch, device=device)
            # 保存模型
            save_model(model, epoch)
        # 测试模型
        if val_path_list.__len__()!=0:
            val(model, val_loader, loss_function, device=device)
        # 预测模型
        if test_path_list.__len__()!=0:
            arr_dict = read_data(test_path_list[np.random.randint(0, test_path_list.__len__())])
            image = torch.from_numpy(arr_dict['image']).unsqueeze(0).unsqueeze(0).float().to(device)
            img_pred_logit, pred = predict(model, image, device=device)
            plt.imsave(f'img_pred_image.png', image.cpu().numpy()[0, 0, 50, :, :], cmap='gray')
            plt.imsave(f'img_pred_logit.png', img_pred_logit[0, 0, 50, :, :], cmap='gray')
            plt.imsave(f'img_pred_color.png', create_mask_color(np.max(pred[0], axis=0)))

            # 
            from IEDeep.tools.itktools import debug_saveitk
            debug_saveitk(image.cpu().numpy()[0,0], "image.mhd", dtype=np.float32, target=None, useCompression=True)
            debug_saveitk(pred[0], "image_pred.mhd", dtype=np.int16, target=None, useCompression=True)
            assert False

        # assert False
            
