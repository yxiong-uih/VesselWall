# -*- coding: utf-8 -*-
import os
import time
import shutil
import logging
import math
import importlib

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn

import numpy as np



def load_config(config_file):
    """
    Load a training config file
    :param config_file: the path of config file with py extension
    :return: a configuration dictionary
    """
    dirname = os.path.dirname(config_file)
    basename = os.path.basename(config_file)
    modulename, _ = os.path.splitext(basename)
    os.sys.path.append(dirname)
    lib = importlib.import_module(modulename)
    del os.sys.path[-1]
    return lib.cfg

class SegLogger(object):
    r"""
    setup logger for logging training messages
    :param log_file: the location of logger file
    :param log_name: the name of logger
    """
    def __init__(self,log_file,log_name = 'info'):
        dirname = os.path.dirname(log_file)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.info_lg = logging.getLogger(log_name)
        self.info_lg.setLevel(logging.INFO)
        
        if len(self.info_lg.handlers)==0:
            self.handler = logging.FileHandler(log_file)
            self.handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.handler.setFormatter(formatter)
            
            self.console = logging.StreamHandler()
            self.console.setLevel(logging.INFO)
            self.console.setFormatter(formatter)
            
            self.info_lg.addHandler(self.handler)
            self.info_lg.addHandler(self.console)
        
    def info(self,msg):
        self.info_lg.info(msg)
    
    def release(self):
        self.info_lg.removeHandler(self.handler)
        self.info_lg.removeHandler(self.console)
        logging.shutdown() 

class SegBaseTrain(object):
    def __init__(self,config_file):
        if not os.path.isfile(config_file):
            raise ValueError('config not found:{}'.format(config_file))
        cfg = load_config(config_file)
        self._config_file = config_file
        self._general = cfg.general
        self._interaction = cfg.interaction
        self._dataset = cfg.dataset
        self._loss = cfg.loss
        self._net = cfg.net
        self._train = cfg.train

        self.resampler = None

        self._batch_count = 0
        
        if self._net.name.lower() in ['vnet2d','vnet2d_multi_channels','unet2d', 'srnet2d']:
            self._net.type = 2
        else:
            self._net.type = 3
        
        self.DiceCalculator=DiceCalculate_Tensor(class_num=self._net.output_channels)
        
        if 'imseg_dir' not in self._general:
            self._general.imseg_dir = ''
        if 'lr_resume' not in self._train:
            self._train.lr_resume = False
        
        if 'save_hard_dice' not in self._train and 'save_hard_dsc_loss' in self._train:
            print('train.save_hard_dsc_loss is not used in future version,please modify as train.save_hard_dsc')
            self._train.save_hard_dice = self._train.save_hard_dsc_loss
        
    def _load_net_module(self):
        r"""
        Load network module
        :param net_name: the name of network module
        :return: the module object
        """
        lib = importlib.import_module('IADeep.seg.net.' + self._net.name) #以名字的方式导入
        return lib
    
    def _load_checkpoint(self, net):
        r"""
        load network parameters from directory
        :param net: the network object
        :return: loaded epoch index, loaded batch index
        """
        epoch_idx = self._general.resume_epoch
        chk_file = os.path.join(self._general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
        if not os.path.isfile(chk_file):
            raise ValueError('checkpoint file not found: {}'.format(chk_file))
        state = torch.load(chk_file)
        net.load_state_dict(state['state_dict'])
        return (state['epoch'], state['batch']) 
    
    def _save_checkpoint(self,net, epoch_idx, batch_idx):
        r"""
        save model and parameters into a checkpoint file (.pth)
        :param net: the network object
        :param epoch_idx: the epoch index
        :param batch_idx: the batch index
        :return: None
        """
        chk_folder = os.path.join(self._general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
        if not os.path.isdir(chk_folder):
            os.makedirs(chk_folder)
        filename = os.path.join(chk_folder, 'params.pth'.format(epoch_idx))
        state = {'epoch': epoch_idx,
         'batch': batch_idx,
         'net': self._net.name,
         'state_dict': net.state_dict(),
         'label':           self._dataset.label,
         'spacing':         self._dataset.spacing,
         'sampling_method': self._dataset.sampling_method,
         'normalizer':      self._dataset.normalize_method,
         'clip':            self._dataset.normalize_clip,
         'mean_stddev':     self._dataset.normalize_mean_stddev,
         'min_max_v':       self._dataset.normalize_min_max_v,
         'min_max_p':       self._dataset.normalize_min_max_p,
         'min_max_t':       self._dataset.normalize_min_max_t,
         'input_channels':  self._net.input_channels,
         'output_channels': self._net.output_channels
         }
        torch.save(state, filename)
        
        shutil.copy(self._config_file,os.path.join(chk_folder, 'config.py'))
    
    def _save_final_model(self,net):
        
        state = {'net':     self._net.name,
         'state_dict':      net.state_dict(),
         'label':           self._dataset.label,
         'spacing':         self._dataset.spacing,
         'sampling_method': self._dataset.sampling_method,
         'normalizer':      self._dataset.normalize_method,
         'clip':            self._dataset.normalize_clip,
         'mean_stddev':     self._dataset.normalize_mean_stddev,
         'min_max_v':       self._dataset.normalize_min_max_v,
         'min_max_p':       self._dataset.normalize_min_max_p,
         'min_max_t':       self._dataset.normalize_min_max_t,
         'input_channels':  self._net.input_channels,
         'output_channels': self._net.output_channels
         }
        
        filename = os.path.join(self._general.save_dir,'net.model') # save net with processing infomation
        torch.save(state, filename)
        filename = os.path.join(self._general.save_dir,'net_dict.model') #save net with dict(weights)
        torch.save(net.state_dict(), filename)
        
        shutil.copy(self._config_file,os.path.join(self._general.save_dir, 'train_config.py'))   
    
    def _net_initial(self):
        r"""
        net initial
        """
        
        gpu_ids = self._general.gpu_ids
        
        net_module = self._load_net_module()
        input_channels = self._net.input_channels
        output_channels = self._net.output_channels
        net = net_module.Net(input_channels,output_channels)
        max_stride = net.max_stride()
        
        if self._net.init.lower() == 'kaiming':
            net_module.vnet_kaiming_init(net)
        elif self._net.init.lower() == 'xavier':
            net_module.vnet_xavier_init(net)
        else:
            raise ValueError('unknown init method:{}'.format(self._net.init))

        #net = nn.parallel.DataParallel(net,[gpu_ids])
        torch.cuda.set_device(gpu_ids)
        net = net.cuda()

        if self._dataset.crop_size[0] % max_stride != 0 or self._dataset.crop_size[1] % max_stride != 0 or ( self._net.type == 3 and self._dataset.crop_size[2] % max_stride != 0):
        #if self._dataset.crop_size[0] % max_stride != 0 or self._dataset.crop_size[1] % max_stride != 0:
            raise ValueError('crop size not divisible by max_stride')
        
        return net

    def _loss_function(self):
        """
        loss_function
        now support 'dice','focal','wfocal','cross-entropy','mse'
        """
        loss_func = None
        if self._loss.name.lower() == 'dice':
            class_num = self._net.output_channels
            if class_num == 2:
                loss_func = BinaryDiceLoss(net_type=self._net.type)
            else:
                loss_func = SoftDiceLoss(class_num)
        elif self._loss.name.lower() == 'focal':
            class_num = self._net.output_channels
            if isinstance(self._loss.focal_obj_alpha,float):
                if class_num == 2:
                    alpha = [1-self._loss.focal_obj_alpha,self._loss.focal_obj_alpha]
                else:
                    alpha = None
            else:
                assert len(self._loss.focal_obj_alpha) == class_num, 'the length of focal_obj_alpha must be equal to the class number '
                alpha = self._loss.focal_obj_alpha
            gamma = self._loss.focal_gamma
            loss_func = FocalLoss(class_num=class_num,alpha=alpha, gamma=gamma)
        elif self._loss.name.lower() == 'wfocal':
            class_num = self._net.output_channels
            if isinstance(self._loss.focal_obj_alpha,float):
                alpha = [1-self._loss.focal_obj_alpha,self._loss.focal_obj_alpha]
            else:
                assert len(self._loss.focal_obj_alpha) == class_num, 'the length of focal_obj_alpha must be equal to the class number '
                alpha = self._loss.focal_obj_alpha
            gamma = self._loss.focal_gamma
            loss_func = WFocalLoss(class_num=class_num,alpha=alpha, gamma=gamma)
        elif self._loss.name.lower() == 'ce':
            class_num = self._net.output_channels
            if isinstance(self._loss.focal_obj_alpha,float):
                if class_num == 2:
                    alpha = [1-self._loss.focal_obj_alpha,self._loss.focal_obj_alpha]
                else:
                    alpha = None
            else:
                assert len(self._loss.focal_obj_alpha) == class_num, 'the length of focal_obj_alpha must be equal to the class number '
                alpha = self._loss.focal_obj_alpha
                
            loss_func=NLLLoss_s(class_num=class_num,weight=alpha)
            
        elif self._loss.name.lower() == 'weightedce':
            weight = np.array(self._loss.weight)
            weight = torch.from_numpy(weight).cuda()
            loss_func = WeightedBCELoss2d(weight=weight)
        elif self._loss.name.lower() == 'mse':
             class_num = self._net.output_channels
             loss_func = MSELoss_s(class_num=class_num)
        else:
            raise ValueError('unknown loss function:{}'.format(self._loss.name))
        
        return loss_func
        
    def _optimizer(self,net):
        r"""
        optimimizer offers SGD and Adam methods
        """
        
        opt = None
        
        if self._train.optimizer.lower() =='sgd':
            opt = optim.SGD(net.parameters(), lr=self._train.lr_init)
        elif self._train.optimizer.lower() =='adam':
            opt = optim.Adam(net.parameters(), lr=self._train.lr_init, betas=self._train.betas)
        else:
            raise ValueError('unknown optimizer {}'.format(self._train.optimizer))
        
        return opt
       
    def _lr_scheduler(self,opt):
        r"""
        scheduler
        'StepLR': Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs.
        'MultiStepLR': Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones. 
        'ExponentialLR': Set the learning rate of each parameter group to the initial lr decayed by gamma every epoch. 
        'LambdaLR':Sets the learning rate of each parameter group to the initial lr times a given function. 
        'CosineAnnealingLR': Set the learning rate of each parameter group using a cosine annealing schedule
        """
        
        scheduler = None
        
        last_epoch = -1
        
        if self._train.lr_scheduler.lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(opt,self._train.lr_step_size,self._train.lr_gamma,last_epoch)
        elif self._train.lr_scheduler.lower() == 'multisteplr':
            scheduler = optim.lr_scheduler.MultiStepLR(opt,self._train.lr_milestones,self._train.lr_gamma,last_epoch)
        elif self._train.lr_scheduler.lower() == 'exponentiallr':
            scheduler = optim.lr_scheduler.ExponentialLR(opt,self._train.lr_gamma,last_epoch)
        elif self._train.lr_scheduler.lower() == 'lambdalr':
            lr_decay_lambda = lambda epoch: self._train.lr_gamma**(epoch // self._train.lr_epoch_per_decay,last_epoch)
            scheduler = optim.lr_scheduler.LambdaLR(opt,lr_lambda=lr_decay_lambda)
        elif self._train.lr_scheduler.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,self._train.lr_cosine_Tmax,self._train.lr_consine_eta_min,last_epoch)
        else:
            raise ValueError('unknown lr_scheduler {}'.format(self._train.lr_scheduler))
        
        return scheduler
    
    def _transforms(self):
        
        transforms=[]
        
        label = self._dataset.label
        
        if type(label) is list or label > 0:
            labeler = Labeler_s(label)
            transforms.append(labeler)
            
        if self._dataset.sampling_method is not None:
            resampler = Resampler_s(self._dataset.spacing,self._dataset.sampling_method)
            transforms.append(resampler)
        
        if self._dataset.crop_mode is not -1:
            cropper = RandomCropper_s(self._dataset.crop_size,self._dataset.crop_mode,label = label,label_resort = True)
            transforms.append(cropper)
        
        if self._dataset.normalize_method is not None:
            if self._dataset.normalize_method.lower() == 'fix':
                mean_stddev = self._dataset.normalize_mean_stddev
                normalizer = FixedNormalizer_s(mean=mean_stddev[0],stddev=mean_stddev[1],clip=self._dataset.normalize_clip)
            elif self._dataset.normalize_method.lower() == 'adaptive':
                min_max_v = self._dataset.normalize_min_max_v
                min_max_p = self._dataset.normalize_min_max_p
                normalizer = AdaptiveNormalizer_s(min_max_v[0],min_max_v[1],self._dataset.normalize_clip,min_max_p[0],min_max_p[1])
            elif self._dataset.normalize_method.lower() == 'adaptive_t':
                min_max_v = self._dataset.normalize_min_max_v
                min_max_t = self._dataset.normalize_min_max_t
                normalizer = AdaptiveNormalizer_t_s(min_max_v[0],min_max_v[1],self._dataset.normalize_clip,min_max_t[0],min_max_t[1])
            else:
                 raise ValueError('Unknown method of normalization. Normalizer only supports Fix, Adaptive or Adaptive_t')
            transforms.append(normalizer)
        
        return transforms
        
    def get_train_dataloader(self,transforms):
        
        train_dataset    = SegBaseDataset(self._general.imseg_train_list,transforms,self._general.imseg_dir)
        train_dataloader = DataLoader(train_dataset,batch_size = self._train.batchsize,shuffle = True,num_workers=self._train.num_threads,pin_memory=True)
        
        return (train_dataloader,len(train_dataset))
    
    def get_val_dataloader(self,transforms):

        val_dataset    = SegBaseDataset(self._general.imseg_val_list,transforms,self._general.imseg_dir)
        val_dataloader = DataLoader(val_dataset,batch_size = self._train.batchsize,shuffle = False,num_workers=self._train.num_threads,pin_memory=True)
        
        return (val_dataloader,len(val_dataset))
    
    def _pre_operator(self,epoch_idx,net):
        
        return
    
    def _save_hard_data(self,sample,pred,epoch_idx,batch_idx):
        
        if self._train.save_hard_results == -1:
            return
        
        image = sample['image']
        target = sample['label']
        im_path = sample['im_path']
        seg_path = sample['seg_path']
        spacing = sample['spacing']
        #####
        # global flagsave
        # if flagsave:
        #     print('save gt')
        #     seg = Image3d()
        #     seg.from_numpy(target[0].numpy().astype(np.uint8))
        #     seg.set_spacing(spacing[0])
        #     imio.write_image(seg,'G:/Vnet/PlaqueSeg/Result/gt.mhd',True)
        #     flagsave =False
        # #####
        
        dice=self.DiceCalculator(pred,target,batch_average=False)
        
        hard_sample_path = os.path.join(self._general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
        hard_sample_file = os.path.join(self._general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx),'hard_sample.txt')
        
        for i in range(len(seg_path)):
            dice_i = dice[i,1:].numpy()
            if np.mean(dice_i)<self._train.save_hard_dice:
                if self._train.save_hard_results == 0 or self._train.save_hard_results == 1:
                    
                    im_path_i=im_path[i]
                    seg_path_i = seg_path[i]
                    with open(hard_sample_file, "a") as f:
                        if isinstance(im_path_i,str):
                            f.writelines(im_path_i)
                            f.writelines('/n')
                        else:
                            for j in range(len(im_path_i)):
                                f.writelines(im_path_i[j])
                                f.writelines('/n')
                        f.writelines(seg_path_i)
                        f.writelines('/n')
                        
                    if self._train.save_hard_results == 1:
                        image_i = image[i].numpy()
                        target_i = target[i].numpy()
                        pred_i = pred[i].detach().numpy()
                        pred_i = imtool.convert_binary_to_multi_label(pred_i,self._dataset.label)
                        dice_str = '_b_'+str(batch_idx)+'_d'
                        for j in range(dice_i.size):
                            dice_str += '_{:.2f}'.format(dice_i[j])
                            
                        p_path,seg_filename = os.path.split(seg_path_i)
                        _,p_path=os.path.split(p_path)
                        if isinstance(im_path_i,str):
                            im_path_i=[im_path_i]
                        for j in range(len(im_path_i)):
                            _,im_j_name = os.path.split(im_path_i[j])
                            im_j_name,ext=os.path.splitext(im_j_name)
                            if ext is '.dcm':
                                ext = '.mhd'
                            im_j_path = os.path.join(hard_sample_path,p_path+im_j_name+'_b_'+str(batch_idx)+ext)
                            im_j = Image3d()
                            im_j.from_numpy(image_i[j])
                            im_j.set_spacing(spacing[i])
                            imio.write_image(im_j,im_j_path)
                        
                        seg_name,ext=os.path.splitext(seg_filename)
                        if ext is '.png':
                            ext = '.mhd'
                        seg_path_t = os.path.join(hard_sample_path,p_path+seg_name+'_b_'+str(batch_idx)+ext)
                        seg = Image3d()
                        seg.from_numpy(imtool.convert_multi_label_1_to_multi_label(target_i[0].astype(np.uint8),self._dataset.label))
                        seg.set_spacing(spacing[i])
                        imio.write_image(seg,seg_path_t,True)
                        
                        pre_path = os.path.join(hard_sample_path,p_path+seg_name+'_pred_'+dice_str+ext)
                        pre = Image3d()
                        pre.from_numpy(pred_i.astype(np.uint8))
                        pre.set_spacing(spacing[i])
                        imio.write_image(pre,pre_path,True)
    
    def _train_epoch(self,net,loss_func,opt,transforms,logger,log_file,epoch_idx):
        
        dataloader,train_data_len = self.get_train_dataloader(transforms)
        
        for index, sample in enumerate(dataloader,1):
            
            start_t = time.time()
            image  = sample['image'].cuda()
            target = sample['label'].cuda()
            opt.zero_grad()
            pred = net(image)
            
            train_loss = loss_func(pred,target)
            train_loss.backward(retain_graph=True)
            opt.step()
            
            batch_idx = epoch_idx * math.ceil(train_data_len / self._train.batchsize) + index-1
            sample_duration = (time.time() - start_t) * 1.0 / self._train.batchsize
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
            msg = msg.format(epoch_idx, batch_idx, train_loss, sample_duration)
            
            logger.info(msg)
            if batch_idx % self._train.plot_snapshot == 0:
                dloss_plot_file = os.path.join(self._general.save_dir, 'train_loss.html')
                plot_loss(log_file, dloss_plot_file, name='loss', display=self._loss.name+' loss')
                
            if epoch_idx % self._train.save_epochs == 0:
                self._save_checkpoint(net, epoch_idx, batch_idx)
            
            if batch_idx>= self._train.save_hard_start_batch:
                self._save_hard_data(sample,pred.cpu(),epoch_idx,batch_idx)

    def _val_epoch(self,net,loss_func,opt,transforms,logger,log_file,epoch_idx):
        
        dataloader,val_data_len = self.get_val_dataloader(transforms)
        loss_sum = 0
        dice_sum = 0
        start_t = time.time()
        
        net.train(mode=False)
        
        with torch.no_grad():
            for index, sample in enumerate(dataloader,1):
                image = sample['image'].cuda()
                target = sample['label'].cuda()
                pred = net(image)
                train_loss = loss_func(pred,target)
                dice       = self.DiceCalculator(pred.cpu(),sample['label'],all_average=True)
                loss_sum += train_loss.item()
                dice_sum += dice.item()
        
        val_duration = (time.time() - start_t) * 1.0 /val_data_len/self._train.batchsize
        
        msg = 'epoch: {}, val_loss: {:.4f}, val_dice:{:.4f},time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, loss_sum/index, dice_sum/index,val_duration)
        logger.info(msg)
        dloss_plot_file = os.path.join(self._general.save_dir, 'val_loss.html')
        plot_loss(log_file, dloss_plot_file, name='loss', display=self._loss.name+ ' loss',iter_word='epoch')
        dloss_plot_file = os.path.join(self._general.save_dir, 'val_dice.html')
        plot_loss(log_file, dloss_plot_file, name='dice', display='Dice',iter_word='epoch')
        
        net.train(mode=True)

    def obtain_sr_model(self, input_channels, output_channels):

        net_name = r'srnet2d'
        net_module = importlib.import_module('IADeep.seg.net.' + net_name)
        net = net_module.Net(input_channels, output_channels)  # 所有的net都得叫VNet，后续可以考虑就叫Net

        # net = nn.parallel.DataParallel(net,device_ids=self.gpu_ids)
        gpu_ids = 0
        torch.cuda.set_device(gpu_ids)

        net = net.cuda()
        net_file = r'F:/MRPlaque/VesselWallSegmentation/Result/3_BackUp/checkpoints/chk_180/params.pth'
        net_state = torch.load(net_file, map_location=lambda storage, loc: storage.cuda(gpu_ids))
        net.load_state_dict(net_state['state_dict'])
        net.eval()

        return net


    def train_model(self):
        if self._general.seed>= 0:
            np.random.seed(self._general.seed)
            torch.manual_seed(self._general.seed)
            torch.cuda.manual_seed(self._general.seed)
        print(self._general.save_dir)
        
        if self._general.resume_epoch < 0 and os.path.isdir(self._general.save_dir) and self._general.clear_save_dir:
            print('clear the save dir.')
            shutil.rmtree(self._general.save_dir)

        cudnn.benchmark = True
        assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
        
        net = self._net_initial() #initial net
        opt = self._optimizer(net) #get optimizer
        
        loss_func = self._loss_function() #get loss function
        
        train_log_file = os.path.join(self._general.save_dir, 'train_log.txt')
        train_logger = SegLogger(train_log_file,'train_log')
        
        val_log_file = os.path.join(self._general.save_dir, 'val_log.txt')
        val_logger = SegLogger(val_log_file,'val_log')
        
        if self._general.resume_epoch >= 0:
            epoch_start, batch_start = self._load_checkpoint(net) #load epoch net param
            epoch_start = epoch_start + 1
            batch_start = batch_start + 1
            self._batch_count = batch_start
        else:
            epoch_start = 0
        
        scheduler = self._lr_scheduler(opt) # set lr scheduler
        
        transforms = self._transforms() #image transform functions,(list)

        netsr = self.obtain_sr_model(self._net.input_channels, self._net.output_channels)

        for e in range(self._train.epochs):
            self._pre_operator(e+epoch_start,net)
            if self._train.lr_resume is True:
                scheduler.step(epoch=e)
            else:
                scheduler.step(epoch=e+epoch_start)
            #self._train_epoch_SR(net, loss_func, opt, transforms, train_logger, train_log_file, e + epoch_start)
            self._train_epoch_addSR(net, netsr, loss_func, opt, transforms, train_logger, train_log_file, e + epoch_start)
            if self._general.imseg_val_list is not None:
                self._val_epoch(net, loss_func, opt, transforms, val_logger, val_log_file, e + epoch_start)
            self._save_final_model(net)

        train_logger.release()
        val_logger.release()
        print(' Finish Training !')

        