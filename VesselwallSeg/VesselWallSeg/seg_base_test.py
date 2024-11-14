import os

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.backends import cudnn
import time
import importlib


import xlwt

class SinglePatchDataSet2D(Dataset):
    # 用于测试的时候对图形截取block用
    def __init__(self, images, crop_size, stride, filling_value):
        assert images.dim() == 3, 'please input tensor with 3 dimensions'
        assert not(False in crop_size<=0),'please the crop_size must be lager than 0'
        assert not(False in stride<0), 'please the stride must be lager than 0'
        
        self.images = images  # torch,images.dim()==3 [chw]
        self.crop_size = np.array(crop_size)  # (2,) [yx]
        self.stride = np.array(stride)  # (2,) [yx]
        self.filling_value = filling_value  # 1

        # 向上取整 不够补零
        block_shape = np.ceil((np.array(images[0].numpy().shape) - crop_size) / stride) + 1
        self.block_shape = block_shape.astype(int)

    def __len__(self):
        # 和算卷积的输出尺寸差不多
        return self.block_shape[0] * self.block_shape[1]

    def __index2yx(self, index):
        # 给定块的索引 算出块的左上角顶点位置
        y = index//self.block_shape[1]
        x = index%self.block_shape[1]
        pos = np.array([y,x]) * self.stride
        return pos
    
    def __getitem__(self,index):
        
        pos = self.__index2yx(index)
        box = np.concatenate((pos,pos+self.crop_size))
        tensor_blocks = self._crop_box(box)
        tensor_box = torch.tensor(box,dtype=torch.float)
        
        return tensor_blocks,tensor_box
    
    def _crop_box(self,box_points):
        
        new_box_points = box_points.reshape((-1, 2)).astype(int)

        # 强制为最小点和最大点顺序
        min_yx = np.min(new_box_points, axis=0)
        max_yx = np.max(new_box_points, axis=0)
        new_box_points = new_box_points.ravel()
    
        # 计算box在图像中的部分 这部分直接
        tensor_image_dtype = self.images.dtype
        tensor_image_shape = self.images.shape
        box_size = np.array([tensor_image_shape[0],0,0]) #tensor is 3 dimensions
        box_size[1::]= np.abs(max_yx - min_yx)
        
        tensor_box = (torch.ones(tuple(box_size), dtype=tensor_image_dtype) * self.filling_value).type(tensor_image_dtype)
    
        # 先求出在图像上的部分
        imageindex_min_yx = np.maximum(min_yx, 0)
        imageindex_min_yx = np.minimum(imageindex_min_yx, tensor_image_shape[1::])
        imageindex_max_yx = np.maximum(max_yx, 0)
        imageindex_max_yx = np.minimum(imageindex_max_yx, tensor_image_shape[1::])
    
        # 然后求取这部分在box中的位置索引 box
        boxindex_min_yx = imageindex_min_yx - min_yx
        boxindex_max_yx = imageindex_max_yx - min_yx
    
        # 将原图像中的部分复制到box中
        tensor_box[:,
            boxindex_min_yx[0] : boxindex_max_yx[0],
            boxindex_min_yx[1] : boxindex_max_yx[1],
        ] = self.images[:,
            imageindex_min_yx[0] : imageindex_max_yx[0],
            imageindex_min_yx[1] : imageindex_max_yx[1],
        ]
        
        return tensor_box

class test_slide2D(object):
    def __init__(self,stride,crop_size,batch_size,class_num,fill_value = 0):
        assert stride is not None and len(stride) == 2,'only support 2D'
        assert crop_size is not None and len(crop_size) == 2,'only support 2D'
        self.stride = stride[::-1] #反一下，输入的时候是[x,y]，计算时候索引是反过来
        self.crop_size = crop_size[::-1]#反一下，输入的时候是[x,y]，计算时候索引是反过来
        self.class_num = class_num
        self.batch_size = batch_size
        self.fill_value = fill_value
        
    def __call__(self,net,im):
        
        im  = ToTensor()(im) #torch, 3 dims[cdhw]
        
        dataset = SinglePatchDataSet2D(im,self.crop_size,self.stride,self.fill_value)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False, pin_memory=True)
        predict = np.zeros([self.class_num,im.shape[1],im.shape[2]],dtype=np.float32)
        precount= np.zeros([im.shape[1],im.shape[2]],dtype=np.uint8)
        
        for idx,(blocks,box) in enumerate(dataloader):
            blocks = blocks.cuda()
            tempPred = net(blocks).cpu().numpy()
            box = box.cpu().numpy()
            
            predict,precount = self.__restore(predict,tempPred,box,precount)
        
        return predict,precount
    
    def __restore(self,heatmaps,pred,box,precount):
        """
        heatmaps: cdhw
        pred : ncdhw 里面的一些小块的预测
        pos : n4 小块的位置
        """
        assert np.all(box == np.round(box).astype(int))
        box = np.round(box).astype(int)
        image_shape_2 = np.tile(heatmaps[0].shape, 2)
        
        for idx,(patch,image_pos) in enumerate(zip(pred,box)):
            image_pos[image_pos<0] = 0
            image_pos[image_pos>image_shape_2] = image_shape_2[image_pos>image_shape_2]
            patch_pos = image_pos - np.tile(image_pos[0:2],2)
            tmp_image = heatmaps[:, image_pos[0] : image_pos[2], image_pos[1] : image_pos[3]]
            tmp_patch = patch[:, patch_pos[0] : patch_pos[2], patch_pos[1] : patch_pos[3]]
            
            # 交叠处 默认使用最大值来搞
            res = np.maximum(tmp_image, tmp_patch)
            
            heatmaps[:, image_pos[0] : image_pos[2], image_pos[1] : image_pos[3]] = res
            precount[image_pos[0] : image_pos[2], image_pos[1] : image_pos[3]] += np.ones([res.shape[1],res.shape[2]],dtype=np.uint8)
        
        return heatmaps,precount

class SegBaseTest(object):
    
    def __init__(self,net_file,gpu_ids = 0,save_results_path=None,net_id=0):
        
        if not os.path.isfile(net_file):
            raise ValueError('config not found:{}'.format(net_file))
        self.__state = torch.load(net_file, map_location=lambda storage, loc: storage.cuda(gpu_ids))
        self.__save_results_path = save_results_path
        
        if isinstance(net_id,str):
            self.__net_id = net_id
        else:
            self.__net_id = str(net_id)
        
        self.__Dice = None
        self.__Mean_Dice = None
        
        self.max_stride = 16
        self.gpu_ids = gpu_ids
        
        self.resampler = None
        self.labeler = None
        self.normalizer = None
        
        self.fill_value = 0
        self.class_num = 0
        
        if self.__state['net'].lower() in ['vnet2d']:
            self._net_type = 2
        else:
            self._net_type = 3
    
    def _net_initial(self):
        
        net_name = self.__state['net']
        input_channels = self.__state['input_channels']
        output_channels = self.__state['output_channels']
        self.class_num = output_channels
        
        cudnn.benchmark = True
        assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
        
        net_module = importlib.import_module('IADeep.seg.net.' + net_name)
        net = net_module.Net(input_channels,output_channels) #所有的net都得叫VNet，后续可以考虑就叫Net
        self.max_stride = net.max_stride()
        
        #net = nn.parallel.DataParallel(net,device_ids=self.gpu_ids)
        torch.cuda.set_device(self.gpu_ids)

        net = net.cuda()
        
        net.load_state_dict(self.__state['state_dict'])
        net.eval()
        
        return net
    
    def test(self,test_file_list,prob_thresh = 0.5,is_gt = True,is_heatmap = False,is_slide = False,stride=None,crop_size=None,batch_size=1):
        '''
        test_file_list:测试文件名
        prob_thresh：仅仅在只有前景和背景的情况下有效，多目标以最大值为准
        is_gt：是否存在GT，如果不存在GT无法计算Dice
        is_heatmap:是否保存heatmap
        is_slide:是否需要滑窗，滑窗测试
        stride：滑窗步长
        crop_size：裁剪大小
        batch_size:滑窗时候的网络前向计算的批大小
        '''
        
        label = self.__state['label']
        
        net =self._net_initial()
        
        self._transform()
        
        if is_gt is True:
            im_list,seg_list,_ = read_imlist_file(test_file_list)
            self.__Dice = np.zeros([len(seg_list),np.array(label).size])
            
        else:
            im_list,_ = read_imlist_file_without_gt(test_file_list)
            
        if is_slide:
            assert not (False in (np.array(crop_size)%self.max_stride == 0)),'crop_size must be {:d} times'.format(self.max_stride)
            
            if self._net_type == 2:
                test_fun = test_slide2D(stride,crop_size,batch_size,self.class_num,self.fill_value)
            else:
                test_fun = test_slide3D(stride,crop_size,batch_size,self.class_num,self.fill_value)
        else:
            test_fun = test_no_slide()
        
        with torch.no_grad():
            for index in range(len(im_list)):
                start_t = time.time()
                im_path  = im_list[index]
                print(im_path)
                
                im = imio.read_image(im_path)
                #transform
                
                im = self.resampler(im)
                
                if self.normalizer is not None:
                    self.normalizer(im)
                
                heatMap,precount = test_fun(net,im) #输入image3d，返回是numpy

                
                if heatMap.shape[0] == 2:
                    heatMap = heatMap[1]
                    predict = heatMap > prob_thresh
                    predict = predict.astype(np.uint8)
                    predict[predict == 1] = label
                    predict = self.resampler.Inv_Seg(predict)
                    heatMap_i = self.resampler.Inv_HeatMap(heatMap)
                else:
                    predict=convert_binary_to_multi_label(heatMap,label)
                    predict = self.resampler.Inv_Seg(predict)
                    heatMap_i = []
                    for i in range(heatMap.shape[0]):
                        heatMap_i.append(self.resampler.Inv_HeatMap(heatMap[i]))

                if is_gt:
                    seg_path = seg_list[index]
                    seg = read_image(seg_path)
                    seg_np = seg.to_numpy()
                    pre_np = predict.to_numpy()
                    
                    if isinstance(label,list) is False:
                        label = [label]
                    
                    dice = []
                    
                    for i in range(len(label)):
                        intersection = np.sum((pre_np == label[i])*(seg_np == label[i]))
                        union = np.sum(pre_np == label[i])+np.sum(seg_np == label[i])
                        dice.append(2*intersection/union)
                        
                    self.__Dice[index,:]=dice
                    print('Dice:',dice)
                
                # save result
                if isinstance(im_path,list):
                    out_path = im_path[0]
                else:
                    out_path = im_path
                
                pre_file_name,ext = os.path.splitext(out_path)
                
                pre_path = pre_file_name + '_net_'+self.__net_id+'_pre'+ext

                if precount is not None:
                    precount = self.resampler.Inv_Seg(precount)
                    precount_path = pre_file_name + '_net_'+self.__net_id+'_precount'+ext

                sample_duration = (time.time() - start_t) * 1.0 
                print(pre_path + 'is saved and the excute time:',sample_duration)
                
                # save HeatMap
                if is_heatmap:
                    if isinstance(heatMap_i,Image3d):
                        heat_path = pre_file_name+'_net_'+self.__net_id+'_heat'+ext
                    else:
                        for i in range(len(heatMap_i)):
                            heat_path = pre_file_name+'_net_'+self.__net_id+'_heat_'+str(i)+ext

        if is_gt:
            self.__Mean_Dice = np.mean(self.__Dice)
            print("mean dice:",self.__Mean_Dice)
            self.save_to_excel(im_list,self.__Dice,label)
    
    def get_dice(self):
        print("mean dice:",self.__Mean_Dice)
        return (self.__Dice,self.__Mean_Dice)

