import glob
import os
from typing import Any

import torch
from natsort import natsorted


class Restore(object):
    def __init__(self, ckpt_dir) -> None:
        self.ckpt_dir = ckpt_dir
        pass
    def __call__(self, model, resume_epoch=-1, optimizer=None, scheduler=None) -> Any:
        
        ckpt_dir = os.path.abspath(self.ckpt_dir)
 
        ckpt_filter = f'*_epoch_{resume_epoch}*' if resume_epoch!=-1 else "*.ckpt"
        ckpt_names = glob.glob(ckpt_dir + os.sep + ckpt_filter)
        ckpt_path = ""
        if ckpt_names!=[]:
            ckpt_name = natsorted(ckpt_names)[-1]
            ckpt_name = os.path.basename(ckpt_name)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path)
            resotre_state_dict = state['model_state_dict']

            # if optimizer is not None:
            #     optimizer_state_dict =  state.get('optimizer_state_dict', None)
            #     print(optimizer_state_dict)
            #     print(optimizer_state_dict.param_groups)
            #     if optimizer_state_dict is not None:
            #         optimizer.load_state_dict(optimizer_state_dict)
            # if scheduler is not None:
            #     scheduler_state_dict =  state.get('scheduler_state_dict', None)
            #     if scheduler_state_dict is not None:
            #         scheduler.load_state_dict(scheduler_state_dict)

            # # 重新收集
            # original_state_dict = model.state_dict()
            # skip_keys = []
            # for k in resotre_state_dict.keys():
            #     if k not in original_state_dict: 
            #         continue
            #     if resotre_state_dict[k].size() != original_state_dict[k].size(): 
            #         skip_keys.append(k)
            # for k in skip_keys: 
            #     del resotre_state_dict[k]

            missing_keys, unexpected_keys = model.load_state_dict(resotre_state_dict, strict=False)

            resume_epoch = int(ckpt_name[ckpt_name.rfind("epoch"):].split("_")[1])
            resume_epoch = resume_epoch + 1

            # score = state.get("score", None)
            # score = f"score {score:.4f}" if score is not None else ''
            # print(f"model load from '{ckpt_path}', start epoch from {resume_epoch}. {score}")

        return model, resume_epoch
