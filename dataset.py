import numpy as np
import random
import torch 
from torch.utils.data import Dataset

class MCSDataset(Dataset):
    def __init__(self, num_frames = 1, split = 'train', translation = True):
        super().__init__()

        self.num_frames = num_frames
        self.split = split
        self.translation = translation

        if self.split not in ['train', 'val', 'split']:
            raise ValueError(f"{self.split} is not a valid split")
        
    def action_to_label(self, action):
        return self._action_to_label[action]
    
    def get_pose_data(self, data_index, frame_idx):
        pose = self._load(data_index, frame_idx)
        label = self.get_label(data_index)
        return pose, label 

    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        


        


