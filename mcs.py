import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class MCSPoses(Dataset):
    dataname = "mcs"

    def __init__(self, datapath="dataset/MCSPoses", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "mcsposes.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]

        self._actions = [x for x in data["y"]]

        total_num_actions = 14
        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = mcs_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 20, 3)
        return pose


mcs_coarse_action_enumerator ={0: 'LLT', 
                               1: 'RLT', 
                               2: 'BAPF', 
                               3: 'RCMJ', 
                               4: 'BAP', 
                               5: 'RLTF', 
                               6: 'CMJ', 
                               7: 'SQT', 
                               8: 'PUF', 
                               9: 'LCMJ', 
                               10: 'LSLS', 
                               11: 'RSLS', 
                               12: 'LLTF', 
                               13: 'PU'}


