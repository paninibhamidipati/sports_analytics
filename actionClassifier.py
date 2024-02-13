import numpy as np
import torch 
import torch.nn as nn
import pickle as pkl
import os 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from einops import rearrange



class MCSDataset(Dataset):
    def __init__(self, split = 'train', datapath = './', dataset_name = 'mcsposes.pkl', min_frames=100, data_filter=True):
        super().__init__()

        self.split = split
        self.datapath = datapath
        self.dataset_name = dataset_name
        self.min_frames = min_frames

        if self.split not in ['train', 'val', 'split']:
            raise ValueError(f"{self.split} is not a valid split")
        
        pkldatafilepath = os.path.join(datapath, "mcsposes.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))
        
        poses = data['joints3D']
        actions = data['y']
        self.MCSPoses = []
        self.actions = []

        if data_filter:
            for i, pose in enumerate(poses):
                if pose.shape[0] > self.min_frames:
                    self.MCSPoses.append(poses[i][:100, :, :])
                    self.actions.append(actions[i])
        else:
            self.MCSPoses = poses
            self.actions = actions

        self.actions = torch.tensor(self.actions)
        self.num_samples = len(self.MCSPoses)

        self.video_lengths = [pose.shape[0] for pose in self.MCSPoses]
        self.unique_actions = len(torch.unique(self.actions))


    def __getitem__(self, index):
        pose = self.MCSPoses[index]
        label = self.actions[index]
        return pose, label


    def __len__(self):
        return self.num_samples
    
    def get_video_lengths(self, x):
        return self.video_lengths

    def plot_video_len_dist(self):
        plt.hist(self.video_lengths, bins=20, color='#ADD8E6', edgecolor='black')
        plt.title('Histogram of video lengths')
        plt.xlabel('Number of Frames')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('video_len_dist.png')
        

dataset = MCSDataset()


batch_size = 4
learning_rate = 0.001

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1)

dataiter = iter(dataloader)
sample = dataiter.next()
pose, label = sample
print(len(dataloader))
print(pose.shape, label)

class LinearModel(nn.Module):
    def __init__(self, input_features, num_actions=14):
        super().__init__()
        self.input_features = input_features
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_features, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, self.num_actions)
    
    def forward(self, x):
        out = nn.ReLU()(self.fc1(x))
        out = nn.ReLU()(self.fc2(out))
        out = self.fc3(out)
        out = nn.Sigmoid()(out)

        return out 

class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size=256, num_actions=14):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_actions)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)

        return out


# model = LinearModel(6000)
model = LSTMModel(60)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 50

# x = torch.randn(batch_size, 100, 20, 3)
# print(x[:, :100, :, :].shape)
# x = rearrange(x, 'b n h w -> b (n h w)')
# print(x.shape)
# out = model(x)
# print(out.shape)
n_total_steps = len(dataloader)
losses = []

model_name = "LSTM"
for epoch in range(num_epochs):
    for i, (motion, label) in enumerate(dataloader):
        
        # motion = motion[:, :100, :, :]
        if model_name == 'Linear':
            motion = rearrange(motion, 'b n h w -> b (n h w)')
        if model_name == 'LSTM':
            b, t, h, w = motion.size()
            motion = motion.view(b, t, h * w)
        motion = motion.float()
        output = model(motion)
        loss = criterion(output, label)

        optimizer.zero_grad()
        optimizer.step()
        losses.append(loss)

        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss}')
        
with torch.no_grad():
    plt.plot(losses)
    plt.savefig('action_classification_loss.png')


        










