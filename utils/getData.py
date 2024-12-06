import cv2 as cv
import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset

class data_a(Dataset):
    def __init__(self, ori = None, aug=None, 
                 folds=[1, 2, 3, 4, 5], subdir=['Test', 'Train', 'Test']):
        
        self.dataset = []
        onehot = np.eye(6)
        
        for fold in range(1, 6):
            if ori:
                path = (ori+f"/fold{fold}"+f"/{subdir}")
                for _, diese in enumerate(sorted(os.listdir(path))):
                    for img in os.listdir(path+"/"+diese):
                        image = cv.resize(cv.imread(path+"/"+diese+f"/{img}"), dsize=(32, 32)) / 255
                        self.dataset.append([image, onehot[_]])
            
            if aug and subdir=='Train':
                path = (aug+f"/fold{fold}_AUG/Train/")
                for _, diese in enumerate(sorted(os.listdir(path))):
                    for img in os.listdir(path+"/"+diese):
                        image = cv.resize(cv.imread(path+"/"+diese+f"/{img}"), dsize=(32, 32)) / 255
                        self.dataset.append([image, onehot[_]])
    
    def __len__(self):  
        return len(self.dataset)
    
    def __getitem__(self, item):
        features, label = self.dataset[item]
        return(torch.tensor(features, dtype=torch.float32),
               torch.tensor(label, dtype=torch.float32))