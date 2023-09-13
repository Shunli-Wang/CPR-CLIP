import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import random
from random import sample
from torch.utils.data import Dataset
from itertools import combinations

class SingleErrorDataset(Dataset):
    ''' Compose features from 13 classes. '''
    def __init__(self, featDataName):
        # Load feature
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabel = np.array([i[0][0] for i in self.data]) # (896, )
        self.allData = np.array([i[1] for i in self.data])

        if self.data[0][1].shape[0] == 8*2048:
            self.allData = np.array([i[1].reshape(8, 2048).mean(0) for i in self.data])
        else:
            self.allData = np.array([i[1] for i in self.data])

        self.initDataset()

    def initDataset(self):

        # Find all cls idx
        self.idxCls = [np.where(self.allLabel == i)[0] for i in range(14)]

        # Combine
        self.labelList, self.featList = [], []

        # # 1
        # combin = [[i] for i in range(1, 14)]
        # # Store all fake data
        # for combinElem in combin:
        #     crossIdxList1 = self.idxCls[combinElem[0]]
        #     for i in range(len(self.idxCls[0])):
        #         selectedIdx1 = random.randint(0, len(crossIdxList1)-1) 
        #         idx1 = crossIdxList1[selectedIdx1]
        #         feat1 = self.allData[idx1]

        #         self.featList.append([feat1]) # Add
        #         self.labelList.append(list(combinElem)) 

        # 2 
        combin2 = list(combinations([i for i in range(1, 14)], 2)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...
        combin2 = [list(i) for i in combin2]
        combin2 = sample(combin2, int(len(combin2)/2))
        # Store all fake data
        for combinElem in combin2:
            random.shuffle(combinElem) # Change the sequence
            crossIdxList1 = self.idxCls[combinElem[0]]
            crossIdxList2 = self.idxCls[combinElem[1]]
            for i in range(len(self.idxCls[0])):
                selectedIdx1 = random.randint(0, len(crossIdxList1)-1)
                selectedIdx2 = random.randint(0, len(crossIdxList2)-1)
                idx1, idx2 = crossIdxList1[selectedIdx1], crossIdxList2[selectedIdx2]
                feat1, feat2 = self.allData[idx1], self.allData[idx2]
                
                randWeight = np.random.rand() # get random weight 

                self.labelList.append(list(combinElem))

                randWeightEnable = 1 # 0 for disable; 1 for enable
                
                if randWeightEnable == 0:
                    self.featList.append([feat1, feat2]) # Add
                elif randWeightEnable == 1:
                    self.featList.append([feat1 * randWeight, feat2 * (1 - randWeight)]) # Split
        
        # 3 
        # combin3 = [[1, 8, 10], [1, 8, 12], [5, 7, 9], [4, 6, 10], [1, 5, 10], [1, 5, 9], [1, 5, 6], [1, 4, 10], [1, 4, 9], [1, 4, 6], ]
        combin3 = list(combinations([i for i in range(1, 14)], 3)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...
        combin3 = [list(i) for i in combin3] 
        combin3 = sample(combin3, int(len(combin3)/2))
        for combinElem in combin3:
            random.shuffle(combinElem) # Change the sequence 
            crossIdxList1 = self.idxCls[combinElem[0]]
            crossIdxList2 = self.idxCls[combinElem[1]]
            crossIdxList3 = self.idxCls[combinElem[2]]
            for i in range(len(self.idxCls[0])):
                selectedIdx1 = random.randint(0, len(crossIdxList1)-1)
                selectedIdx2 = random.randint(0, len(crossIdxList2)-1)
                selectedIdx3 = random.randint(0, len(crossIdxList3)-1)
                idx1, idx2, idx3 = crossIdxList1[selectedIdx1], crossIdxList2[selectedIdx2], crossIdxList3[selectedIdx3]
                feat1, feat2, feat3 = self.allData[idx1], self.allData[idx2], self.allData[idx3]
                
                randWeight = np.random.rand() # get random weight

                self.labelList.append(list(combinElem))

                randWeightEnable = 1 # 0 for disable; 1 for enable
                
                if randWeightEnable == 0:
                    self.featList.append([feat1, feat2]) # Add
                elif randWeightEnable == 1:
                    self.featList.append([feat1 * randWeight / 2, feat2 * (1 - randWeight), feat3 * randWeight / 2]) # Split

    def __len__(self):
        return len(self.labelList)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            # feat = self.featList[index]
            # if len(self.featList[index]) == 1:
            #     feat = self.featList[index][0]
            if len(self.featList[index]) == 2:
                feat = self.featList[index][0] + self.featList[index][1] # [(2048, ), (2048, )]
            elif len(self.featList[index]) == 3:
                feat = self.featList[index][0] + self.featList[index][1] + self.featList[index][2] # [(2048, ), (2048, ), (2048, )]

            onehot = torch.zeros(14)
            onehot[self.labelList[index]] = 1.
            label = onehot
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class CompositeErrorDataset(Dataset):
    ''' This is used for double-class testing. '''
    def __init__(self, featDataName):
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        # self.allLabel = np.array([np.where(i[0][0]==1)[0] for i in self.data])  # [1, 5]
        self.allLabel = np.array([i[0].reshape(-1) for i in self.data])  # [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]

        if self.data[0][1].shape[0] == 8*2048:
            self.allData = np.array([i[1].reshape(8, 2048).mean(0) for i in self.data])
        else:
            self.allData = np.array([i[1] for i in self.data])

        print('Loading data: ', len(self.data))
        print('label shape: ', self.allLabel.shape)
        print('Feature shape: ', self.allData.shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            feat = self.allData[index]
            label = self.allLabel[index]
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

def getDataLoader(pklTrain, pklTest):
    # #### Single Dataset & Dataloader
    trainSet = SingleErrorDataset(pklTrain)
    train_loader = DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)
    # #### Composite Dataset & Dataloader 
    testSet = CompositeErrorDataset(pklTest)
    test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

    return trainSet, train_loader, testSet, test_loader
