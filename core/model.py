import torch.nn as nn
from .clip_utils import KLLoss

class baseCEModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.loss_evaluator = nn.CrossEntropyLoss()
    
    def forward(self, x, target=None):
        x = self.fc1(x)
        cls_score = self.fc2(x)

        if self.training:
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score

class baseBCEModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.act = nn.ReLU()

        self.loss_evaluator = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target=None):
        # x = self.act(self.fc1(x))
        x = self.fc1(x)
        x = self.act(self.dropout(x))
        cls_score = self.fc2(x)

        if self.training:
            # labelOneHot = nn.functional.one_hot(target, num_classes=14).to(x.device).float()
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score

class ImagineNet_FC(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.act = nn.ReLU()

        self.loss_evaluator = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target=None):
        x = self.fc1(x)
        x = self.act(self.dropout(x))
        cls_score = self.fc2(x)

        if self.training:
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score

class CLIP_img(nn.Module): 
    def __init__(self, in_channels, out_channels, num_cls): 
        super().__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), out_channels)

        # out for class
        self.fc3 = nn.Linear(out_channels, num_cls)
        self.act = nn.ReLU()

        # CLIP Loss
        self.KLLoss = KLLoss()
        # BCE Loss
        self.BCELoss = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target=None):
        x = self.fc1(x)
        x = self.act(self.dropout(x))
        feat = self.fc2(x)
        scoreCls = self.fc3(feat)

        return feat, scoreCls

class CLIP_text(nn.Module):
    def __init__(self, model):
        super(CLIP_text, self).__init__()
        self.model = model
        # CLIP Loss
        self.KLLoss = KLLoss()

    def forward(self, text):
        return self.model.encode_text(text)
