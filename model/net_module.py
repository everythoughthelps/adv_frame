import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_extract(nn.Module):
    def __init__(self, clz_model):
        super(Feature_extract, self).__init__()
        self.conv1 = clz_model.conv1
        self.bn1 = clz_model.bn1
        self.relu = clz_model.relu
        self.maxpool = clz_model.maxpool

        self.block1 = clz_model.layer1
        self.block2 = clz_model.layer2
        self.block3 = clz_model.layer3
        self.block4 = clz_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        print(x.size())
        #x = self.block4(x)

        return x

class classifier(nn.Module):
    def __init__(self, feature_extract,class_num):
        super().__init__()
        self.fea_ext_module = feature_extract
        self.classifier = nn.Linear(1000,class_num,bias=True)
    def forward(self,x):
        x = self.fea_ext_module(x)
        x = self.classifier(x)
        return x
