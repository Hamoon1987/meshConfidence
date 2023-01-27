import torch.nn as nn
import torch
# Build The Model
# nn.Module has all functions for building Neural Networks

class Model(nn.Module):
    def __init__(self, input_features, L1, L2, L3, output_features):
        # Use super to inherit from the nn.Module
        super(Model, self).__init__()
        # Here we just define attriutes of the object
        self.fc1 = nn.Linear(input_features, L1)
        self.fc2 = nn.Linear(L1, L2)
        self.fc3 = nn.Linear(L2, L3)
        self.fc4 = nn.Linear(L3, output_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Here we define the functinality of the class
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = self.sigmoid(out)
        return out

def classifier_wj_model(pretrain=True):
    classifier_wj = Model(14, 28, 56, 28, 14)
    if pretrain:
        classifier_wj.load_state_dict(torch.load('classifier/wj/classifier_wj.pt'))
    return classifier_wj