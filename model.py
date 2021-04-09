#coding= utf-8
import torch.nn as nn
import torchkeras
from torchsummary import summary

class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.pool2 = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(32, 40)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        # y = self.sigmoid(x)
        return x

if __name__ == "__main__":
    net = SimpleNet().cuda()
    print(net)
    #torchkeras.summary(net, input_shape=(3, 32, 32))
    summary(net, (3, 32, 32))
    #print(net.metric_name)
