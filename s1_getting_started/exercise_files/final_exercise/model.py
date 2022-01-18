from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

num_l1 = 100
num_l2 = 100
channels = 1
height = 28
width = 28

#define convolutional layer parameters
num_filters_conv1 = 16
kernel_size_conv1 = 5
stride_conv1 = 1
padding_conv1 = 2

#define second convolutional layer parameters
num_filters_conv2 = 32

def compute_conv_dim(dim_size):
    return int((dim_size - kernel_size_conv1 + 2 * padding_conv1) / stride_conv1 + 1)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        
        self.conv_1 = nn.Conv2d(in_channels=channels,
                             out_channels=num_filters_conv1,
                             kernel_size=kernel_size_conv1,
                             stride=stride_conv1,
                             padding=padding_conv1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_out_height = compute_conv_dim(height)
        self.conv1_out_width = compute_conv_dim(width)

        self.conv_2 = nn.Conv2d(in_channels=num_filters_conv1,
                             out_channels=num_filters_conv2,
                             kernel_size=kernel_size_conv1,
                             stride=stride_conv1,
                             padding=padding_conv1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(in_channels=num_filters_conv2,
                             out_channels=num_filters_conv2,
                             kernel_size=kernel_size_conv1,
                             stride=stride_conv1,
                             padding=padding_conv1)

        #calculate nr of features that go into the fully connected layer
        self.l1_in_features = num_filters_conv2 * int(self.conv1_out_height/4) * int(self.conv1_out_width/4)    
        
        self.l_1 = nn.Linear(in_features=self.l1_in_features, 
                          out_features=num_l1,
                          bias=True)
        
        self.l_2 = nn.Linear(in_features=num_l1, 
                          out_features=num_l2,
                          bias=True)
        
        self.l_out = nn.Linear(in_features=num_l2, 
                            out_features=self.num_classes,
                            bias=False)
        
        
        # adding dropout to the network
        self.dropout = nn.Dropout2d(p=0.5)


    def forward(self, x):

        #convolutional + pooling layer
        x = self.pool1(self.conv_1(x))
        x = F.relu(x)

        #second convolutional + pooling layer
        x = self.pool2(self.conv_2(x))
        x = F.relu(x)

        #third convolutional layer
        x = self.conv_3(x)
        x = F.relu(x)

        #fully connected layer
        x = x.view(-1, self.l1_in_features)
        x = F.relu(self.dropout(self.l_1(x)))

        #second fully connected layer
        x = F.relu(self.dropout(self.l_2(x)))

        return self.l_out(x)
