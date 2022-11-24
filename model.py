import torch.nn as nn
import torch.nn.functional as F
import torch

class CustomNet(nn.Module):
    def __init__(self, first_conv_out=16, first_fc_out=128):
        super().__init__()

        self.first_conv_out = first_conv_out
        self.first_fc_out = first_fc_out

        # All Conv layers.
        self.conv1 = nn.Conv2d(3, self.first_conv_out, 5)
        self.conv2 = nn.Conv2d(self.first_conv_out, self.first_conv_out*2, 3)
        self.conv3 = nn.Conv2d(self.first_conv_out*2, self.first_conv_out*4, 3)
        self.conv4 = nn.Conv2d(self.first_conv_out*4, self.first_conv_out*8, 3)
        self.conv5 = nn.Conv2d(self.first_conv_out*8, self.first_conv_out*16, 3)

        # All fully connected layers.
        self.fc1 = nn.Linear(self.first_conv_out*16, self.first_fc_out)
        self.fc2 = nn.Linear(self.first_fc_out, self.first_fc_out//2)
        self.fc3 = nn.Linear(self.first_fc_out//2, 4)

        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):    
        # Passing though convolutions.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # Flatten.
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = CustomNet(32, 512)
    tensor = torch.randn(1, 3, 224, 224)
    output = model(tensor)
    print(output.shape)
