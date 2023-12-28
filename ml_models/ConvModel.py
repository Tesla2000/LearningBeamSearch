from torch import nn


class ConvModel(nn.Module):
    def __init__(self, n_tasks: int, **_):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(n_tasks, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(n_tasks, 3), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout()
        self.dense1 = nn.Linear(in_features=450, out_features=224)
        self.drop2 = nn.Dropout(.25)
        self.dense2 = nn.Linear(in_features=224, out_features=1)

    def forward(self, x):
        x = x.float()
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.drop2(x)
        x = self.dense2(x)
        return x

