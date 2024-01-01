from torch import nn

from ml_models.abstract.DropoutModel import DropoutModel


class ConvModel(DropoutModel):
    # in_features_translator = {
    #     3: 900,
    #     4: 1125,
    #     5: 1350,
    #     6: 1575,
    #     7: 1800,
    #     8: 2025,
    #     9: 2250,
    #     10: 2475,
    # }

    def __init__(self, n_tasks: int, n_machines: int):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(n_tasks, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(n_tasks, 3), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout()
        self.dense1 = nn.Linear(in_features=(n_tasks + 1) * n_machines * 9, out_features=224)
        self.drop2 = nn.Dropout(.25)
        self.dense2 = nn.Linear(in_features=224, out_features=1)

    def predict(self, x):
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
        return self.dense2(x)
