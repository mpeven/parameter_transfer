import torch



class ToyModel(torch.nn.Module):
    def __init__(self, feat_size):
        super(ToyModel, self).__init__()
        # Start: 1*28*28
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=7, stride=1),#, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(16),
            # torch.nn.AvgPool2d((2,2)),
            torch.nn.Conv2d(16, 32, kernel_size=7, stride=1),#, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(32),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.Conv2d(32, 64, kernel_size=7, stride=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm2d(64),
        )
        self.preds = torch.nn.Linear(16, feat_size)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            # torch.nn.Linear(784, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(512, feat_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        return self.fc(x)
