import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            # 输入: [batch, 3, 32, 32]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # -> [batch, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [batch, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [batch, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 128, 4, 4]

            nn.Flatten(),  # -> [batch, 128 * 4 * 4]
            nn.Linear(128 * 4 * 4, 256),  # -> [batch, 256]
            nn.ReLU(),
            nn.Linear(256, num_class)  # -> [batch, num_class]
        )

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break
