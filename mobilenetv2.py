import torch
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models

def build_model():
    model = models.mobilenet_v2(pretrained=True,progress=False)
    model.classifier[1] = nn.Linear(1280, 100)
    print(model.classifier)
    return model

def train(model):

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


if __name__ == "__main__":

    imagenet_data = torchvision.datasets.CIFAR100('/home/keras/nearline/imagenet',download=True)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
    net=build_model()