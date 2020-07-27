from copy import copy

import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import transforms

from memory import Memory
from model import Net

def test_accuracy(loader,net,mem):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data

            embed = net(images)
            predicted, _, _, _ = mem.query(embed, labels, True)
            predicted = predicted.squeeze(-1)
            # print(labels,predicted,predicted==labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes,trainloader,testloader

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show(classes,loader):
    # get some random images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images.detach()))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    return images,labels

def show_results(classes,loader,net):
    images,labels=show(classes,loader)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(len(predicted))))

def train(classes,loader,net,mem,testloader,epochs=2):

    optimizer = optim.Adam(net.parameters(), lr=1e-4, eps=1e-4)
    # erase memory before training episode
    mem.build()
    for epoch in range(epochs):


        running_loss = 0.0
        correct = 0
        incorrect = 0

        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            y = labels.unsqueeze(-1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            embed = net(inputs)
            labels_hat, softmax_embed, loss, update_args = mem.query(embed, y, False)
            loss.backward()
            optimizer.step()

            tmp = mem.update(*update_args)

            try:
                correct += tmp[0].size()[0]
                incorrect += tmp[1].size()[0]
            except:
                pass

            # print statistics
            running_loss += loss.item()
            cnt=200
            if i % cnt == (cnt-1):  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f hits %.3f accuracy %.1f inserts %6d updates %6d' %
                      (epoch + 1, i + 1, running_loss / cnt, correct / (correct + incorrect),
                       100 * test_accuracy(copy(testloader), net, mem),mem.inserts,mem.updates))
                running_loss = 0.0

if __name__ == "__main__":
    classes,trainloader,testloader=loaders()
    net = Net()
    memory_size = 128
    key_dim = 84
    mem = Memory(memory_size, key_dim,margin=.1,top_k=16)
    net.add_module("memory", mem)
    #net.cuda()
    train(classes,trainloader,net,mem,testloader,epochs=64)