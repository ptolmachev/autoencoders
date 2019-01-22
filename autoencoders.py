import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

# Defining Autoencoder Neural Network
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#Loading the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

#Parameters
num_epochs = 20
batch_size = 100

#Creating train_loader - generatior object
train_loader = torch.utils.data.DataLoader(
                 dataset=mnist_trainset,
                 batch_size=batch_size,
                 shuffle=True)

#Creating an instance of an AutoEncoder
AEN = AutoEncoder()

optimizer = optim.Adam(AEN.parameters(), lr = 0.005)
criterion = nn.MSELoss()

#Training loop
for i in range(num_epochs):

    print("Epoch {}/{} \n".format(i,num_epochs))

    for batch_num, (x, x_label) in enumerate(train_loader):
        if batch_num % 100 == 0:
            print("\rBatch number: {} ".format(batch_num))
        data = x.view(-1, 28 * 28)
        encoded, decoded = AEN(data)

        loss = criterion(decoded, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\nLoss: {} \n".format(loss.detach().numpy()))

    # Visualising current performance
    sample = data[0] #taking an image
    encoded_samp,decoded_samp = AEN(sample)
    img_decoded = decoded_samp.view(28,28).detach().numpy()
    img_target = sample.view(28, 28).detach().numpy()

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_target, cmap = 'gray')
    a.set_title('Before')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_decoded, cmap = 'gray')
    a.set_title('After Compression-Decompression')
    plt.show()


