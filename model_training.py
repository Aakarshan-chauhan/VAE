import torch as t
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets

from torchvision import datasets
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, latent_space):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_space),
            
        )
        
        self.means=  nn.Linear(latent_space, latent_space)
        self.variance = nn.Linear(latent_space, latent_space)
        
    def forward(self, inputs):
        x = self.model.forward(inputs)
        mu = self.means(x)
        sigma = nn.Sigmoid()(self.variance(x))
        
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, latent_space):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_space, 128),
            nn.ReLU(),
            nn.Linear(128, 7744),
            nn.ReLU(),
        )
        
        self.transconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.transconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, output_padding=1)
        
        
    def forward(self,inputs):
        x = self.model.forward(inputs)
        x = x.view(-1, 64, 11, 11)
        x = nn.ReLU()(self.transconv1(x))
        x = self.transconv2(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_space=28):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_space)
        self.decoder = Decoder(latent_space)
        
        self.latent_space = latent_space
        
        musample = t.zeros([latent_space,], dtype=t.float32)
        sigmasample = t.ones([latent_space,],dtype=t.float32)
        self.normalDist = t.distributions.Normal(musample, sigmasample)
        self.kl = 0
        
    def encode(self, inputs):
        mu, sigma = self.encoder(inputs)
        z = mu + sigma * self.normalDist.rsample().to(device)
        
        self.kl = (sigma**2 + mu**2 - t.log(sigma) - 1/2).sum(dim=1).mean(dim=0)
        
        return z
    
    def forward(self, inputs):
        z = self.encode(inputs)
        return self.decoder(z)
    
    def save_model(self):
        t.save(self.encoder, "Encoder_weights")
        t.save(self.decoder, "Decoder_weights")
    
    def load_model(self):
        self.encoder = t.load("Encoder_weights")
        self.encoder.eval()
        
        self.decoder = t.load("Decoder_weights")
        self.decoder.eval()

def train(model, data, epochs=20):
    opt = t.optim.Adam(model.parameters())
    mse = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, data.shape[0], 128):

            x = data[i:i+31].to(device)
            opt.zero_grad()
            x_hat = model(x)
            loss = mse(x, x_hat) + model.kl 
            loss.backward()
            opt.step()
            epoch_loss += loss

        print(epoch_loss)

def show(img):
    img = img.permute(1,2,0)
    plt.imshow(img)

def predict(model, img):
    preds = model(img.to(device))[0].cpu().detach()
    show(img)

if __name__ == "__main__":
    datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=None)
    try:
        device = t.device('cuda')
    except:
        device= t.device('cpu')

    i = 0
    ximgs = np.empty((60000, 28, 28, 1), dtype=np.float32)
    y = np.empty((60000, 1), dtype=np.float32)

    for img, label in mnist_trainset:
        img = np.array(img, dtype=np.float32)[:, :, np.newaxis]
        ximgs[i] = img
        y[i] = [label]
        i+= 1
    ximgs = t.from_numpy(ximgs)
    ximgs = ximgs.permute(0, 3, 1, 2)

    v = VAE(10).to(device)
    train(v, ximgs[:60000], 10)


    