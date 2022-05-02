import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 32 #10
epochs = 100 #200
lr = 1e-4
d = 2
NOISE_FACTOR = 0.2

# Create a pytorch custom dataset using cloud free satellite images
source = '/home/ubuntu/Autoencoder/Final/Satellite_Images/Train'

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image.to(device,torch.float)

transform = transforms.Compose([transforms.ToTensor()])

# Apply transformation to the dataset
dataset = CustomDataSet(source, transform=transform)

# Create dataloader and set batch size
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

# Print running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Print image size (b x c x h x w)
print(next(iter(dataloader)).shape)
dataiter = iter(dataloader)
images = dataiter.next()
print(torch.min(images), torch.max(images))

# Constructing encoder
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=300, out_features=50),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(in_features=50, out_features=25),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(25),
            nn.Linear(in_features=25, out_features=16),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=8),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(in_features=8, out_features=2),
            nn.BatchNorm1d(2),
            )
    def forward(self, x):
        x = x.view(-1, 3 * 10 * 10)
        x = self.encoder(x)
        return x

# Constructing decoder
class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=8),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(in_features=8, out_features=16),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=25),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(25),
            nn.Linear(in_features=25, out_features=50),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(in_features=50, out_features=300),
        )

    def forward(self, x):
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
encoder = Encoder().to(device)
decoder = Decoder().to(device)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params_to_optimize, lr=lr,weight_decay=1e-6)

# Set noise to be added to the encoder
def add_noise(inputs, NOISE_FACTOR):
    noisy = inputs + torch.randn_like(inputs) * NOISE_FACTOR
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

# Training autoencoder
def training(dataloader, num_epochs):

    train_loss = []

    for epoch in range(1, num_epochs + 1):

        encoder.train()
        decoder.train()

        # monitor training loss
        runing_loss= 0.0

        # Training
        for data  in  dataloader:
            img = data
            img = img.to(device)

            image_noisy = add_noise(img, NOISE_FACTOR)
            image_noisy = image_noisy.to(device)
            # ===================forward=====================
            encoded_data = encoder(image_noisy)
            decoded_data = decoder(encoded_data)
            loss = criterion(decoded_data, img.view(-1, 3 * 10 * 10))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item() * img.size(0)
            # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss))

        loss = runing_loss / len(dataloader)
        train_loss.append(loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss))

        torch.save(encoder.state_dict(), '/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_encoder_autoencoder.pth')
        torch.save(decoder.state_dict(), '/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_decoder_autoencoder.pth')

    return train_loss

train_loss = training(dataloader, epochs)

# Plot the Training Loss
plt.figure()
plt.style.use('ggplot')
plt.plot(train_loss)
plt.title('MLP Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_train_ae__loss.png')

# Plot the Training Loss in log
plt.figure()
plt.style.use('ggplot')
plt.plot(train_loss)
plt.title('MLP Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_train_ae__loss_log.png')

# Print Sample images of reconstruction
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Batch of test images
dataiter = iter(dataloader)
images = dataiter.next()

# Sample outputs
encoded_data = encoder(images)
decoded_data = decoder(encoded_data)
images = images.cpu().numpy()
output = decoded_data.view(batch_size, 3, 10, 10)
output = output.cpu().detach().numpy()

#Original Images

print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])

plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_Original.png')

#Reconstructed Images

print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(output[idx])

plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/MLP/mlp_Reconstructed.png')