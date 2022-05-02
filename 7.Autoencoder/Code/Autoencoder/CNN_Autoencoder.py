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


batch_size = 32 #10
epochs = 100 #200
lr = 0.0001
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

    def __init__(self, encoded_space_dim):
        super().__init__()
        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,  stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,  stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,  stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(2 * 2 * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

# Constructing decoder
class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2 * 2 * 64),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 2, 2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=3, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=3, padding=3),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

encoder = Encoder(encoded_space_dim=d).to(device)
decoder = Decoder(encoded_space_dim=d).to(device)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-6)

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training
        for data  in  dataloader:
            img = data
            img = img.to(device)

            image_noisy = add_noise(img, NOISE_FACTOR)
            image_noisy = image_noisy.to(device)
            # ===================forward=====================
            encoded_data = encoder(image_noisy)
            decoded_data = decoder(encoded_data)
            loss = criterion(decoded_data, img)
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
        #
        torch.save(encoder.state_dict(), '/home/ubuntu/Autoencoder/Final/Path/CNN/cnn_encoder_autoencoder.pth')
        torch.save(decoder.state_dict(), '/home/ubuntu/Autoencoder/Final/Path/CNN/cnn_decoder_autoencoder.pth')

    return train_loss

train_loss = training(dataloader, epochs)

# Plot the Training Loss
plt.figure()
plt.style.use('ggplot')
plt.plot(train_loss)
plt.title('CNN Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/CNN/cnn_train_ae__loss.png')

# Plot the Training Loss in log
plt.figure()
plt.style.use('ggplot')
plt.plot(train_loss)
plt.title('CNN Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/CNN/cnn_train_ae__loss_log.png')

# Print Sample images of reconstruction
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Batch of test images
dataiter = iter(dataloader)
images = dataiter.next()

# Sample outputs of test set
encoded_data = encoder(images)
decoded_data = decoder(encoded_data)
images = images.cpu().numpy()
output = decoded_data.view(batch_size, 3, 10, 10)
output = decoded_data.cpu().detach().numpy()

# Original Images

print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])

plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/CNN/Cnn_Original.png')

# Reconstructed Images

print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(output[idx])

plt.show()
plt.savefig('/home/ubuntu/Autoencoder/Final/Path/CNN/Cnn_Reconstructed.png')


