import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import lightning as L
import torch.utils.data as data

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self,x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))

    def forward(self,x):
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self,encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self,batch,batch_idx):
        x , _ = batch
        x  = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if batch_idx%5000 == 0:
            print(f"batch_idx: {batch_idx} , Training loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        x , _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        # print(f"batch_idx: {batch_idx} , val_loss: {val_loss}")
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        # print(f"batch_idx: {batch_idx} , test_loss: {test_loss}")
        self.log("test_loss", test_loss)
        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download = True, transform = transform, train = True)
test_set = datasets.MNIST(root="MNIST", download = True, transform = transform , train = False)


train_set_size = int(len(train_set) * 0.8)
val_set_size = int(len(train_set)) - train_set_size

train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size])

train_loader = DataLoader(train_set)
val_loader = DataLoader(val_set)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = L.Trainer(devices=8, accelerator="gpu", max_epochs=1)
trainer.fit(autoencoder, train_loader, val_loader)

encoder = Encoder()
decoder = Decoder()
model = LitAutoEncoder.load_from_checkpoint(
    'lightning_logs/version_8/checkpoints/epoch=0-step=6000.ckpt',
    encoder=encoder,  # Pass required args
    decoder=decoder
)

model.eval()

a = torch.randn(1, 28*28).to("cuda")
y_hat = model(a)
print(y_hat.shape)