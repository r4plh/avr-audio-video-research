import torchvision.models as models
import torch.nn as nn
import pytorch_lightning as L
import torch.nn.functional as F
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

class ImageNetTransferLearning(L.LightningModule):

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()



        for param in self.feature_extractor.parameters():
            param.requires_grad = False        
        
        self.classifier = nn.Linear(num_filters, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.feature_extractor(x)
        z = z.flatten(1)
        y_hat = self.classifier(z)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        z = self.feature_extractor(x)
        z = z.flatten(1)
        y_hat = self.classifier(z)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = ImageNetTransferLearning()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Download training set
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download test set  
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_set_size = int(len(train_dataset) * 0.8)
val_set_size = int(len(train_dataset)) - train_set_size

train_dataset, val_dataset = data.random_split(train_dataset, [train_set_size, val_set_size])

train_loader = DataLoader(train_dataset, 64 , True)
val_loader = DataLoader(val_dataset, 64 , True)



# trainer = L.Trainer(devices=8, accelerator="gpu", max_epochs=10)
# trainer.fit(model, train_loader, val_loader)

import cv2

model = ImageNetTransferLearning.load_from_checkpoint('lightning_logs/version_11/checkpoints/epoch=49-step=3950.ckpt')

img = cv2.imread('ship.png')
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img_rgb.transpose(2, 0, 1)
img_rgb = torch.from_numpy(img_rgb).float()
img_rgb = img_rgb.unsqueeze(0)

img_rgb = img_rgb.to("cuda")

y_hat = model.forward(img_rgb)
print(y_hat)

probs = F.softmax(y_hat, dim=1)[0]
top3_prob, top3_idx = torch.topk(probs, 3)



classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

predicted_idx = y_hat.argmax(dim=1).item()
predicted_class = classes[predicted_idx]
confidence = F.softmax(y_hat, dim=1)[0][predicted_idx].item()

print(f"\nPredicted: {predicted_class} (confidence: {confidence:.2%})")

probs = F.softmax(y_hat, dim=1)[0]
top3_prob, top3_idx = torch.topk(probs, 3)

print("\nTop 3 predictions:")
for i in range(3):
    idx = top3_idx[i].item()
    print(f"{i+1}. {classes[idx]}: {top3_prob[i].item():.2%}")