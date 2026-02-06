from argparse import ArgumentParser
import lightning as L
import torch.nn as nn

class MyModel(L.LightningModule):
    def __init__(self, layer_1_dim):
        super().__init__()
        self.layer_1_dim = layer_1_dim
        self.layer_1 = nn.Linear(layer_1_dim, 10)

parser = ArgumentParser()

args = parser.parse_args()

parser.add_argument("--devices", type=int, default=2)
parser.add_argument("--layer_1_dim", type = int, default=128)

trainer = L.Trainer(devices=args.devices)
model = MyModel(args.layer_1_dim)
