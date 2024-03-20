from utils import ComicCoverDataset
from torch.utils.data import DataLoader
from simclr.models import SimCLR
from simclr.loss_fn import nt_xent
from torchvision.transforms import ToTensor
import pandas as pd
from torch.optim import Adam
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


NUM_EPOCHS = 200
GPU_IDX = 0

torch.cuda.set_device(GPU_IDX)

# Define transformations
transform = ToTensor()

# Create the dataset
mapping_table = pd.read_csv("annotations.csv")
trainset = ComicCoverDataset(mapping_table, transform=transform)
trainloader = DataLoader(
    trainset, batch_size=512, shuffle=True,
    num_workers=2, pin_memory=True, prefetch_factor=2)

# create the model
model = SimCLR().cuda()

loss_fn = nt_xent
optimizer = Adam(
    model.parameters(), lr=1e-3, weight_decay=5e-4)
lr_scheduler = CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

for epoch in range(NUM_EPOCHS):
    model.train()
    acc_loss = 0
    for idx, xs in enumerate(trainloader):
        print(f'Processing batch {idx + 1}/{len(trainloader)}')
        optimizer.zero_grad()
        xs = xs.cuda()
        zs = model(xs)
        loss = loss_fn(zs)
        acc_loss += loss.item()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    acc_loss /= len(trainloader)

    # save the model
    torch.save(model.state_dict(), 'ft_simclr_model.pt')

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, loss={acc_loss:.4f}', flush=True)
