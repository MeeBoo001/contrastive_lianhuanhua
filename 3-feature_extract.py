from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import torch
from torchvision import transforms
from utils import ComicCoverDataset
from simclr.models import SimCLR

state_path = 'ft_simclr_model.pt'
gpu_idx = 1

transform = transforms.ToTensor()

torch.cuda.empty_cache()
torch.cuda.set_device(gpu_idx)

# Create the dataset
mapping_table = pd.read_csv("annotations.csv")
dataset = ComicCoverDataset(mapping_table, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=512, shuffle=False, num_workers=2)

# Load the model
model = SimCLR().cuda()
model.load_state_dict(torch.load(state_path))
model.eval()

feature_vectors = []

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        print(f'Processing batch {idx + 1}/{len(dataloader)}')
        batch = batch.cuda()
        features = model.feature(batch)
        feature_vectors.append(features)

feature_vectors = torch.cat(feature_vectors, dim=0).cpu()

# save the feature vectors
torch.save(feature_vectors, 'feature_vectors.pt')
