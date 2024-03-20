from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import torch
import os
from torchvision import transforms
from utils import ComicCoverDataset
from simclr.models import SimCLR
from torchvision.models import resnet18


state_path = 'ft_simclr_model.pt'
# gpu_idx = 0

transform = transforms.ToTensor()

# torch.cuda.empty_cache()
# torch.cuda.set_device(gpu_idx)

# Create the dataset
mapping_table = pd.read_csv("annotations.csv")
dataset = ComicCoverDataset(mapping_table, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=512, shuffle=False,)


model = SimCLR()  # .cuda()

# if fine-tuned model exists, load it
# otherwise use the pre-trained model
if os.path.exists(state_path):
    print(f'Loading fine-tuned model from {state_path}')
    model.load_state_dict(torch.load(state_path))
else:
    print("No fine-tuned model found, using pre-trained model")

model.eval()

feature_vectors = []

with torch.no_grad():
    for idx, xs in enumerate(dataloader):
        print(f'Processing batch {idx + 1}/{len(dataloader)}')
        # xs = xs.cuda()
        features = model.feature(xs)
        feature_vectors.append(features)

feature_vectors = torch.cat(feature_vectors, dim=0).cpu()

# save the feature vectors
torch.save(feature_vectors, 'feature_vectors.pt')
