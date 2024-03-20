from torch.utils.data import Dataset
from PIL import Image


class ComicCoverDataset(Dataset):
    def __init__(self, mapping_table, transform=None):
        self.mapping_table = mapping_table
        self.transform = transform

    def __len__(self):
        return len(self.mapping_table)

    def __getitem__(self, idx):
        img_path = "data/"+self.mapping_table.filename[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        return image
