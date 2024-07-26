import os
import pickle as pkl
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class MusicGenDataset(Dataset):
    def __init__(self, data_dir, layer_num=None, DEBUG=False):
        self.data_dir = data_dir
        self.layer_num = layer_num
        self.DEBUG = DEBUG
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

        test_file = self.file_list[0]
        with open(os.path.join(self.data_dir, test_file), 'rb') as handle:
            data = pkl.load(handle)

        self.num_frames = data['activations'][self.layer_num].shape[0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        with open(file_path, 'rb') as handle:
            data = pkl.load(handle)

        if self.DEBUG:
            print(f'Loaded {file_path} num layers: {len(data["activations"])} shape: {data["activations"][self.layer_num].shape}')

        if self.layer_num is None:
            return data['activations'], data['activations'][23], data['audio']
        return data['activations'][self.layer_num], data['activations'][23], data['audio']


class MusicGenDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, layer_num=None, batch_size=32, num_workers=4, DEBUG=False, val_split=0.2, test_split=0.0):
        super().__init__()
        self.data_dir = data_dir
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.DEBUG = DEBUG
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        # Create a full dataset
        full_dataset = MusicGenDataset(self.data_dir, self.layer_num, self.DEBUG)
        total_size = len(full_dataset)

        # Calculate split sizes
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
