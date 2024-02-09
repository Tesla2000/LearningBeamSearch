from torch.utils.data import Dataset


class RLDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]
