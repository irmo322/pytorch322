from torch.utils.data import Dataset


class ParallelDataset(Dataset):

    def __init__(self, datasets):
        if len(datasets) == 0:
            raise ValueError("ParallelDataset needs at least one dataset.")
        self.datasets = datasets
        lengths = [len(dataset) for dataset in self.datasets]
        if len(set(lengths)) > 1:
            raise ValueError("ParallelDataset needs datasets of identical sizes.")
        self.length = lengths[0]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return [dataset[i] for dataset in self.datasets]
