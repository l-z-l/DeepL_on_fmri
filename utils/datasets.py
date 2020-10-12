import torch
from torch.utils.data import Dataset
import numpy as np
from utils.data import signal_to_connectivities


# read the roi data, and then convert to connectivity matrix
class ConnectivityDataset(Dataset):
    def __init__(self, dataDir='../data', roi="MSDL", num_subject=273):
        dataset = str(num_subject) + "_" + roi

        labels = np.load(dataDir + "/" + dataset + "_label.npy", allow_pickle=True)
        subjects = np.load(dataDir + "/" + dataset + ".npy", allow_pickle=True)
        labels = [x if (x == "CN") else "CD" for x in labels]
        classes, classes_idx, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
        # helper info
        self.label_name = classes
        self.label_count = classes_count
        # subjects and labels
        self._subjects = torch.as_tensor(
            signal_to_connectivities(subjects,
                                     kind='tangent',
                                     discard_diagonal=True,
                                     vectorize=True), dtype=torch.float
        )
        self._labels = torch.as_tensor(classes_idx, dtype=torch.float)
        self._shape = len(self._subjects[0])

    def __getitem__(self, item):
        return self._subjects[item], self._labels[item]

    def __len__(self):
        return len(self._labels)

    @property
    def labels(self):
        return self._labels

    @property
    def shape(self):
        return self._shape


class ConnectivityDatasets(Dataset):
    def __init__(self, dataDir='../data', roi_type=None, num_subject=273):
        if roi_type is None:
            roi_type = ['MSDL']
        if len(roi_type) == 0:
            raise ValueError("invalid roi types")
        self.datasets = [ConnectivityDataset(dataDir, roi, num_subject) for roi in roi_type]
        length = len(self.datasets[0])
        # length check
        for i in range(1, len(self.datasets)):
            assert (len(self.datasets[i]) == len(self.datasets[i - 1]))
        # label check: make sure all labels are consistent
        for i in range(0, length):
            for j in range(1, len(self.datasets)):
                assert (self.datasets[j][i][1] == self.datasets[j - 1][i][1])

    def __getitem__(self, item):
        subjects = [dataset[item][0] for dataset in self.datasets]
        label = self.datasets[0][item][1]
        return subjects, label

    def __len__(self):
        return len(self.datasets[0])

    @property
    def shape(self):
        return torch.tensor([dataset.shape for dataset in self.datasets])


class RawConnectivityDataset(Dataset):
    pass
