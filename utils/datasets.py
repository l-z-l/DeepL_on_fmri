import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from utils.data import signal_to_connectivities


# factory for data
class TensorFactory:
    @staticmethod
    def create_label_data_from_path(label_path):
        labels = np.load(label_path, allow_pickle=True)
        return TensorFactory.create_label_data(labels)

    @staticmethod
    def create_label_data(labels):
        labels = [x if (x == "CN") else "CD" for x in labels]
        _, classes_idx, _ = np.unique(labels, return_inverse=True, return_counts=True)
        return torch.as_tensor(classes_idx, dtype=torch.float)


# factory for dataset
class DatasetFactory:
    @staticmethod
    def create_train_test_connectivity_datasets_from_subject(train_subject, train_label, test_subject, test_label):
        subjects = np.concatenate([train_subject, test_subject], axis=0)
        train_idx = len(train_subject)
        subjects = torch.as_tensor(
            signal_to_connectivities(subjects,
                                     kind='tangent',
                                     discard_diagonal=True,
                                     vectorize=True), dtype=torch.float
        )

        train_subject = subjects[:train_idx]
        test_subject = subjects[train_idx:]
        return TensorDataset(train_subject,
                             TensorFactory.create_label_data(train_label)), \
               TensorDataset(test_subject,
                             TensorFactory.create_label_data(test_label))

    @staticmethod
    def create_train_test_connectivity_datasets_from_path(train_path, test_path):
        train_subject = np.load(train_path + ".npy", allow_pickle=True)
        train_label = np.load(train_path + "_label.npy", allow_pickle=True)
        test_subject = np.load(test_path + ".npy", allow_pickle=True)
        test_label = np.load(test_path + "_label.npy", allow_pickle=True)
        # merge train and test to compute connecivity matrix
        return DatasetFactory.create_train_test_connectivity_datasets_from_subject(train_subject, train_label,
                                                                                   test_subject, test_label)


# read the roi data, and then convert to connectivity matrix
class ConnectivityDataset(Dataset):
    def __init__(self, dataDir='../data', roi="MSDL", num_subject=273):
        dataset = str(num_subject) + "_" + roi
        subjects = np.load(dataDir + "/" + dataset + ".npy", allow_pickle=True)
        labels = TensorFactory.create_label_data_from_path(dataDir + "/" + dataset + "_label.npy")
        # subjects and labels
        self._subjects = torch.as_tensor(
            signal_to_connectivities(subjects,
                                     kind='tangent',
                                     discard_diagonal=True,
                                     vectorize=True), dtype=torch.float
        )
        self._labels = torch.as_tensor(labels, dtype=torch.float)
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
