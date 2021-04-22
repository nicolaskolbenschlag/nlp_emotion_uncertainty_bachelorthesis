# *_*coding:utf-8 *_*
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

class MyDataset(Dataset):
    def __init__(self, data, partition, subjectivities_per_sample):
        super(MyDataset, self).__init__()
        self.partition = partition
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        # pad features and labels
        feature_lens = []
        for feature in features:
            feature_lens.append(len(feature))
        self.feature_lens = torch.tensor(feature_lens)
        if partition == 'train':
            self.features = pad_sequence([torch.tensor(feature, dtype=torch.float) for feature in features], batch_first=True)  # Note: default batch_first = False
            self.labels = pad_sequence([torch.tensor(label, dtype=torch.float) for label in labels], batch_first=True)
            self.metas = pad_sequence([torch.tensor(meta) for meta in metas], batch_first=True)  # will not be used
        
        else:
            self.features = [torch.tensor(feature, dtype=torch.float) for feature in features]
            self.labels = [torch.tensor(label, dtype=torch.float) for label in labels]
            self.metas = [torch.tensor(meta) for meta in metas]
        
        # NOTE incorporate subjecities among annotations
        subjectivities = []
        for meta in self.metas:
            sample_id = int(meta[0][0])

            # NOTE for train, samples get split up, so we had to specify the id to distinguish between sub-samples from a video
            if partition == "train":
                first_timestamp = int(meta[0][1])
                sample_id = f"{sample_id}_{first_timestamp}"
            else:
                sample_id = str(sample_id)

            # subjectivity = torch.tensor(subjectivities_per_sample[sample_id], dtype=torch.float)
            subjectivity = subjectivities_per_sample[sample_id]

            subjectivities += [subjectivity]
        
        if partition == "train":
            subjectivities = pad_sequence(subjectivities, batch_first=True)
        self.subjectivities = subjectivities


    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]
        # return feature, feature_len, label, meta
        
        subjectivity = self.subjectivities[idx]
        return feature, feature_len, label, meta, subjectivity