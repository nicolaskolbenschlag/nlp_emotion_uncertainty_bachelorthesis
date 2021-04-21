import pandas as pd
import numpy as np
import torch
import utils

def calculate_rolling_subjectivities(params):
    annotations_per_vid = {}

    annotators = [2,4,5,7,8]
    for annotator in annotators:
        data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc, params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache, refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=annotator)

        for partition in ["devel", "test", "train"]:
            metas = data[partition]["meta"]
            labels_ = data[partition]["label"]

            assert labels_.shape[2] == 1# NOTE currently only one emo dim supported
            for emo_dim in range(labels_.shape[2]):
                labels = labels_[:,:,emo_dim]

                # NOTE iterate over samples and store them according to their id
                for i, meta in enumerate(metas):

                    vid_id = meta[0][0]
                    timestamps = meta[1]
                    segment_ids = meta[2]

                    label_series = labels[i]

                    if vid_id not in annotations_per_vid.keys():
                        annotations_per_vid[vid_id] = []
                    
                    annotations_per_vid[vid_id] += [label_series]
    
    subjectivities = {}
    for vid_id, annotations in annotations_per_vid.items():

        subjectivity_of_sample = []

        for annotation_1 in annotations:
            for annotation_2 in annotations:
                if annotation_1 == annotation_2:
                    continue
                
                # NOTE calculate rolling measuremt of subjectivity between each available annotation
                subjectivity = pd.Series(annotation_1).rolling(10).corr(annotation_2)
                subjectivity = subjectivity.fillna(0.)
                subjectivity_of_sample += [subjectivity]
        
        # NOTE calculate element-wise mean to get average subjectivity at each timestep
        subjectivity_of_sample = np.mean(np.stack(subjectivity_of_sample), axis=0)
        # NOTE convert to tensor and store to return
        subjectivity_of_sample = torch.Tensor(subjectivity_of_sample).float()
        subjectivities[vid_id] = subjectivity_of_sample
    
    return subjectivities