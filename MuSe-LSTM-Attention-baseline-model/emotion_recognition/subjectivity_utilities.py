import pandas as pd
import numpy as np
import torch
import utils
import config

import uncertainty_utilities

def get_annotations_per_sample(params):
    annotations_per_vid = {}

    for annotator in range(1,18):
        try:
            data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc, params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache, refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=annotator)
        except Exception as e:
            print(f"Exception for annotator {annotator}: {e}")
            continue

        for partition in ["train", "devel", "test"]:
            metas = data[partition]["meta"]
            labels_ = data[partition]["label"]

            for emo_dim in range(labels_[0].shape[1]):
                labels = [l[:,emo_dim] for l in labels_]

                # NOTE iterate over samples and store them by their id
                for i, meta in enumerate(metas):

                    # NOTE meta list of shape (seq_len, 3); with 2nd dim: [vid_id,timestamp,segment_id]
                    vid_id = int(meta[0][0])

                    label_series = labels[i]
                    
                    # NOTE only at the validation set for MuSe 2020 labels: for any reason, the raw annotations are always exactly by 3 timesteps longer than the fusioned annotations; therefore we have to cut it down, so that later the shapes of variances and subjectivites match
                    if partition == "devel" and not config.USE_2021_FEATURES:
                        label_series = label_series[:-3]

                    # NOTE for training, samples get split up, so if we would not specify id, we would get multiple sub-series per sample and per annotator
                    if partition == "train":
                        first_timestamp = int(meta[0][1])
                        vid_id = f"{vid_id}_{first_timestamp}"
                    else:
                        vid_id = str(vid_id)
                    
                    if vid_id not in annotations_per_vid.keys():
                        annotations_per_vid[vid_id] = {}
                    if emo_dim not in annotations_per_vid[vid_id].keys():
                        annotations_per_vid[vid_id][emo_dim] = []
                    
                    annotations_per_vid[vid_id][emo_dim] += [label_series]
    
    return annotations_per_vid

def calculate_rolling_subjectivities(annotations_per_vid):
    subjectivities = {}
    for vid_id, emo_dims in annotations_per_vid.items():

        subjectivity_of_sample_all_emo_dims = []
        for emo_dim, annotations in emo_dims.items():

            subjectivity_of_sample = []
            for k, annotation_1 in enumerate(annotations):
                for j, annotation_2 in enumerate(annotations[k:]):
                    if k == j:
                        continue
                    
                    # NOTE calculate rolling measuremt of subjectivity between each available annotation
                    rolling_window = 3
                    subjectivity = [
                        pd.Series(annotation_1[i - rolling_window : i]).corr(pd.Series(annotation_2[i - rolling_window : i]))
                            for i in range(rolling_window, len(annotation_1) + 1)
                        ]
                    subjectivity = [subjectivity[0]] * (rolling_window - 1) + subjectivity
                    
                    # NOTE [0,0,0].corr([0,0,0]) = nan; therefore interpolate to fill nan
                    if np.isnan(subjectivity[0]):
                        subjectivity[0] = 0.
                    subjectivity = pd.Series(subjectivity).interpolate()

                    # NOTE maybe use rolling mean over subjectivity, that measurement becomes smoother
                    # rolling_mean_window = 3
                    # subjectivity = subjectivity.rolling(rolling_mean_window).mean()
                    # subjectivity[:rolling_mean_window-1] = subjectivity[rolling_mean_window-1]

                    subjectivity_of_sample += [subjectivity]
                            
            assert len(subjectivity_of_sample) >= 1, f"too less annotations for sample {vid_id}"
            
            # NOTE calculate element-wise mean to get average subjectivity at each timestep
            subjectivity_of_sample = np.stack(subjectivity_of_sample).mean(axis=0)
            # NOTE convert to tensor and store to return
            subjectivity_of_sample = torch.Tensor(subjectivity_of_sample).float()
            
            # subjectivities[vid_id] = subjectivity_of_sample
            subjectivity_of_sample_all_emo_dims += [subjectivity_of_sample]
    
        # NOTE concatenate all dims of emotion (i.e. valence and arousal)
        subjectivity_of_sample_all_emo_dims = torch.Tensor(np.column_stack(subjectivity_of_sample_all_emo_dims))
        subjectivities[vid_id] = subjectivity_of_sample_all_emo_dims
    
    return subjectivities

def calculate_global_subjectivities(annotations_per_vid, window: int):
    subjectivities = {}
    for vid_id, emo_dims in annotations_per_vid.items():

        subjectivity_of_sample_all_emo_dims = []
        for emo_dim, annotations in emo_dims.items():

            subjectivity_of_sample = []
            for k, annotation_1 in enumerate(annotations):
                for annotation_2 in annotations[k+1:]:
                    
                    if window is None:
                        subjectivity = uncertainty_utilities.ccc_score(annotation_1, annotation_2)
                    
                    else:
                        subjectivity = []
                        for i in range(0, len(annotation_1) + 1 - window, window):
                            subjectivity += [uncertainty_utilities.ccc_score(annotation_1[i : i + window], annotation_2[i : i + window])]
                    
                    subjectivity_of_sample += [subjectivity]

            if window is None:
                subjectivity_of_sample = np.mean(subjectivity_of_sample)
            else:
                subjectivity_of_sample = [np.mean(s) for s in subjectivity_of_sample]

            subjectivity_of_sample_all_emo_dims += [subjectivity_of_sample]
    
        subjectivity_of_sample_all_emo_dims = torch.Tensor(subjectivity_of_sample_all_emo_dims)
        subjectivities[vid_id] = subjectivity_of_sample_all_emo_dims
    
    return subjectivities

def calculate_subjectivities(params):
    annotations_per_vid = get_annotations_per_sample(params)
    short_term = calculate_rolling_subjectivities(annotations_per_vid)
    globals = calculate_global_subjectivities(annotations_per_vid, params.global_uncertainty_window)
    return short_term, globals
