import pandas as pd
import numpy as np
import torch
import utils

def calculate_rolling_subjectivities(params):
    annotations_per_vid = {}

    for annotator in range(1,18):#[2,4,5,7,8], range(1,16)
        try:
            data = utils.load_data(params, params.feature_set, params.emo_dim_set, params.normalize, params.label_preproc, params.norm_opts, params.segment_type, params.win_len, params.hop_len, save=params.cache, refresh=params.refresh, add_seg_id=params.add_seg_id, annotator=annotator)
        except Exception as e:
            print(f"Exception for annotator {annotator}: {e}")
            continue

        for partition in ["train", "devel", "test"]:
            metas = data[partition]["meta"]
            labels_ = data[partition]["label"]

            assert labels_[0].shape[1] == 1# NOTE currently only one emo dim supported
            for emo_dim in range(labels_[0].shape[1]):
                labels = [l[:,emo_dim] for l in labels_]

                # NOTE iterate over samples and store them by their id
                for i, meta in enumerate(metas):

                    # NOTE meta list of shape (seq_len, 3); with 2nd dim: [vid_id,timestamp,segment_id]
                    vid_id = int(meta[0][0])

                    label_series = labels[i]

                    # NOTE for training, samples get split up, so if we would not specify id, we would get multiple sub-series per sample and per annotator
                    if partition == "train":
                        first_timestamp = int(meta[0][1])
                        vid_id = f"{vid_id}_{first_timestamp}"
                    else:
                        vid_id = str(vid_id)
                    
                    if vid_id not in annotations_per_vid.keys():
                        annotations_per_vid[vid_id] = []
                    
                    annotations_per_vid[vid_id] += [label_series]
    
    subjectivities = {}
    for vid_id, annotations in annotations_per_vid.items():

        subjectivity_of_sample = []

        for i, annotation_1 in enumerate(annotations):
            for j, annotation_2 in enumerate(annotations[i:]):
                if i == j:
                    continue
                
                # NOTE calculate rolling measuremt of subjectivity between each available annotation
                rolling_window = 10
                subjectivity = [0.] * (rolling_window - 1)
                subjectivity += [
                    pd.Series(annotation_1[i - rolling_window : i]).corr(pd.Series(annotation_2[i - rolling_window : i]))
                        for i in range(rolling_window, len(annotation_1) + 1)
                    ]
                subjectivity = pd.Series(subjectivity).fillna(.0)# NOTE [0,0,0].corr([0,0,0]) = nan
                subjectivity_of_sample += [subjectivity]
        
        ################################
        assert len(subjectivity_of_sample) >= 1, f"too less annotations for sample {vid_id}"
        # err = False
        # for a in subjectivity_of_sample:
        #     if len(a) != len(subjectivity_of_sample[0]):
        #         err = True
        #         break
        # if err:
        #     print(f"vid{vid_id}: {len(annotations)} - 1st: {len(annotations[0])}")
        #     # print(f"vid{vid_id}: {subjectivity_of_sample[0]}")
        #     tmp = [len(s) for s in subjectivity_of_sample]
        #     print(f"vid{vid_id}: {tmp}")
        # ################################
        
        # NOTE calculate element-wise mean to get average subjectivity at each timestep
        subjectivity_of_sample = np.stack(subjectivity_of_sample).mean(axis=0)
        # NOTE convert to tensor and store to return
        subjectivity_of_sample = torch.Tensor(subjectivity_of_sample).float()
        subjectivities[vid_id] = subjectivity_of_sample
    
    return subjectivities