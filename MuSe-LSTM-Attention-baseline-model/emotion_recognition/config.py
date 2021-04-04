# *_*coding:utf-8 *_*
import os

# path
# PATH_TO_MUSE_2020 = 'C:/Users/Lea/PycharmProjects/MuSe-annotations/data/'# change this path to yours  #'../../../databases/MuSe-data-base/data/' #

# PATH_TO_ALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/feature_segments/label_aligned')
# PATH_TO_UNALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/feature_segments/unaligned')
# PATH_TO_LABELS = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/label_segments')
# PATH_TO_LABELS_RAW = os.path.join(PATH_TO_LABELS, 'raw_annotations')
# PATH_TO_TRANSCRIPTIONS = os.path.join(PATH_TO_MUSE_2020, 'muse-wild/transcription_segments')

# PATH_TO_METEDATA = os.path.join(PATH_TO_MUSE_2020, 'raw/metadata')
# PARTITION_FILE = os.path.join(PATH_TO_METEDATA, 'partition.csv')
# META_FILE = os.path.join(PATH_TO_METEDATA, 'video_metadata.csv')
# ANNOTATOR_MAPPING = 'annotator_id_mapping.json'

# PATH_TO_RAW_AUDIO = os.path.join(PATH_TO_MUSE_2020, 'raw/audio_norm')
# PATH_TO_RAW_VIDEO = os.path.join(PATH_TO_MUSE_2020, 'raw/video')

DATA_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/data'
MODEL_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/model'
LOG_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/log'
PREDICTION_FOLDER = 'MuSe-LSTM-Attention-baseline-model/output/prediction'
# FUSION_FOLDER = 'output/fusion'



# numerical
EPSILON = 1e-6

USE_2021_FEATURES = True

if USE_2021_FEATURES:
    PARTITION_FILE = "c1_muse_wild/partition.csv"
    PATH_TO_ALIGNED_FEATURES = "../emotion_uncertainty_oberseminar/MuSe-LSTM-Attention-baseline-model/extracted_features/"
    PATH_TO_LABELS = "c1_muse_wild/label_segments/"
    PATH_TO_LABELS_RAW = "c1_muse_wild/label_segments/raw_annotations"
    PATH_TO_TRANSCRIPTIONS = "c1_muse_wild/transcription_segments/"
    ANNOTATOR_MAPPING = "c1_muse_wild/annotator_id_mapping.json"

else:
    base_dir = "../../EmCaR/8_MuSe2021/confidential/"

    PARTITION_FILE = base_dir + "metadata/partition.csv"
    PATH_TO_ALIGNED_FEATURES = base_dir + "feature_segments/"
    PATH_TO_LABELS = base_dir + "c1_muse_wild/label_segments/"
    # PATH_TO_LABELS_RAW = base_dir + "c1_muse_wild/label_segments/raw_annotations"
    PATH_TO_TRANSCRIPTIONS = base_dir + "c1_muse_wild/transcription_segments/"
    ANNOTATOR_MAPPING = base_dir + "toolbox/annotator_id_mapping.json"