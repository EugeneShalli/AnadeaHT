import torch


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 1
NUM_EPOCHS = 5
TEST = True

DATA_PATH = '/workspace/livecell_base_preprocessing_rle.csv'

WIDTH = 704
HEIGHT = 520

resize_factor = False # 0.5
transforms = False

# Normalize to resnet mean and std if True.
NORMALIZE = True
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

# No changes tried with the optimizer yet.
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Changes the confidence required for a pixel to be kept for a mask. 
# Only used 0.5 till now.
# MASK_THRESHOLD = 0.5
# MIN_SCORE = 0.5
# cell type specific thresholds (set according to original paper)
cell_type_dict = {'A172': 1, 'BT474': 2, 'BV2': 3, 'Huh7': 4, 'MCF7': 5, 'SHSY5Y': 6, 'SKOV3': 7, 'SkBr3': 8}
mask_threshold_dict = {1: 0.6, 2: 0.6, 3:  0.65, 4: 0.65, 5: 0.6, 6: 0.55, 7: 0.65, 8: 0.7}
min_score_dict = {1: 0.3, 2: 0.3, 3:  0.3, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3}
# mask_threshold_dict = {1: 0.6, 2: 0.6, 3:  0.65, 4: 0.65, 5: 0.6, 6: 0.55, 7: 0.65, 8: 0.7}
# min_score_dict = {1: 0.6, 2: 0.6, 3:  0.65, 4: 0.65, 5: 0.6, 6: 0.55, 7: 0.65, 8: 0.7}

# Use a scheduler if True. 
USE_SCHEDULER = True

PCT_IMAGES_VALIDATION = 0.15

BOX_DETECTIONS_PER_IMG = 540
