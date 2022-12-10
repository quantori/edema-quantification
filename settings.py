# Supervisely to Intermediate
SUPERVISELY_DATASET_DIR = 'dataset/MIMIC-CXR-Edema-Supervisely'
INCLUDE_DIRS = []
EXCLUDE_DIRS = []
INTERMEDIATE_SAVE_DIR = 'dataset/MIMIC-CXR-Edema-Intermediate'

# Intermediate to COCO
INTERMEDIATE_DATASET_DIR = 'dataset/MIMIC-CXR-Edema-Intermediate'
COCO_SAVE_DIR = 'dataset/MIMIC-CXR-Edema-COCO'
EXCLUDE_CLASSES = []
TRAIN_SIZE = 0.8
SEED = 11
BOX_EXTENSION = {
    # figure name: (horizontal, vertical)
    'Cephalization': (0, 0),
    'Artery': (0, 0),
    'Heart': (0, 0),
    'Kerley': (0, 0),
    'Bronchus': (0, 0),
    'Effusion': (0, 0),
    'Bat': (0, 0),
    'Infiltrate': (0, 0),
}
