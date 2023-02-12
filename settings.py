from typing import List

# Supervisely to Intermediate
SUPERVISELY_DATASET_DIR = 'data/sly'
INCLUDE_DIRS: List[str] = []
EXCLUDE_DIRS: List[str] = []
INTERMEDIATE_SAVE_DIR = 'data/interim'
FIGURE_TYPE = {
    'Cephalization': 'line',
    'Artery': 'bitmap',
    'Heart': 'rectangle',
    'Kerley': 'line',
    'Bronchus': 'bitmap',
    'Effusion': 'polygon',
    'Bat': 'polygon',
    'Infiltrate': 'polygon',
    'Cuffing': 'bitmap',
}

# Intermediate to COCO
INTERMEDIATE_DATASET_DIR = 'data/interim'
COCO_SAVE_DIR = 'data/coco'
EXCLUDE_CLASSES: List[str] = []
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
    'Cuffing': (0, 0),
}
