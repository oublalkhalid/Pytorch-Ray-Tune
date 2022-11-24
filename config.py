import os

# For scheduler
MAX_NUM_EPOCHS = 50
GRACE_PERIOD = 1

# Training parameters.
EPOCHS = 50
# Data root.
DATA_ROOT_DIR = os.path.abspath('../input/blood-cells/dataset2-master/dataset2-master/images')
# Number of parallel processes for data fetching.
NUM_WORKERS = 2

# For search run.
CPU = 1
GPU = 1
NUM_SAMPLES = 5
