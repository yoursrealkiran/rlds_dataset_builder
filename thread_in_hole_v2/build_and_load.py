import tensorflow_datasets as tfds
from thread_in_hole_v2_dataset_builder import ThreadInHoleDataset  # Importing the class

# Build the dataset
builder = ThreadInHoleDataset(data_dir='/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_6/')
builder.download_and_prepare()

# Load the dataset
ds = tfds.load('thread_in_hole_dataset', split='train')

# Print some info
for example in ds.take(1):
    print(example)
