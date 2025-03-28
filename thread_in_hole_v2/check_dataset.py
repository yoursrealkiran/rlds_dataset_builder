import tensorflow_datasets as tfds

# Load the dataset
dataset, info = tfds.load('thread_in_hole_dataset', with_info=True)

# Check the first sample in the 'train' split
for sample in dataset['train'].take(2):
    print(sample)
