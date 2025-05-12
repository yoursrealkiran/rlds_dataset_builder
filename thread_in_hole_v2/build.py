import tensorflow_datasets as tfds
import thread_in_hole_v2_dataset_builder  # Your actual path

builder = tfds.builder("thread_in_hole_dataset", data_dir="/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_6")
builder.download_and_prepare()
