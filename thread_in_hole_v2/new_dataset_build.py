from new_dataset_builder import ThreadInHoleDataset

builder = ThreadInHoleDataset(data_dir='/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_22/')
builder.download_and_prepare()

ds_train = builder.as_dataset(split='train', shuffle_files=True)
ds_val = builder.as_dataset(split='validation', shuffle_files=False)

for example in ds_train.take(1):
    print(example)
