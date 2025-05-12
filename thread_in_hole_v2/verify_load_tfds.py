import tensorflow_datasets as tfds

# Make sure to import your dataset module to register it
import thread_in_hole_v2_dataset_builder  # Adjust path if needed

# Define the dataset path
data_dir = "/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_4"

# Try loading the dataset
try:
    builder = tfds.builder("thread_in_hole_dataset", data_dir=data_dir)
    builder.download_and_prepare()  # Safe to call again if already prepared
    print("✅ Dataset prepared successfully.")

    # Load the dataset (this should not trigger any preparation if it was done)
    ds = builder.as_dataset(split='train')  # or 'all' if you used a custom split
    print("✅ Dataset loaded successfully.")
    
    # Print a sample
    for episode in ds.take(1):
        print("Sample episode keys:", list(episode.keys()))
        print("Steps in episode:", len(episode['steps']))
        break

except Exception as e:
    print("❌ Failed to load dataset:", str(e))
