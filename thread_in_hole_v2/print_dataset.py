import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# === STEP 1: Add your dataset builder path to PYTHONPATH if needed ===
sys.path.append("/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_dataset_builder/thread_in_hole_v2")

# === STEP 2: Import your builder ===
from  thread_in_hole_v2_dataset_builder import ThreadInHoleDataset

# === STEP 3: Instantiate the builder ===
data_dir = "/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_7/"
builder = ThreadInHoleDataset(data_dir=data_dir)

# === STEP 4: Load (no need to rebuild if already built) ===
# builder.download_and_prepare()  # Only needed if not yet built

# === STEP 5: Load the dataset ===
ds = builder.as_dataset(split='train', shuffle_files=False)

# === STEP 6: Iterate through and visualize the dataset ===
for i, episode in enumerate(ds.take(2)):
    print(f"\n====== Episode {i + 1} ======")
    print("File Path:", episode['episode_metadata']['file_path'].numpy().decode())

    for j, step in enumerate(episode['steps']):
        obs = step['observation']
        left_img = obs['left_img'].numpy()
        right_img = obs['right_img'].numpy()
        state = obs['state'].numpy()
        action = step['action'].numpy()
        instruction = step['language_instruction'].numpy().decode()

        print(f"\n--- Step {j + 1} ---")
        print("Instruction:", instruction)
        print("State:", state)
        print("Action:", action)
        print("Reward:", step['reward'].numpy())
        print("Is Terminal:", step['is_terminal'].numpy())

        # Optional: visualize stereo images
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].imshow(left_img)
        axs[0].set_title('Left Image')
        axs[0].axis('off')

        axs[1].imshow(right_img)
        axs[1].set_title('Right Image')
        axs[1].axis('off')
        plt.show()

        if j >= 200:
            break
