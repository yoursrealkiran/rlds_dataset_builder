import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# === STEP 1: Add your dataset builder path to PYTHONPATH if needed ===
sys.path.append("/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_dataset_builder/thread_in_hole_v2")

# === STEP 2: Import your builder ===
from  new_dataset_builder import ThreadInHoleDataset

# === STEP 3: Instantiate the builder ===
data_dir = "/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_24/"
builder = ThreadInHoleDataset(data_dir=data_dir)

# === STEP 4: Load (no need to rebuild if already built) ===
# builder.download_and_prepare()  # Only needed if not yet built

# === STEP 5: Load the dataset ===
#ds = builder.as_dataset(split='train', shuffle_files=False) # Use it when you wanna view train dataset (target_episode_index start from 0)
ds = builder.as_dataset(split='val', shuffle_files=False) # Use it when you wanna view validation dataset (target_episode_index from 0)

# === Choose the episode index you want to inspect ===
target_episode_index = 0  # Change this to any episode number you want

# === Iterate until the desired episode is reached ===
for i, episode in enumerate(ds):
    if i == target_episode_index:
        print(f"\n====== Episode {i} ======")
        print("File Path:", episode['episode_metadata']['file_path'].numpy().decode())

        for j, step in enumerate(episode['steps']):
            obs = step['observation']
            left_img = obs['left_img'].numpy()
            right_img = obs['right_img'].numpy()
            state = obs['state'].numpy()
            action = step['action'].numpy()
            instruction = step['language_instruction'].numpy().decode()
            instruction2=step['language_embedding'].numpy().shape
            print(f"\n--- Step {j} ---")
            print("Instruction:", instruction)
            print("Instruction2:", instruction2)
            print("State:", state)
            print("Action:", action)
            print("Reward:", step['reward'].numpy())
            print("Is Terminal:", step['is_terminal'].numpy())

            # Visualize stereo images
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            axs[0].imshow(left_img)
            axs[0].set_title('Left Image')
            axs[0].axis('off')

            axs[1].imshow(right_img)
            axs[1].set_title('Right Image')
            axs[1].axis('off')
            plt.show()

            if j >= 60:  # Limit to 60 steps
                break

        break  # Exit loop after desired episode is processed
