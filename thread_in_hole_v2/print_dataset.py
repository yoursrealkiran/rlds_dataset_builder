import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# === STEP 1: Add your dataset builder path to PYTHONPATH if needed ===
sys.path.append("/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_dataset_builder/thread_in_hole_v2")

# === STEP 2: Import your builder ===
from  new_dataset_builder import ThreadInHoleDataset

# === STEP 3: Instantiate the builder ===
data_dir = "/mnt/cluster/workspaces/students/muthuraki/master_thesis/rlds_datasets/octo/version_25/"
builder = ThreadInHoleDataset(data_dir=data_dir)

# === STEP 4: Load (no need to rebuild if already built) ===
# builder.download_and_prepare()  # Only needed if not yet built

# === STEP 5: Load the dataset ===
ds = builder.as_dataset(split='train', shuffle_files=False) # Use it when you wanna view train dataset (target_episode_index start from 0)
#ds = builder.as_dataset(split='val', shuffle_files=False) # Use it when you wanna view validation dataset (target_episode_index from 0)

# === Choose the episode index you want to inspect ===
target_episode_index = 7  # Change this to any episode number you want

# === Iterate until the desired episode is reached ===
for i, episode in enumerate(ds):
    if i == target_episode_index:
        print(f"\n====== Episode {i} ======")
        print("File Path:", episode['episode_metadata']['file_path'].numpy().decode())

        for j, step in enumerate(episode['steps']):
            obs = step['observation']
            input_img = obs['input_img'].numpy()
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

            # Visualize single image
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(input_img)
            ax.set_title('Input Image')
            ax.axis('off')
            plt.show()


            if j >= 200:  # Limit to 60 steps
                break

        break  # Exit loop after desired episode is processed
