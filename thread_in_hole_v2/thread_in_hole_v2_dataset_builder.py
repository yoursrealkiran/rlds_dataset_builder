from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import os
from PIL import Image

import random


class ThreadInHoleDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Converted dataset from CSV files with normalized positions and next-step actions.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load Universal Sentence Encoder for embedding language instructions
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'left_img': tfds.features.Image(shape=(270, 480, 3), 
                                                        dtype=np.uint8, 
                                                        encoding_format='png'),
                        'right_img': tfds.features.Image(shape=(270, 480, 3), 
                                                         dtype=np.uint8, 
                                                         encoding_format='png'),
                        'state': tfds.features.Tensor(shape=(8,), 
                                                      dtype=np.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                    'language_embedding': tfds.features.Tensor(shape=(512,), dtype=np.float32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Collect all episode.csv paths in the dataset
        csv_paths = glob.glob('/mnt/cluster/datasets/thread_in_hole/v4/*/episode.csv', recursive=True)
        if not csv_paths:
            csv_paths = ["/mnt/cluster/datasets/thread_in_hole/v4/0/episode.csv"]
        return {
            'train': self._generate_examples(paths=csv_paths),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        def load_image(image_path):
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize((480, 270))
                return np.array(img)
        
        # Load instruction pools from files
        with open("/mnt/cluster/workspaces/students/muthuraki/rlds_dataset_builder/language_instructions/green_cube.txt", "r") as f:
            green_cube_instructions = [line.strip() for line in f if line.strip()]

        with open("/mnt/cluster/workspaces/students/muthuraki/rlds_dataset_builder/language_instructions/dark_blue_cylinder.txt", "r") as f:
            dark_blue_cylinder_instructions = [line.strip() for line in f if line.strip()]

        with open("/mnt/cluster/workspaces/students/muthuraki/rlds_dataset_builder/language_instructions/green_cylinder.txt", "r") as f:
            green_cylinder_instructions = [line.strip() for line in f if line.strip()]

        # Sort paths to ensure consistent indexing for language instruction mapping
        paths = sorted(paths)

        #  Helper function to assign language instruction based on episode index
        def get_instruction(index: int) -> str:
            # index starts from 0, so adjust ranges by -1
            if 0 <= index <= 19:  # 1-20
                return random.choice(green_cylinder_instructions)
            elif 20 <= index <= 39:  # 21-40
                return random.choice(dark_blue_cylinder_instructions)
            elif 40 <= index <= 59:  # 41-60
                return random.choice(green_cube_instructions)
            else:
                return "Insert the thread into the hole"



        for index, episode_path in enumerate(paths):
            df = pd.read_csv(episode_path)
            #df = df[::2].reset_index(drop=True) # Downsampling is to be included.

            # Normalize positions to initial position (Non-transformed position values are used here)
            initial_pos = df.loc[0, ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']].values.astype(np.float32)
            pos_cols = ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']
            #print(df[pos_cols].head())
            df[pos_cols] = df[pos_cols].astype(np.float32).values - initial_pos

            #print(initial_pos)
            #print(df[pos_cols].head())
            #exit(0)

            # Compute action as delta to next position
            next_pos = df[pos_cols].shift(-1)
            curr_pos = df[pos_cols]
            action_delta = next_pos - curr_pos
            action_delta = action_delta.fillna(0.0)  # Last row becomes 0 delta
            df[['actionx', 'actiony', 'actionz']] = action_delta


            episode = []
            num_steps = len(df)
            base_dir = os.path.dirname(episode_path)

            # Assign instruction and compute sentence embedding
            language_instruction = get_instruction(index)
            language_embedding = self._embed([language_instruction]).numpy()[0]

            for i, row in df.iterrows():
                left_img = load_image(os.path.join(base_dir, row['frameLeftRectifiedPath']))
                right_img = load_image(os.path.join(base_dir, row['frameRightRectifiedPath']))

                state = np.array([ # all proprio data has been set to 0 in observation, relative_tip_positions are not used
                    0.0, 
                    0.0,
                    0.0,
                    0.0, 0.0, 0.0, 0.0,   # not using quat transformation now, so it has been set to '0' 
                    0.0                   # not using gripper 
                ], dtype=np.float32)

                action = np.array([
                    row['actionx'],         
                    row['actiony'],
                    row['actionz'],
                    0.0, 0.0, 0.0,        # no rotation delta, so it has been set to '0'
                    0.0                   # not using gripper
                ], dtype=np.float32)

                step = {
                    'observation': {
                        'left_img': left_img,
                        'right_img': right_img,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (num_steps - 1)),
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                episode.append(step)

            yield episode_path, {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
