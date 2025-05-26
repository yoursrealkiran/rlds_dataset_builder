from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import os
from PIL import Image



class ThreadInHoleDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Converted dataset from CSV files with normalized positions and next-step actions.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        csv_paths = glob.glob('/mnt/cluster/datasets/thread_in_hole/v2/*/episode.csv', recursive=True)
        if not csv_paths:
            csv_paths = ["/mnt/cluster/datasets/thread_in_hole/v2/0/episode.csv"]
        return {
            'train': self._generate_examples(paths=csv_paths),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        def load_image(image_path):
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize((480, 270))
                return np.array(img)

        def _parse_example(episode_path):
            df = pd.read_csv(episode_path)
            df = df[::6].reset_index(drop=True)

            # Normalize positions to initial position (Non-transformed position values are used here)
            initial_pos = df.loc[0, ['abs_pos_x', 'abs_pos_y', 'abs_pos_z']].values.astype(np.float32)
            pos_cols = ['abs_pos_x', 'abs_pos_y', 'abs_pos_z']
            #print(df[pos_cols].head())
            df[pos_cols] = df[pos_cols].astype(np.float32).values - initial_pos

            #print(initial_pos)
            #print(df[pos_cols].head())
            #exit(0)

            # Assigning the next position (relative to the initial position) directly as the action.
            df[['actionx', 'actiony', 'actionz']] = df[pos_cols].shift(-1) 
            # The .shift(-1) makes the last row have NaN, fillna(0.0) replaces NaN with 0.0.
            df[['actionx', 'actiony', 'actionz']] = df[['actionx', 'actiony', 'actionz']].fillna(0.0) 

            episode = []
            num_steps = len(df)
            base_dir = os.path.dirname(episode_path)
            language_instruction = "Insert the thread into the hole in blue hole base"
            language_embedding = self._embed([language_instruction]).numpy()[0]

            for i, row in df.iterrows():
                left_img = load_image(os.path.join(base_dir, row['left_img']))
                right_img = load_image(os.path.join(base_dir, row['right_img']))

                state = np.array([
                    row['abs_pos_x_t'],
                    row['abs_pos_y_t'],
                    row['abs_pos_z_t'],
                    0.0, 0.0, 0.0, 0.0,   # not using quat tranformation now, so it has been set to '0' 
                    1.0  # gripper closed
                ], dtype=np.float32)

                action = np.array([
                    row['actionx'],
                    row['actiony'],
                    row['actionz'],
                    0.0, 0.0, 0.0,  # no rotation delta, so it has been set to '0'
                    1.0             # gripper closed
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

        for episode_path in paths:
            yield from _parse_example(episode_path)
