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
        '1.0.0': 'Converted dataset from CSV files with downsampling and computed actions.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'left_img': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='RGB observation from left_img.',
                        ),
                        'right_img': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='RGB observation from right_img.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state: [x, y, z, xquat, yquat, zquat, wquat, gripper_closed].', 
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Action: [dx, dy, dz, droll, dpitch, dyaw, gripper_closed].',
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32, doc='Discount factor'),
                    'reward': tfds.features.Scalar(dtype=np.float32, doc='Reward'),
                    'is_first': tfds.features.Scalar(dtype=np.bool_, doc='Is first step'),
                    'is_last': tfds.features.Scalar(dtype=np.bool_, doc='Is last step'),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_, doc='Is terminal'),
                    'language_instruction': tfds.features.Text(),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='USE embedding'
                    ),
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
        def load_left_image(image_path):
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize((256, 256))
                return np.array(img)

        def load_right_image(image_path):
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize((128, 128))
                return np.array(img)

        def _parse_example(episode_path):
            df = pd.read_csv(episode_path)
            df = df[::6].reset_index(drop=True)
            df[["actionx", "actiony", "actionz"]] = np.nan_to_num(
                np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]].shift(-1)) -
                np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]])
            )
            episode = []
            num_steps = len(df)
            base_dir = os.path.dirname(episode_path)
            language_instruction = "Insert the thread into the hole in blue hole base"
            language_embedding = self._embed([language_instruction]).numpy()[0]

            for i, row in df.iterrows():
                left_img_file = os.path.join(base_dir, row['left_img'])
                right_img_file = os.path.join(base_dir, row['right_img'])
                left_img_array = load_left_image(left_img_file)
                right_img_array = load_right_image(right_img_file)

                # State with gripper closed
                state = np.array([
                    float(row['abs_pos_x_t']),
                    float(row['abs_pos_y_t']),
                    float(row['abs_pos_z_t']),
                    float(row['xquat']),
                    float(row['yquat']),
                    float(row['zquat']),
                    float(row['wquat']),
                    1.0  # gripper closed
                ], dtype=np.float32)

                # Action with rotation_delta zero + gripper closed
                action = np.array([
                    float(row['actionx']),
                    float(row['actiony']),
                    float(row['actionz']),
                    0.0, 0.0, 0.0,  # droll, dpitch, dyaw (rotation_delta)
                    1.0             # gripper closed
                ], dtype=np.float32)

                step = {
                    'observation': {
                        'left_img': left_img_array,
                        'right_img': right_img_array,
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

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            return episode_path, sample

        for episode_path in paths:
            yield _parse_example(episode_path)
