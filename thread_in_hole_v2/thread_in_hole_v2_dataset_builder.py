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
    """DatasetBuilder for demonstration episodes stored as CSV files with custom columns."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Converted dataset from CSV files with downsampling and computed actions.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the Universal Sentence Encoder from TF Hub for language embeddings.
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
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
                        # 'state' was renamed as 'proprio' for Octo model
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state: [abs_pos_x_t, abs_pos_y_t, abs_pos_z_t, xquat, yquat, zquat, wquat].', 
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float32,
                        doc='Robot action computed as the difference between consecutive target positions: [actionx, actiony, actionz].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount factor, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward, set to 1 on the final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on the terminal step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='(Empty) Language instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Language embedding computed from the instruction (empty in this demo).'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original CSV file.'
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # Look for CSV files in subfolders of /mnt/data/
        csv_paths = glob.glob('/mnt/cluster/datasets/thread_in_hole/v2/*/episode.csv', recursive=True)
        if not csv_paths:
            # Fallback to the single uploaded file if no subfolder structure is found.
            csv_paths = ["/mnt/cluster/datasets/thread_in_hole/v2/0/episode.csv"]
        return {
            'train': self._generate_examples(paths=csv_paths),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        def load_left_image(image_path):
            """Load an image from the given path and resize it to (64, 64)."""
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize((256, 256))
                return np.array(img)
            
        def load_right_image(image_path):
            """Load an image from the given path and resize it to (64, 64)."""
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize((128, 128))
                return np.array(img)

        def _parse_example(episode_path):
            # Load raw CSV data.
            df = pd.read_csv(episode_path)
            # Downsample: take every 6th row.
            df = df[::6].reset_index(drop=True)
            # Compute the action difference from consecutive rows for target positions.
            df[["actionx", "actiony", "actionz"]] = np.nan_to_num(
                np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]].shift(-1)) -
                np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]])
            )
            episode = []
            num_steps = len(df)
            # Base directory to resolve relative paths.
            base_dir = os.path.dirname(episode_path)
            # Language instruction for the task to be performed
            language_instruction = "Insert the thread into the hole in blue hole base"
            language_embedding = self._embed([language_instruction]).numpy()[0]

            
            for i, row in df.iterrows():
                # Load images using file paths relative to the CSV's directory.
                left_img_file = os.path.join(base_dir, row['left_img'])
                right_img_file = os.path.join(base_dir, row['right_img'])
                left_img_array = load_left_image(left_img_file)
                right_img_array = load_right_image(right_img_file)

                # Parse state: absolute position and orientation.
                state = np.array([
                    float(row['abs_pos_x_t']),
                    float(row['abs_pos_y_t']),
                    float(row['abs_pos_z_t']),
                    float(row['xquat']),
                    float(row['yquat']),
                    float(row['zquat']),
                    float(row['wquat'])
                ], dtype=np.float32)

                # Use computed actionx, actiony, actionz as the action.
                action = np.array([
                    float(row['actionx']),
                    float(row['actiony']),
                    float(row['actionz'])
                ], dtype=np.float32)

                step = {
                    'observation': {
                        'left_img': left_img_array,
                        'right_img': right_img_array,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    # Reward: 1 on the final step, 0 otherwise.
                    'reward': float(i == (num_steps - 1)),
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    # No language instruction provided.
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
