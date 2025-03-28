from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import pandas as pd
import os

class ThreadInHoleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for thread-in-hole dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'left_image': tfds.features.Text(doc='Path to the Left camera RGB observation.',),
                        'right_image': tfds.features.Text(doc='Path to the Right camera RGB observation.',),
                        'abs_pos_x_t': tf.float32,
                        'abs_pos_y_t': tf.float32,
                        'abs_pos_z_t': tf.float32,
                        'abs_dx_t': tf.float32,
                        'abs_dy_t': tf.float32,
                        'abs_dz_t': tf.float32,
                    }),
                    'action': tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float32,
                        doc='Desired position, consists of [3x desired position].',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/mnt/cluster/datasets/thread_in_hole/v2/*/episode.csv'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data
            df = pd.read_csv(episode_path)
            df = df[::6]
            df[["actionx", "actiony", "actionz"]] = np.nan_to_num( np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]].shift(-1)) - np.array(df[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]] ))
            
            df.dropna(inplace=True)

            # Get the demo folder path  
            demo_folder = os.path.dirname(episode_path)

            # assemble episode
            episode_steps = [] 
            num_rows = len(df)  # Store length before overwriting df

            for i, row in df.iterrows():
                episode_steps.append({
                    'observation': {
                        'left_image': os.path.join(demo_folder,row['left_img']),  # Store path to the image
                        'right_image': os.path.join(demo_folder,row['right_img']),  # Store path to the image
                        
                        'abs_pos_x_t': row['abs_pos_x_t'],
                        'abs_pos_y_t': row['abs_pos_y_t'],
                        'abs_pos_z_t': row['abs_pos_z_t'],
                        'abs_dx_t': row['abs_dx_t'],
                        'abs_dy_t': row['abs_dy_t'],
                        'abs_dz_t': row['abs_dz_t']
                    },
                    'action': np.array([row['actionx'], row['actiony'], row['actionz']], dtype=np.float32),
                    'is_first': i == 0,
                    'is_last': i == (num_rows - 1),
                })

            # create output data sample
            sample = {
                'steps': episode_steps,
                'episode_metadata': {'file_path': episode_path}
            }

            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


