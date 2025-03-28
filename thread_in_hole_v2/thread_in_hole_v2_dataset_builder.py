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
                        'left_image': tfds.features.Image(
                            shape=(960, 540, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Left camera RGB observation.',
                        ),
                        'right_image': tfds.features.Image(
                            shape=(960, 540, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Right camera RGB observation.',
                        ),
                        'abs_pos_x': tf.float32,
                        'abs_pos_y': tf.float32,
                        'abs_pos_z': tf.float32,
                        'abs_pos_x_t': tf.float32,
                        'abs_pos_y_t': tf.float32,
                        'abs_pos_z_t': tf.float32,
                        'xquat': tf.float32,
                        'yquat': tf.float32,
                        'zquat': tf.float32,
                        'wquat': tf.float32
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
            data = pd.read_csv(episode_path)
            data = data[::6]
            data[["actionx", "actiony", "actionz"]] = np.nan_to_num( np.array(data[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]].shift(-1)) - np.array(data[["abs_pos_x_t", "abs_pos_y_t", "abs_pos_z_t"]] ))
            
            # Get the demo folder path
            demo_folder = os.path.dirname(episode_path)

            # assemble episode
            data = []
            for i, row in data.iterrows():
                data.append({
                    'observation': {
                        'left_image': cv2.imread(os.path.join(demo_folder, row['left_img'])),
                        'right_image': cv2.imread(os.path.join(demo_folder, row['right_img'])),
                        'abs_pos_x': row['abs_pos_x'],
                        'abs_pos_y': row['abs_pos_y'],
                        'abs_pos_z': row['abs_pos_z'],
                        'abs_pos_x_t': row['abs_pos_x_t'],
                        'abs_pos_y_t': row['abs_pos_y_t'],
                        'abs_pos_z_t': row['abs_pos_z_t'],
                        'xquat': row['xquat'],
                        'yquat': row['yquat'],
                        'zquat': row['zquat'],
                        'wquat': row['wquat'],
                    },
                    'action': np.array([row['actionx'], row['actiony'], row['actionz']], dtype=np.float32),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                })

            # create output data sample
            sample = {
                'steps': data,
                'episode_metadata': {
                    'file_path': episode_path
                }
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


if __name__  == "__main__":
    threadinholedataset = ThreadInHoleDataset()
    print("done")