from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import os
from PIL import Image, ImageOps

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
        csv_paths = glob.glob('/mnt/cluster/datasets/thread_in_hole/v5/*/episode.csv', recursive=True)
        if not csv_paths:
            csv_paths = ["/mnt/cluster/datasets/thread_in_hole/v5/0/episode.csv"]
        return {
            'train': self._generate_examples(paths=csv_paths),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:

        # Input images are resized for training model from scratch, below method is similar to the resize_image from dlimp i.e, TensorFlow Native Version, which is used in octo during training'''

        def load_image(image_path, target_size=(270, 480)):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, dtype=tf.uint8)
            image = tf.image.resize(image, target_size, method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
            return image.numpy() # Convert TensorFlow tensor into numpy array as expected by TFDS

        '''Use the below commented load_padded_image function, instead of above load_image function,
           if you want the input images to be padded to the size expected by the pretrained model, it uses PIL'''
        # def load_padded_image(image_path, target_size=(256, 256)):
        #     with Image.open(image_path) as img:
        #         img = img.convert("RGB")
        #         # Scale and pad to match the target size
        #         img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
        #         return np.array(img)
        '''Use the below commented load_padded_image_tf function, if you want to pad the input images to the size expected by the pretrained model using 
        the same method used in octo model, i.e, TensorFlow Native Version'''
        # def load_padded_image_tf(image_path, target_size=(256, 256)):
        #     """Load and pad image using TensorFlow with Octo-compatible high-quality resizing."""
        #     # Read and decode image
        #     image = tf.io.read_file(image_path)
        #     image = tf.image.decode_image(image, channels=3, dtype=tf.uint8)
        #
        #     # Get current dimensions
        #     shape = tf.shape(image)
        #     height, width = shape[0], shape[1]
        #
        #     # Calculate scaling factor to fit within target size while preserving aspect ratio
        #     target_height, target_width = target_size
        #     scale_height = target_height / tf.cast(height, tf.float32)
        #     scale_width = target_width / tf.cast(width, tf.float32)
        #     scale = tf.minimum(scale_height, scale_width)
        #
        #     # Calculate new dimensions
        #     new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
        #     new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
        #
        #     # Resize with high-quality Lanczos3 + antialiasing (matching Octo approach)
        #     image = tf.image.resize(image, [new_height, new_width],
        #                             method="lanczos3", antialias=True)
        #     image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        #
        #     # Pad to target size with black borders
        #     pad_height = target_height - new_height
        #     pad_width = target_width - new_width
        #     pad_top = pad_height // 2
        #     pad_bottom = pad_height - pad_top
        #     pad_left = pad_width // 2
        #     pad_right = pad_width - pad_left
        #
        #     image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        #                    constant_values=0)
        #
        #     return image.numpy()

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
            # Updated ranges based on new requirements
            if 0 <= index <= 49:  # 0-49 (50 episodes)
                return random.choice(green_cube_instructions)
            elif 50 <= index <= 98:  # 50-98 (49 episodes)
                return random.choice(green_cylinder_instructions)
            elif 99 <= index <= 148:  # 99-148 (50 episodes)
                return random.choice(dark_blue_cylinder_instructions)
            else:
                return "Insert the thread into the hole"

        for index, episode_path in enumerate(paths):
            df = pd.read_csv(episode_path)
            #df = df[::2].reset_index(drop=True) # Downsampling is not used/included.

            # Normalize positions to initial position (Non-transformed position values are used here)
            initial_pos = df.loc[0, ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']].values.astype(np.float32)
            pos_cols = ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']
            df[pos_cols] = df[pos_cols].astype(np.float32).values - initial_pos

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

                # uncomment the below, if using load_padded_image function
                #left_img = load_padded_image(os.path.join(base_dir, row['frameLeftRectifiedPath']), target_size=(256, 256))
                #right_img = load_padded_image(os.path.join(base_dir, row['frameRightRectifiedPath']), target_size=(128, 128))

                state = np.array([  
                    row['relative_tip_position_x'],
                    row['relative_tip_position_y'],
                    row['relative_tip_position_z'],
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
