from typing import Iterator, Tuple, Any, Optional, List, Dict
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import os
import random


class ThreadInHoleDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Converted dataset from CSV files with normalized positions and next-step actions with all episodes and steps included.',
    }

    def __init__(self, *args, episodes_config: Optional[List[Dict]] = None,
                 **kwargs):
        # Automatically discover all episode CSVs and set num_episodes accordingly
        csv_paths = sorted(glob.glob('/mnt/cluster/datasets/thread_in_hole/v4/*/episode.csv', recursive=True))
        if not csv_paths:
            raise FileNotFoundError("No episode CSV files found at specified path.")

        self.num_episodes = len(csv_paths)

        # Define three fixed instructions for three episode groups
        # self.fixed_instructions = [
        #     "Insert the white thread into the round hole of the green cylinder",
        #     "Insert the white thread into the round hole of the dark blue cylinder",
        #     "Insert the white thread into the square hole of green cube",
        # ]

        if episodes_config is not None:
            if len(episodes_config) != self.num_episodes:
                raise ValueError(f"episodes_config must have {self.num_episodes} items, got {len(episodes_config)}")
            self.episodes_config = episodes_config
        else:
            self.episodes_config = []

            for i, ep_path in enumerate(csv_paths):
                base_dir = os.path.dirname(ep_path)
                language_instruction = self._select_instruction(i)

                df = pd.read_csv(ep_path)
                initial_pos = df.loc[0, ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']].values.astype(np.float32)
                pos_cols = ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']
                # State values from first step padded to 8 dims
                state_values_first_step = (df[pos_cols].astype(np.float32).values - initial_pos)[0].tolist() + [0, 0, 0, 0, 0]

                # Action delta from first step or zeros if no next step
                if len(df) > 1:
                    next_pos = df[pos_cols].shift(-1).fillna(0.0).values
                    curr_pos = df[pos_cols].values
                    action_delta_first_step = (next_pos - curr_pos)[0].tolist() + [0, 0, 0, 0]
                else:
                    action_delta_first_step = [0.0] * 7

                episode_cfg = {
                    'csv_path': ep_path,
                    'left_img_base_dir': base_dir,
                    'language_instruction': language_instruction,
                    'state_values_first_step': state_values_first_step,
                    'action_delta_first_step': action_delta_first_step,
                    'reward': 1.0,
                }
                self.episodes_config.append(episode_cfg)

        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    @staticmethod
    def _select_instruction(index: int) -> str:
        if 0 <= index <= 19:
            return "Insert the white thread into the round hole of the green cylinder"
        elif 20 <= index <= 39:
            return "Insert the white thread into the round hole of the dark blue cylinder"
        elif 40 <= index <= 59:
            return "Insert the white thread into the square hole of green cube"
        else:
            return "Insert the thread into the hole of the block"

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'left_img': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format='png'),
                        'right_img': tfds.features.Image(shape=(128, 128, 3), dtype=np.uint8, encoding_format='png'),
                        'state': tfds.features.Tensor(shape=(8,), dtype=np.float32),
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
        # Use 90% of episodes for training, 10% for validation
        train_end = int(self.num_episodes * 0.9)
        return {
            'train': self._generate_examples(start_epi=0, end_epi=train_end),
            'val': self._generate_examples(start_epi=train_end, end_epi=self.num_episodes),
        }

    def _generate_examples(self, start_epi: int, end_epi: int) -> Iterator[Tuple[str, Any]]:
        def load_padded_image_tf(image_path, target_size=(256, 256)):
            """Load and pad image using TensorFlow with high-quality resizing to target_size."""
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, dtype=tf.uint8)

            shape = tf.shape(image)
            height, width = shape[0], shape[1]

            target_height, target_width = target_size
            scale_height = target_height / tf.cast(height, tf.float32)
            scale_width = target_width / tf.cast(width, tf.float32)
            scale = tf.minimum(scale_height, scale_width)

            new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_height, new_width], method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)

            pad_height = target_height - new_height
            pad_width = target_width - new_width
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)
            return image.numpy()


        for epi in range(start_epi, end_epi):
            cfg = self.episodes_config[epi]
            df = pd.read_csv(cfg["csv_path"])
            initial_pos = df.loc[0, ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']].values.astype(np.float32)
            pos_cols = ['relative_tip_position_x', 'relative_tip_position_y', 'relative_tip_position_z']
            df[pos_cols] = df[pos_cols].astype(np.float32).values - initial_pos

            language_instruction = cfg["language_instruction"]
            language_embedding = self._embed([language_instruction]).numpy()[0]

            steps = []
            base_dir = cfg["left_img_base_dir"]

            for i, row in df.iterrows():
                left_img_path = os.path.join(base_dir, row['frameLeftRectifiedPath'])
                right_img_path = os.path.join(base_dir, row['frameRightRectifiedPath'])
                left_img = load_padded_image_tf(left_img_path, target_size=(256, 256))
                right_img = load_padded_image_tf(right_img_path, target_size=(128, 128))


                state = np.array([
                    row['relative_tip_position_x'],
                    row['relative_tip_position_y'],
                    row['relative_tip_position_z'],
                    0, 0, 0, 0, 0,
                ], dtype=np.float32)

                if i < len(df) - 1:
                    action_delta = np.array([
                        df.loc[i + 1, 'relative_tip_position_x'] - row['relative_tip_position_x'],
                        df.loc[i + 1, 'relative_tip_position_y'] - row['relative_tip_position_y'],
                        df.loc[i + 1, 'relative_tip_position_z'] - row['relative_tip_position_z'],
                        0, 0, 0, 0,
                    ], dtype=np.float32)
                else:
                    action_delta = np.zeros(7, dtype=np.float32)

                step = {
                    'observation': {
                        'left_img': left_img,
                        'right_img': right_img,
                        'state': state,
                    },
                    'action': action_delta,
                    'discount': 1.0,
                    'reward': float(i == len(df) - 1),
                    'is_first': (i == 0),
                    'is_last': (i == len(df) - 1),
                    'is_terminal': (i == len(df) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step)

            split_name = "val" if epi >= int(self.num_episodes * 0.9) else "train"
            yield f"{split_name}_episode_{epi:03d}", {
                'steps': steps,
                'episode_metadata': {
                    'file_path': cfg["csv_path"],
                }
            }
