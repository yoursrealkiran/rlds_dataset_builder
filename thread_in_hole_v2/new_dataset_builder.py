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
        # UPDATED: two-level pattern: color/episode_dir/episode.csv
        csv_paths = sorted(
            glob.glob('/mnt/cluster/temp/ariel/thread_in_hole/simulation_v1_mod/*/*/episode.csv',
                      recursive=True)
        )
        if not csv_paths:
            raise FileNotFoundError("No episode CSV files found at specified path.")
        self.num_episodes = len(csv_paths)

        if episodes_config is not None:
            if len(episodes_config) != self.num_episodes:
                raise ValueError(f"episodes_config must have {self.num_episodes} items, got {len(episodes_config)}")
            self.episodes_config = episodes_config
        else:
            self.episodes_config = []
            for ep_path in csv_paths:
                # Base dir is the per-episode directory (contains images and episode.csv)
                episode_dir = os.path.dirname(ep_path)
                language_instruction = self._select_instruction_from_path(ep_path)

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
                    'left_img_base_dir': episode_dir,
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
        # Unused; kept for backward compatibility
        if 0 <= index <= 29:
            return "Insert the red thread into the round hole of the white cylinder"
        elif 30 <= index <= 59:
            return "Insert the red thread into the round hole of the yellow cylinder"
        elif 60 <= index <= 89:
            return "Insert the red thread into the round hole of green cylinder"
        else:
            return "Insert the thread into the hole of the cylinder block"

    @staticmethod
    def _select_instruction_from_path(ep_path: str) -> str:
        # UPDATED: infer task from the color folder (two levels up from episode.csv)
        # .../simulation_v1_mod/<color>/<episode_dir>/episode.csv
        color_dir = os.path.basename(os.path.dirname(os.path.dirname(ep_path))).lower()
        if 'white' in color_dir:
            return "Insert the red thread into the round hole of the white cylinder"
        if 'yellow' in color_dir:
            return "Insert the red thread into the round hole of the yellow cylinder"
        if 'green' in color_dir:
            return "Insert the red thread into the round hole of green cylinder"
        return "Insert the thread into the hole of the cylinder block"

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'input_img': tfds.features.Image(shape=(512, 512, 3), dtype=np.uint8, encoding_format='png'),
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
        # UPDATED: stratify by color folder to get 10% per color in val
        # Build per-color index groups
        groups: Dict[str, List[int]] = {'white': [], 'yellow': [], 'green': [], 'other': []}
        for idx, cfg in enumerate(self.episodes_config):
            color_dir = os.path.basename(os.path.dirname(os.path.dirname(cfg['csv_path']))).lower()
            if 'white' in color_dir:
                groups['white'].append(idx)
            elif 'yellow' in color_dir:
                groups['yellow'].append(idx)
            elif 'green' in color_dir:
                groups['green'].append(idx)
            else:
                groups['other'].append(idx)

        # Deterministic selection: last k from each group
        val_indices: List[int] = []
        for color, idxs in groups.items():
            if not idxs:
                continue
            idxs_sorted = sorted(idxs)
            k = max(1, int(0.1 * len(idxs_sorted)))
            val_indices.extend(idxs_sorted[-k:])

        all_indices = list(range(self.num_episodes))
        val_set = set(val_indices)
        train_indices = [i for i in all_indices if i not in val_set]

        return {
            'train': self._generate_examples(indices=train_indices, split_name='train'),
            'val': self._generate_examples(indices=val_indices, split_name='val'),
        }

    def _generate_examples(self, indices: List[int], split_name: str) -> Iterator[Tuple[str, Any]]:
        def load_image(path, target_size=(512, 512)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, dtype=tf.uint8)
            image = tf.image.resize(image, target_size, method="lanczos3", antialias=True)
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
            return image.numpy()

        for epi in indices:
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
                img_path = os.path.join(base_dir, row['frameLeftRectifiedPath'])
                input_img = load_image(img_path)

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
                        'input_img': input_img,
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

            yield f"{split_name}_episode_{epi:03d}", {
                'steps': steps,
                'episode_metadata': {
                    'file_path': cfg["csv_path"],
                }
            }
