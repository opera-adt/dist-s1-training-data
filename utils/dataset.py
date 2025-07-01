import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Worker function remains the same, defined at the top level for pickling.
def _check_and_get_path(subdir_path_str):
    """Worker function to check for the existence of array_data.npz."""
    subdir_path = Path(subdir_path_str)
    array_data_path = subdir_path / 'array_data.npz'
    if array_data_path.exists():
        return array_data_path
    return None


def _batch_check_paths(batch_paths):
    """Process a batch of paths and return valid ones."""
    valid_paths = []
    for path_str in batch_paths:
        path = Path(path_str)
        array_data_path = path / 'array_data.npz'
        if array_data_path.exists():
            valid_paths.append(array_data_path)
    return valid_paths


class DistS1Dataset(Dataset):
    def __init__(
        self, root_directory, transform=None, search=True, num_workers=None, batch_size=10_000, skip_hidden=True
    ):
        self.root_directory = Path(root_directory)
        self.transform = transform
        self.samples = []
        self.num_workers = num_workers if num_workers is not None else min(5, cpu_count())
        self.batch_size = batch_size
        self.skip_hidden = skip_hidden
        self.parquet_path = self.root_directory / 'dist_s1_torch_dataset.parquet'

        if search:
            self._search_and_cache_samples()
        else:
            self._load_samples_from_parquet()

        if not self.samples:
            raise RuntimeError(f"Found 0 subdirectories with 'array_data.npz' in {self.root_directory}")

    def _search_and_cache_samples(self):
        print('Step 1: Streaming directory processing (memory efficient)...')

        # Stream directories in batches to avoid memory issues
        candidate_dirs = []
        total_dirs = 0

        with os.scandir(self.root_directory) as entries:
            for entry in tqdm(entries, desc='Scanning directories'):
                if entry.is_dir() and (not self.skip_hidden or not entry.name.startswith('.')):
                    candidate_dirs.append(entry.path)
                    total_dirs += 1

                    # Process in batches to avoid memory explosion
                    if len(candidate_dirs) >= self.batch_size:
                        self._process_batch(candidate_dirs)
                        candidate_dirs = []

        # Process remaining directories
        if candidate_dirs:
            self._process_batch(candidate_dirs)

        print(f'Processed {total_dirs} total directories, found {len(self.samples)} valid samples.')

        # Cache the results to parquet
        self._cache_samples_to_parquet()

    def _process_batch(self, batch_paths):
        """Process a batch of directory paths using parallel workers."""
        print(f'Processing batch of {len(batch_paths)} directories...')

        with Pool(processes=self.num_workers) as pool:
            results = list(pool.imap_unordered(_check_and_get_path, batch_paths))

        # Filter out None results and add to samples
        batch_samples = [res for res in results if res is not None]
        self.samples.extend(batch_samples)

        print(f'Batch complete: found {len(batch_samples)} valid samples (total: {len(self.samples)})')

    def _load_samples_from_parquet(self):
        """Load sample directories from parquet file."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f'Parquet file not found: {self.parquet_path}. Use search=True to create it.')

        df = pd.read_parquet(self.parquet_path)
        if 'sample_dirs' not in df.columns:
            raise ValueError(f"Parquet file {self.parquet_path} does not contain 'sample_dirs' column.")

        self.samples = [Path(sample_dir) for sample_dir in df['sample_dirs'].tolist()]
        print(f'Loaded {len(self.samples)} samples from {self.parquet_path}.')

    def _cache_samples_to_parquet(self):
        """Cache sample directories to parquet file."""
        if not self.samples:
            print('No samples found to cache.')
            return

        sample_dirs = [str(sample_path) for sample_path in self.samples]
        df = pd.DataFrame({'sample_dirs': sample_dirs})
        df.to_parquet(self.parquet_path, index=False)
        print(f'Cached {len(self.samples)} sample paths to {self.parquet_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_path = self.samples[idx]

        with np.load(npz_path) as data:
            pre_imgs = data['pre_imgs']
            post_img = data['post_img']
            acq_dts_float = data['acq_dts_float']

        sample = {
            'pre_imgs': torch.from_numpy(pre_imgs).float(),
            'post_img': torch.from_numpy(post_img).float(),
            'acq_dts_float': torch.from_numpy(acq_dts_float).float(),
        }
        return sample
