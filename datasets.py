from abc import abstractmethod
import os
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class BaseDataset(Dataset):
    def __init__(self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        self.data_dir = data_dir
        self.target_modality= target_modality
        self.source_modality = source_modality
        self.stage = stage
        self.image_size = image_size
        self.norm = norm
        self.padding = padding
        self.original_shape = None

    @abstractmethod
    def _load_data(self, contrast):
        pass

    def _pad_data(self, data):
        """ Pad data to image_size x image_size """
        H, W = data.shape[-2:]

        pad_top = (self.image_size - H) // 2
        pad_bottom = self.image_size - H - pad_top
        pad_left = (self.image_size - W) // 2
        pad_right = self.image_size - W - pad_left

        return np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))

    def _normalize(self, data):
        return (data - 0.5) / 0.5


class NumpyDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True,
        # --- Subset control (test-time sampling) ---
        max_items: int = 0,          # 0 = no limit; >0 keep only N items
        subset_mode: str = "first",  # "first" | "random"
        subset_seed: int = 42,
        # --- Memory/IO control ---
        lazy: bool = False,          # If True, defer loading files to __getitem__ (per-sample load)
    ):
        super().__init__(
            data_dir,
            target_modality,
            source_modality,
            stage,
            image_size,
            norm,
            padding
        )
        # record chosen indices if subsetting is applied
        self.indices = None
        self._lazy = bool(lazy) and (stage in ("train",))

        if self._lazy:
            # Build file lists only; load per-sample in __getitem__
            self.target_files = self._list_files(self.target_modality)
            self.source_files = self._list_files(self.source_modality)
            N = len(self.target_files)
            if max_items and max_items > 0 and N > max_items:
                if subset_mode == "random":
                    rng = np.random.default_rng(subset_seed)
                    idx = np.sort(rng.choice(N, size=max_items, replace=False))
                else:
                    idx = np.arange(max_items)
                self.indices = idx.astype(int)
                self.target_files = [self.target_files[i] for i in self.indices]
                self.source_files = [self.source_files[i] for i in self.indices]

            # Determine original shape from the first file
            if len(self.target_files) == 0:
                raise RuntimeError(f"No files found for {self.target_modality}/{self.stage} in {self.data_dir}")
            sample = np.load(self.target_files[0])
            self.original_shape = sample.shape[-2:]
            # subject_ids are not used in training; skip loading to save RAM
            self.subject_ids = None
            # Do not pre-pad/normalize here; done in __getitem__
        else:
            # Eager load arrays (val/test)
            self.target = self._load_data(self.target_modality)
            self.source = self._load_data(self.source_modality)

            # Optional: apply subset on loaded arrays before padding/normalize
            N = self.target.shape[0]
            if max_items and max_items > 0 and N > max_items:
                if subset_mode == "random":
                    rng = np.random.default_rng(subset_seed)
                    idx = np.sort(rng.choice(N, size=max_items, replace=False))
                else:
                    idx = np.arange(max_items)
                self.indices = idx.astype(int)
                self.target = self.target[self.indices]
                self.source = self.source[self.indices]

            # Get original shape
            self.original_shape = self.target.shape[-2:]

            # Load subject ids
            self.subject_ids = self._load_subject_ids('subject_ids.yaml')
            # If subset applied and subject_ids is aligned per-slice, subset it as well
            if self.subject_ids is not None and self.indices is not None and len(self.subject_ids) >= len(self.target):
                try:
                    self.subject_ids = self.subject_ids[self.indices]
                except Exception:
                    pass

            # Padding
            if self.padding:
                self.target = self._pad_data(self.target)
                self.source = self._pad_data(self.source)

            # Normalize
            if self.norm:
                self.target = self._normalize(self.target)
                self.source = self._normalize(self.source)

            # Expand channel dim
            self.target = np.expand_dims(self.target, axis=1)
            self.source = np.expand_dims(self.source, axis=1)

    def _load_data(self, contrast):
        data_dir = os.path.join(self.data_dir, contrast, self.stage)
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Sort by slice index
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data = []
        for file in files:
            data.append(np.load(os.path.join(data_dir, file)))

        arr = np.array(data)
        if contrast.lower() == 'mask':
            # keep uint8 for mask; and if subset applied, slice accordingly
            arr = arr.astype(np.uint8)
            if getattr(self, 'indices', None) is not None:
                arr = arr[self.indices]
        else:
            arr = arr.astype(np.float32)
        return arr

    def _list_files(self, contrast):
        data_dir = os.path.join(self.data_dir, contrast, self.stage)
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return [os.path.join(data_dir, f) for f in files]

    def _load_subject_ids(self, filename):
        subject_ids_path = os.path.join(self.data_dir, filename)
        if os.path.exists(subject_ids_path):
            with open(subject_ids_path, 'r') as f:
                subject_ids = np.array(yaml.load(f, Loader=yaml.FullLoader))
        else:
            subject_ids = None

        return subject_ids

    def __len__(self):
        if getattr(self, "_lazy", False):
            return len(self.source_files)
        return len(self.source)

    def __getitem__(self, i):
        if getattr(self, "_lazy", False):
            # Per-sample load and transform
            tgt = np.load(self.target_files[i]).astype(np.float32)
            src = np.load(self.source_files[i]).astype(np.float32)
            if self.padding:
                tgt = self._pad_data(tgt[None, ...])[0]
                src = self._pad_data(src[None, ...])[0]
            if self.norm:
                tgt = self._normalize(tgt)
                src = self._normalize(src)
            # Add channel dim
            tgt = np.expand_dims(tgt, axis=0)
            src = np.expand_dims(src, axis=0)
            return tgt, src, i
        else:
            return self.target[i], self.source[i], i


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset_dir,
        source_modality,
        target_modality,
        dataset_class,
        image_size,
        padding,
        norm,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=1,
        # --- test subset controls ---
        test_max_samples: int = 0,
        test_subset_mode: str = "first",
        test_subset_seed: int = 42,
        # --- val subset controls ---
        val_max_samples: int = 0,
        val_subset_mode: str = "first",
        val_subset_seed: int = 42,
        # --- train memory control ---
        train_lazy: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.image_size = image_size
        self.padding = padding
        self.norm = norm
        self.num_workers = num_workers

        self.dataset_class = globals()[dataset_class]
        # store test subset params
        self.test_max_samples = test_max_samples
        self.test_subset_mode = test_subset_mode
        self.test_subset_seed = test_subset_seed
        # store val subset params
        self.val_max_samples = val_max_samples
        self.val_subset_mode = val_subset_mode
        self.val_subset_seed = val_subset_seed
        # train lazy
        self.train_lazy = train_lazy

    def setup(self, stage: str) -> None:
        target_modality = self.target_modality
        source_modality = self.source_modality

        if stage == "fit":
            self.train_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='train',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
                lazy=self.train_lazy,
            )

            self.val_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='val',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
                # val subset controls
                max_items=self.val_max_samples,
                subset_mode=self.val_subset_mode,
                subset_seed=self.val_subset_seed,
            )

        if stage == "validate":
            self.val_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='val',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
                # val subset controls
                max_items=self.val_max_samples,
                subset_mode=self.val_subset_mode,
                subset_seed=self.val_subset_seed,
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='test',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm,
                # subset only for test
                max_items=self.test_max_samples,
                subset_mode=self.test_subset_mode,
                subset_seed=self.test_subset_seed,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
