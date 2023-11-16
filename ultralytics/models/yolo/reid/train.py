# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from torch.utils.data import Sampler, BatchSampler
from random import choice, choices, shuffle, sample

from ultralytics.data import build_dataloader
from ultralytics.models import yolo
from ultralytics.nn.tasks import ReIDModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class ReIDTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a ReID model.

    Example:
        ```python
        from ultralytics.models.yolo.reid import ReIDTrainer

        args = dict(model='yolov8n-reid.pt', data='coco8-reid.yaml', epochs=3)
        trainer = ReIDTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'reid'
        super().__init__(cfg, overrides, _callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        #sampler = IDSampler(dataset) if mode == 'train' else None
        #batch_sampler = None
        sampler = None
        batch_sampler = IDBatchSampler(dataset, batch_size=batch_size) if mode == 'train' else None
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        #return build_dataloader(dataset, batch_size, workers, False, rank, sampler=sampler, batch_sampler=batch_sampler)
        return build_dataloader(dataset, batch_size, workers, False, rank, sampler=sampler, batch_sampler=batch_sampler)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get ReID model with specified configuration and weights."""
        model = ReIDModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'id_loss', 'triplet_loss'
        return yolo.reid.ReIDValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, and bounding boxes."""
        images = batch['img']
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        plot_images(images,
                    batch_idx,
                    cls,
                    bboxes,
                    paths=paths,
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=False, on_plot=self.on_plot)  # save results.png

# Balanced sampling based on ID
class IDSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

        labels = self.dataset.get_labels()
        self.id_imgs = {}
        for i, obj in enumerate(labels):
            filename = obj['im_file']
            cl = obj['cls'].squeeze()
            if cl.size == 0 or cl.ndim == 0:
                continue
            for c in cl:
                if c not in self.id_imgs:
                    self.id_imgs[c] = []
                self.id_imgs[c].append(i)
        self.id_list = list(self.id_imgs.keys())

    def __iter__(self):
        while True:
            shuffle(self.id_list)
            for i in self.id_list:
                yield choice(self.id_imgs[i])

    def __len__(self):
        return 8000 # This is basically the validation interval

# Creates a batch of containing (batch_size/samples_per_id) identities
# With (at least) samples_per_id samples per identity
class IDBatchSampler(Sampler):
    def __init__(self, dataset, samples_per_id=2, batch_size=16):
        self.dataset = dataset
        self.samples_per_id = samples_per_id
        self.bs = batch_size

        #labels = self.dataset.get_labels()
        labels = self.dataset.labels
        self.id_imgs = {}
        for i, obj in enumerate(labels):
            filename = obj['im_file']
            cl = obj['cls'].squeeze()
            if cl.size == 0 or cl.ndim == 0:
                continue
            for c in cl:
                if c not in self.id_imgs:
                    self.id_imgs[c] = []
                self.id_imgs[c].append(i)
        self.id_list = list([x for x in self.id_imgs.keys() if len(self.id_imgs[x]) > 1])

    def __iter__(self):
        while True:
            ids = choices(self.id_list, k=self.bs//self.samples_per_id)
            idx = []
            for i in ids:
                idx += choices(self.id_imgs[i], k=self.samples_per_id)
            yield idx

    def __len__(self):
        return int(4000*32/self.bs) # This is basically the validation interval

# Creates a batch of containing samples_per_id images of a single identity
# The remaining images are sampled randomly
class SingleIDBatchSampler(Sampler):
    def __init__(self, dataset, samples_per_id=4, batch_size=16):
        self.dataset = dataset
        self.samples_per_id = samples_per_id
        self.bs = batch_size

        #labels = self.dataset.get_labels()
        labels = self.dataset.labels
        self.id_imgs = {}
        for i, obj in enumerate(labels):
            filename = obj['im_file']
            cl = obj['cls'].squeeze()
            if cl.size == 0 or cl.ndim == 0:
                continue
            for c in cl:
                if c not in self.id_imgs:
                    self.id_imgs[c] = []
                self.id_imgs[c].append(i)
        self.id_list = list([x for x in self.id_imgs.keys() if len(self.id_imgs[x]) > 1])
        self.n_imgs = len(labels)

    def __iter__(self):
        while True:
            pos_id = choices(self.id_list, k=1)[0]
            idx = []
            idx += choices(self.id_imgs[pos_id], k=self.samples_per_id)
            idx += sample(range(0, self.n_imgs), k=self.bs-len(idx))
            yield idx

    def __len__(self):
        return int(4000*32/self.bs) # This is basically the validation interval

