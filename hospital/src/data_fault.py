import yaml
import os
import glob
import random
import torch
import PIL.Image as Image
from abc import ABCMeta, abstractmethod
from torch.utils.data.dataset import Dataset


class DataFaultDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, config_file, train=True, transforms=None):
        # read in params from config_file
        with open(config_file, "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, yaml.Loader)

        self.transforms = transforms

    @abstractmethod
    def get_sample(self, index):
        pass

    @abstractmethod
    def apply_attack(self, sample):
        pass

    def __getitem__(self, index):
        im, label = self.get_sample(index)
        if self.transforms:
            im = self.transforms(im)
        sample = self.apply_attack((im, label))
        return sample

    @abstractmethod
    def __len__(self):
        pass


class MissingPixels(object):
    def __init__(self, probability, missing_value):
        self.probability = probability
        self.missing_value = missing_value

    def __call__(self, tensor):
        mask = torch.rand(*(tensor.shape)) <= self.probability
        tensor[mask] = self.missing_value
        return tensor

    def __repr__(self):
        return self.__class__.name__ + f'(probability={self.probability}, missing_value={self.missing_value})'


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class FilterDataset(DataFaultDataset):
    def __init__(self, config_file, root_dir, filters=[], transforms=None):
        super().__init__(config_file, transforms=transforms)
        self._load_data(root_dir)
        self.filters = filters

        for _filter in self.filters:
            if type(_filter) is not dict:
                raise TypeError(f'filters must be a list of dictionaries: {_filter}')
            if 'probability' not in _filter:
                _filter['probability'] = 1
            if 'transform' not in _filter:
                raise TypeError(f'filter must have a transform definied: {_filter}')
            
            transform = _filter['transform']
            if not hasattr(transform, '__call__') or not callable(transform.__call__):
                raise TypeError(f'transform must be callable: {transform}')

    def _load_data(self, root_dir):
        self.images = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)
            # TODO: change this to be flexible to any extension
            fnames = glob.glob(os.path.join(label_folder, "*.png"))
            self.images.extend(fnames)
            self.labels.extend([label] * len(fnames))
    
    def get_sample(self, index):
        im, lbl = self.images[index], self.labels[index]
        im = Image.open(im)
        lbl = self.labels[index]
        return (im, lbl)

    def apply_attack(self, sample):
        data, label = sample
        
        for _filter in self.filters:
            probability = _filter['probability']
            transform = _filter['transform']
            
            if random.random() < probability:
                data = transform(data)

        return (data, int(label))
    
    def __len__(self):
        return len(self.labels)


class LabelFlipDataset(DataFaultDataset):
    def __init__(self, config_file, train=True, transforms=None):
        super().__init__(config_file, train, transforms)

        if train:
            self.probability = self.cfg["attack"]["train"]["probability"]
        else:
            self.probability = self.cfg["attack"]["test"]["probability"]

    def apply_attack(self, sample):
        # This works for _binary_ classification, override for multiclass
        if random.random() < probability:
            data, label = sample
            self.faulty_samples -= 1
            return (data, 1 - label)
        return (data, label)


class MnistLabelFlipDataset(LabelFlipDataset):
    def __init__(self, config_file, root_dir, classes, train=True, transforms=None):
        self._load_data(root_dir)
        super().__init__(config_file, train, transforms)
        self.classes = classes

    def _load_data(self, root_dir):
        self.images = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)
            # TODO: change this to be flexible to any extension
            fnames = glob.glob(os.path.join(label_folder, "*.png"))
            self.images.extend(fnames)
            self.labels.extend([label] * len(fnames))

    def apply_attack(self, sample):
        # overriding for multiclass problem
        data, label = sample
        if random.random() < self.probability:
            newlabel = (int(label) + 6) % 10
            return (data, newlabel)
        return (data, int(label))

    def get_sample(self, index):
        im, lbl = self.images[index], self.labels[index]
        im = Image.open(im)
        lbl = self.labels[index]
        return (im, lbl)

    def __len__(self):
        return len(self.labels)
