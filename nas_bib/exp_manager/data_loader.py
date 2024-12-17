
"""
data params: 
    train_batch_size: (int) Batch size for training data loader. Determines the number of samples per batch during training.
    eval_batch_size: (int) Batch size for evaluation data loader. Determines the number of samples per batch during evaluation or inference.
    train_size: (int, optional) Size of the training set. If provided, overrides the calculation based on train_data_ratio.
    eval_size: (int, optional) Size of the evaluation set. If provided, overrides the calculation based on validation_data_ratio.
    train_data_ratio: (dict) Ratio range for the size of the training set relative to the entire dataset. Specified as a dictionary with 'min' and 'max' keys, representing the minimum and maximum ratios respectively.
    validation_data_ratio: (dict) Ratio range for the size of the evaluation set (validation set) relative to the entire dataset. Specified as a dictionary with 'min' and 'max' keys, representing the minimum and maximum ratios respectively.
    dataset_name: (str) Name of the dataset to load. Currently supports 'CIFAR10', 'CIFAR100', or a custom dataset.
    dataset_path: (str) Path to the dataset directory. This directory should contain the dataset files organized according to the dataset's structure.
    data_augmentation: (string) indicating data augmentation techniques should be applied during training. Default is False.
    preprocessing: (string) indicating preprocessing steps should be applied to the data. Default is False.
"""
import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split

data_dir = '././././data'

class DatasetLoader:
    def __init__(self, config):
        train_method = config.get('train_method', {}).get('config' , {})
        self.train_data_config = train_method.get('data' , {})
        self.dataset_config = config.get('dataset_params' , {})
        self.train_batch_size = self.train_data_config.get('train_batch_size', 32)
        self.eval_batch_size = self.train_data_config.get('val_batch_size', 6324)
        self.train_size = self.train_data_config.get('train_size', None)
        self.eval_size = self.train_data_config.get('val_size', None)
        self.train_data_ratio = self.train_data_config.get('train_data_ratio',0.6)
        self.validation_data_ratio = self.train_data_config.get('validation_data_ratio',  0.2)
        
        self.dataset_name = self.dataset_config.get('dataset_name', 'CIFAR10')
        self.root = self.dataset_config.get('dataset_path', data_dir)

        print("ratio",self.train_data_ratio)
        print("ratio",self.train_data_config.get('train_size', None))
        # Define data transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization added here
        ])

    def get_dataset(self , train=True):
        if self.dataset_name == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'CIFAR100':
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=train, download=True, transform=self.transform)
        else:
            if train:
                train_dir = self.root + '/train'
                # Load from path for custom dataset
                dataset = torchvision.datasets.SomeDataset(train_dir, transform=self.transform)
            else:
                eval_dir = self.root + '/val'
                # Load from path for custom dataset
                dataset = torchvision.datasets.SomeDataset(eval_dir, transform=self.transform)
                # Example of custom dataset organization
        return dataset

    def get_subset_loader(self, subset_size=100 , train = True):
        dataset = self.get_dataset(train)
        if train:
            subset_indices = torch.randperm(len(dataset))[:subset_size]
            subset_dataset = Subset(dataset, subset_indices)
            loader = DataLoader(subset_dataset, batch_size=self.train_batch_size if subset_size == self.train_size else self.eval_batch_size, shuffle=True)
        else:
            loader = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False)
        return loader

    def get_loader(self, train=True):
        dataset = self.get_dataset(train)
        dataset_length = len(dataset)
        
        if train:
            if self.train_size is not None and self.eval_size is not None:
                train_dataset = Subset(dataset, range(dataset_length - self.train_size, dataset_length))
                print(f"Training dataset length: {len(train_dataset)}")

            else:
                train_size = int(dataset_length * self.train_data_ratio)
                val_size = dataset_length - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                print(f"ratio Training dataset length: {len(train_dataset)}")

        else:
            if self.eval_size is None:
                val_size = int(dataset_length * self.validation_data_ratio)
                self.eval_size = val_size
            val_dataset = Subset(dataset, range(dataset_length - self.eval_size, dataset_length))
            print(f"ratio Validation dataset length: {len(val_dataset)}")

        loader = DataLoader(train_dataset if train else val_dataset, batch_size=self.train_batch_size if train else self.eval_batch_size, shuffle=train)
        return loader
