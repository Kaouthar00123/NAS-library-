
import abc
import logging
from nas_bib.exp_manager.data_loader import DatasetLoader
from nas_bib.utils.registre import register_class, get_registered_class
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import torch
from torch.utils.tensorboard import SummaryWriter  # Importez SummaryWriter depuis torch.utils.tensorboard


@register_class(registry="eval_methods")
class Evaluator(abc.ABC):
    def __init__(self, config={}):
        """
        Initialize the evaluate method with a configuration.
        """
        self.config = config
    #dataset needed params 
        dataset_params = self.config.get('dataset_params' , {})
        channels = dataset_params.get('channels', 3)
        hight =  dataset_params.get('hight', 32)
        width = dataset_params.get('width' , 32)

    #Extract training parameters
        self.lr_scheduler = None

        eval_method_config = self.config.get('eval_method', {})

        #loss function config
        loss_function_config = eval_method_config.get('lossFunction', 'CrossEntropyLoss')
        self.criterion = getattr(nn, loss_function_config)()


        # max evaluating time parameter
        self.max_training_time = eval_method_config.get('max_evaluating_time', 300)
        

    @abc.abstractmethod
    
    def evaluate(self, model , testloader , metric_values = None):
        """
        Evaluate the trained model on the test dataset.
        """
        pass