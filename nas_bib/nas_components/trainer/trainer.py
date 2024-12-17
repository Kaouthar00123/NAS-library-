
import abc
from nas_bib.utils.registre import register_class
import torch.nn as nn

@register_class(registry="train_methods")
class Trainer(abc.ABC):
    def __init__(self, config={}):
        """
        Initialize the training method with a configuration.
        """
        self.config = config
    #dataset needed params 
        dataset_params = self.config.get('dataset_params' , {})
        channels = dataset_params.get('channels', 3)
        hight =  dataset_params.get('hight', 32)
        width = dataset_params.get('width' , 32)

    #Extract training parameters
        self.lr_scheduler = None

        train_method_config = self.config.get('train_method', {}).get('config', {})

        #optimizer config 
        optimizer_config = train_method_config.get('optimizer', {})
        self.optimizer_name = optimizer_config.get('name', 'Adam')
        self.optimizer_params = optimizer_config.get('params', {'lr': 0.001})
 
        #loss function config
        loss_function_config = train_method_config.get('lossFunction', 'CrossEntropyLoss')
        self.criterion = getattr(nn, loss_function_config)()

        # training parameters related to data
        data_config = train_method_config.get('data', {})
        self.batch_size = data_config.get('batch_size', 32)
        self.num_epochs = data_config.get('num_epochs', 2)

        # max training time parameter
        self.max_training_time = train_method_config.get('max_training_time', 300)
        
        self.input_shape = (self.batch_size, channels, hight , width)

    @abc.abstractmethod
    def train(self, model , train_loader , metric_values):
        """
        Abstract method to be implemented by child classes.
        """
        pass
