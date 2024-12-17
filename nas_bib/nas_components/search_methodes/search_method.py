from abc import ABC
from nas_bib.utils.registre import register_class
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpace
import logging

logger = logging.getLogger(__name__)

@register_class(registry="search_methods")
class SearchMethod(ABC):
    def __init__(self, config):
        """
        Initialize the search method with a configuration.

        Args:
        - config (dict): Configuration dictionary.
        - trainer: Trainer object for training models.
        - evaluator: Evaluator object for evaluating models.

        """
        self.config = config
      
        # self.metrics = config.get('metrics', ['valid_acc'])

        # self.metrics_values = {metric: [] for metric in self.metrics}
        
    def search(self, train_loader, eval_loader): 
        """
        Execute the search method.

        Args:
        - train_loader: DataLoader for training data.
        - eval_loader: DataLoader for evaluation data.
        """
        pass

    def one_iteration(self, params=None):
        """
        Take a step in the search strategy.

        Returns:
        - model: Sampled architecture model.
        """
        pass

    def get_final_architecture(self):
        """
        Get the best final architecture found by the search strategy.

        Returns:
        - best_model: Best final (n) models.
        """
        pass

    def get_checkpointables(self):
        """
        Get the checkpointables for saving the state of the optimizer.

        Returns:
        - checkpointables (dict): Dictionary containing the search history.
        """
        pass

    def adapt_search_space(self, search_space: SearchSpace, scope: str = None, dataset_api: dict = None):
        """
        Set the search space for the search method.

        Args:
            search_space : An instance of the search space.
            scope (str, optional): Scope for the search space.
            dataset_api (dict, optional): NAS Benchmark API for the given search space.
        """
        pass

    def update_metrics_values(self, metrics_values):
        """
        Update the metrics values of the search method.

        Args:
            metrics_values (dict): Key-value pairs of metrics to be updated.
        """
        pass

    def get_metrics_values(self):
        """
        Get the metrics values of the search method.

        Returns:
        - metrics_values (dict): Current metrics values.
        """
        pass

    def get_best_architectures(self, num_architectures=1, metric='valid_acc'):
        """
        Get the best architectures found during the search.

        Args:
        - num_architectures (int, optional): Number of best architectures to return. Defaults to 1.
        - metric (str, optional): Metric to use for sorting architectures. Defaults to 'valid_acc'.

        Returns:
        - best_architectures (list): List of best architectures found.
        """
        pass

    def save_checkpoint(self):
        """
        Save search method checkpoint.
        """
        pass

    def load_checkpoint(self):
        """
        Load search method checkpoint.
        """
        pass

    def _generate_model(self):
        """
        Generate a random architecture.

        Returns:
        - model (torch.nn.Module): The generated model architecture.
        """
        pass

    def check_continuous_condition(self):
        """
        Determines whether a continuous condition for a search process is valid.

        Returns:
        - bool: True if the continuous condition is valid, otherwise False.
        """
        pass
