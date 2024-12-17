import random
import copy
import logging
import numpy as np
import torch
from nas_bib.nas_components.search_methodes.search_method import SearchMethod
from nas_bib.nas_components.search_methodes.utils import retry_sampling, verify_duplication
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpace
from nas_bib.utils.registre import register_class

logger = logging.getLogger(__name__)

@register_class(registry="search_methods")
class RegularizationEvaluation(SearchMethod):
    def __init__(self, trainer = None, evaluator = None, config={} , dedup=True):
        """
        Initialize the RegularizationEvaluation search method with a configuration.

        Parameters:
        - search_space (object): The search space object.
        - config (dict): Configuration dictionary containing various parameters.
        """
        super().__init__(config=config)  # Call the __init__ method of the parent class
        # Extract parameters from the config dictionary
        self.trainer = trainer 
        
        self.evaluator = evaluator

        search_method_config = config.get('search_method', {})
        self.population_size = search_method_config.get('config', {}).get('population_size', 2)
        self.sample_size = search_method_config.get('config', {}).get('sample_size', 2)
        self.mutate_node_prob = search_method_config.get('config', {}).get('mutate_node_prob', 0.5)
        self.crossover = search_method_config.get('crossover', False)
        self._random_state = np.random.RandomState(config.get('seed', None))
        self._individual_counter = 0
        self.num_iterations = search_method_config.get('num_iterations', 5)
        self.sampled_archs = []
        self.history = set()  # Use a set for faster membership checking
        self.dedup = dedup
        self.iteration = -1
        self.search_space = None
        self.metrics = config.get('metrics', ['valid_acc'])
        print(self.metrics)
        self.metrics_str = ', '.join(self.metrics)
        self.continuous_condition = config.get('continuous_condition', 'num_iterations')
        self.metrics_values = {metric: [] for metric in self.metrics}

    def search(self , train_loader , eval_loader ): 

        while self.check_continuous_condition():

            self.iteration = self.iteration + 1

            model = self.one_iteration()
            i = 0
            if model is not None:
                i = i+1
                self.update_history(model)  
                self.trainer.train(model , train_loader , self.metrics_values)   
                self.evaluator.evaluate(model , eval_loader , self.metrics_values)
                print('*****************************************************')
                # Transformer le modèle en format JSON
                json_data = model.graphmodule_to_json( model_id=i, metrics = self.metrics_values)
                # Écrire les données JSON dans un fichier
                model.write_json_to_file(json_data, filename = self.config['result']['outs_file'])
                print('*****************************************************')
        pass
    
    def one_iteration(self):
        self.iteration = self.iteration + 1
        """
        Perform one step of random search.

        Returns:
        - model: Sampled architecture.
        """
        logger.info("Stepping Evaluation Search")
        
        model = self._generate_model()
        
        self.update_history(model)

        return model

    def _generate_model(self):
        """
        Generate a random architecture.

        Returns:
        - model (torch.nn.Module): The generated model architecture.
        """
        print('iteration:', self.iteration)
        if self.iteration < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            print("Start sampling architectures to fill the population")
            model = self.search_space.sample_random_architecture()
            if self.dedup:
                if not verify_duplication(self.history, model, raise_on_dup=True):
                    return None
        else:
            print("Mutating architectures...")
            sample = []
            while len(sample) < self.sample_size:
                # Randomly select an index
                index = random.randint(0, len(self.history) - 1)
                # Get the element at the selected index
                candidate = list(self.history)[index]  # Get the candidate architecture
                # Get the corresponding valid accuracy from metrics_values list
                valid_acc = self.metrics_values['valid_acc'][index]
                # Append the candidate architecture with its valid accuracy to the sample
                sample.append((candidate, valid_acc))
            parent = max(sample, key=lambda x: x[1])  # Accessing accuracy as x[1]
            print('parent', parent[0])
            original_architecture = copy.deepcopy(parent[0])
            logger.info("Mutating architecture...")
            # mutated model
            model = self.search_space.mutate(original_architecture, mutate_node_prob=self.mutate_node_prob)
        return model

    def check_continuous_condition(self):
        """
        Determines whether a continuous condition for a search process is valid.

        Returns:
        - bool: True if the continuous condition is valid, otherwise False.
        """
        return self.iteration < self.num_iterations

    def update_history(self, model):
        """
        Update the search history with the evaluated model.

        Parameters:
        - model: Evaluated model.
        - accuracy: Accuracy of the evaluated model.
        """
        if self.history:
            self.history.add(model)

    def adapt_search_space(self, search_space: SearchSpace, scope: str = None, dataset_api: dict = None):
        """
        This method has to be called with the search_space and the nas benchmark api before the optimizer
        can be used.

        Args:
            search_space : An instance of the search space, such as NasBench201SearchSpace()
            scope        : Relevant only for one-shot optimizers
            dataset_api  : NAS Benchmark API for the given search space
        """
        # TO DO (.clone())
        self.search_space = search_space
        self.dataset_api = dataset_api

    def get_checkpointables(self):
        """
        Get checkpointable objects for saving the search state.

        Returns:
        - checkpointables (dict): Checkpointable objects.
        """
        return {"model": self.history}

    def save_checkpoint(self):
        """
        Save search method checkpoint.
        """
        # Implementation of saving checkpoint
        print("Checkpoint saved.")

    def load_checkpoint(self):
        """
        Load search method checkpoint.
        """
        # Implementation of loading checkpoint
        print("Checkpoint loaded.")
