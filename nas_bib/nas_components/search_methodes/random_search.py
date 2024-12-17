import logging

from nas_bib.nas_components.search_methodes.utils import (line_fmt, retry_sampling,
    set_random_seeds, verify_duplication)
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpace
from nas_bib.utils.registre import register_class
from nas_bib.nas_components.search_methodes.search_method import SearchMethod

logger = logging.getLogger(__name__)

@register_class(registry="search_methods")
class RandomSearch(SearchMethod):
    def __init__(self, config={}, trainer=None, evaluator=None):
        """
        Random Search strategy for architecture search.

        Parameters:
        - config: Dictionary containing configuration parameters.
        - trainer: Trainer object for training models.
        - evaluator: Evaluator object for evaluating models.
        """
        super().__init__(config=config)  # Call the __init__ method of the parent class
        set_random_seeds(config.get('seed', 0))  # Set random seeds for reproducibility

        self.num_iterations = config.get('num_iterations', 6)-1
        self.dedup = config.get('dedup', True)
        self.metrics = config.get('metrics', ['valid_acc'])
        print(self.metrics)
        self.metrics_str = ', '.join(self.metrics)
        self.continuous_condition = config.get('continuous_condition', 'num_iterations')
        self.metrics_values = {metric: [] for metric in self.metrics}
        
        # Use a set for faster membership checking
        self.history = set()  
        
        self.trainer = trainer 
        self.evaluator = evaluator
        self.actual_iteration = -1
        self.outs_file = self.config.get('result', {}).get('outs_file', 'outs_file')

    def search(self, exp_manager , train_loader=None, eval_loader=None):
        logger.info("\n--------------- Stepping Random Search ---------------")

        logger.info("Stepping Random Search...")
        logger.info(line_fmt.format('Continuous condition:', self.continuous_condition))
        logger.info(line_fmt.format('Total number of iterations:', self.num_iterations+1))
        metrics_str = ', '.join(self.metrics)
        logger.info(f'Metrics to Measure: {metrics_str}')
        logger.info("\n------------------------------------------------------")

        generated_model_id = -1

        while self.check_continuous_condition():
            self.actual_iteration += 1
            logger.info(line_fmt.format('Actual iteration number:', self.actual_iteration))
            model = None
            model = self.one_iteration()
            generated_model_id = generated_model_id+1

            if model is not None:
                logger.info("Start Training the model...")
                if(self.trainer):
                    self.trainer.train(model , train_loader , self.metrics_values)
                else: logger.error('The trainer in invalid , None object.')
                
                logger.info("Start Evaluating the model...")
                
                if(self.evaluator):
                    self.evaluator.evaluate(model , eval_loader , self.metrics_values)
                else: logger.warning('The evaluator in invalid , None object.')

                # Convert the model and corresponding metrics data to JSON format for storage
                # and write the JSON data to a file specified by 'outs_file'
                last_elements = {metric: values[-1] for metric, values in self.metrics_values.items()}
                json_data = model.graphmodule_to_json(model_id=generated_model_id, metrics = last_elements)
                exp_manager.log_scalar(last_elements, self.actual_iteration)
                model.write_json_to_file(json_data, filename=self.outs_file)
            else:
                self.actual_iteration -= 1
                generated_model_id = generated_model_id-1

    def one_iteration(self):
        """
        Perform one step of random search.

        Returns:
        - model: Sampled architecture.
        """
        logger.info("Retry sampling until a valid model...")
        model = retry_sampling(self._generate_model)
        return model

    def _generate_model(self):
        """
        Generate a random architecture.

        Returns:
        - model (torch.nn.Module): The generated model architecture.
        """
        model = self.search_space.sample_random_architecture()
        
        if model is None:
            logger.error('Generated model is None...')
            raise ValueError("Model object cannot be None.")

        if self.dedup:
            # Check for duplication if deduplication is enabled
            if not verify_duplication(self.history, model, raise_on_dup=True):
                logger.info('Generated model already exist...')
                return None
        return model

    def adapt_search_space(self, search_space: SearchSpace, scope=None, dataset_api=None):
        """
        This method has to be called with the search_space before the optimizer
        can be used.

        Args:
            search_space : An instance of the search space, such as NasBench201SearchSpace()
            scope        : Relevant only for one-shot optimizers
            dataset_api  :dataset API for the given search space
        """ 
        # TODO: Add a copy function inside the search space to ensure proper initialization.
        self.search_space = search_space
        self.dataset_api = dataset_api

    def check_continuous_condition(self):
        """
        Determines whether a continuous condition for a search process is valid.

        Returns:
        - bool: True if the continuous condition is valid, otherwise False.
        """
        return self.actual_iteration < self.num_iterations

    def get_best_architectures(self, num_architectures=1, metric='valid_acc'):
        """
        Get the best architectures found during the search.

        Args:
        - num_architectures (int, optional): Number of best architectures to return. Defaults to 1.
        - metric (str, optional): Metric to use for sorting architectures. Defaults to 'valid_acc'.

        Returns:
        - best_architectures (list): List of best architectures found.
        """
        if metric not in self.metrics_values:
            raise ValueError(f"Unsupported metric: {metric}")
        history_list = list(self.history)
        sorted_history = sorted(history_list, key=lambda x: self.metrics_values[metric][history_list.index(x)], reverse=True)
        return sorted_history[:num_architectures]
    
    def save_checkpoint(self):
        """
        Save search method checkpoint.
        """
        #TODO  Implementation of saving checkpoint
        logger.info("Checkpoint saved.")

    def load_checkpoint(self):
        """
        Load search method checkpoint.
        """
        #TODO  Implementation of loading checkpoint
        logger.info("Checkpoint loaded.")
