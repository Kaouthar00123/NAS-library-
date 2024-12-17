import copy
import logging
from nas_bib.nas_components.search_methodes.search_method import SearchMethod
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpace
from nas_bib.nas_components.search_methodes.utils import verify_duplication
from nas_bib.nas_components.predictor.acquisition_functions import acquisition_functions

from nas_bib.utils.registre import register_class, get_registered_class
import torch


@register_class(registry="search_methods")
class BayesianOptimizer(SearchMethod):
    def __init__(self, trainer = None, evaluator = None, config = {}, dedup=True):
        """
        Initialize the Bayesian optimizer with the search space, surrogate model, acquisition function,
        configuration, and trainer.
        """

        super().__init__(config=config, dedup=dedup)  # Call the __init__ method of the parent class

        self.config = config

        self.trainer = trainer 

        self.evaluator = evaluator

        search_method_config = config.get('search_method', {})

        self.surrogate_model_name = search_method_config.get('config', {}).get('surrogate_model', 'GaussianProcessSurrogate')
        print('2: ' , self.surrogate_model_name)

        self.acquisition_func_name = search_method_config.get('config', {}).get('acquisition_func', 'expected_improvement')

        self.surrogate_model = get_registered_class("surrogate_models", self.surrogate_model_name)
        print('1: ' , self.surrogate_model)

        if not self.surrogate_model:
            self.surrogate_model = get_registered_class("surrogate_models", 'GaussianProcessSurrogate')


        self.surrogate_model_instance = self.surrogate_model()
        self.acquisition_func_instance = acquisition_functions.get(self.acquisition_func_name)

        self.iteration = -1

        self.history = set()

        self.num_iterations = search_method_config.get('num_iterations', 5)

    def search(self , exp_manager, train_loader , eval_loader ): 

        while self.check_continuous_condition():

            self.iteration = self.iteration + 1

            print('iteration : ', self.iteration)
            i = 0
            model = self.one_iteration()

            if model is not None:
                i = i+1
                print('*****************************************************')
                print('Generated Model')
                model.print_recursive()
                print('*****************************************************')
                
                self.trainer.train(model , train_loader , self.metrics_values)
                
                print("now evaluating it ")
                self.evaluator.evaluate(model , eval_loader , self.metrics_values)
            
                self.update_surrogate_model(model, self.metrics_values["valid_acc"][-1])

                print('*****************************************************')
                # Transformer le modèle en format JSON
                json_data = model.graphmodule_to_json( model_id=i, metrics = self.metrics_values)
                print(json_data)
                # Écrire les données JSON dans un fichier
                model.write_json_to_file(json_data, filename = self.config['result']['outs_file'])
                print('*****************************************************')

    def one_iteration(self):
        """
        Perform one step of Bayesian optimization, including selecting the next architecture,
        evaluating it, and updating the surrogate model.
        """
        model = self._generate_model()

        self.iteration = self.iteration + 1
        
        self.update_history(model)  
        
        # Check if we should use the surrogate model to evaluate the next architecture
        if self.should_use_surrogate():

            accuracy = self.evaluate_with_surrogate(model)
            # Update metrics list
            metrics_values = {'valid_acc': accuracy}
            self.update_metrics_values(metrics_values)
            print('predicted accuracy ', accuracy)
            return None
            
        return model

    def _generate_model(self):
        """
        Generate a random architecture.

        Returns:
        - model (torch.nn.Module): The generated model architecture.
        """
        model = self.search_space.sample_random_architecture()
        return model

    def evaluate_architecture(self, model):
        """
        Evaluate the given architecture using the trainer and return its accuracy.
        """
        self.trainer.train(model)
        return model.accuracy

    def evaluate_with_surrogate(self, model):
        # Extract parameters from the model
        parameters = [param for param in model.parameters()]

        # Concatenate parameters into a single tensor
        X = torch.cat([param.view(-1) for param in parameters])

        # Reshape X to have a batch dimension if necessary
        X = X.unsqueeze(0)  # Add a batch dimension

        # Call the surrogate model's predict method
        predicted_accuracy, _ = self.surrogate_model_instance.predict(X = X)

        return predicted_accuracy

    def check_continuous_condition(self):
        """
        Determines whether a continuous condition for a search process is valid.

        Returns:
        - bool: True if the continuous condition is valid, otherwise False.
        """
        return self.iteration < self.num_iterations

    def adapt_search_space(self, search_space: SearchSpace, scope: str = None, dataset_api: dict = None):
        """
        This method has to be called with the search_space before the optimizer
        can be used.

        Args:
            search_space : An instance of the search space
            scope        : Relevant only for one-shot optimizers
            dataset_api  : NAS Benchmark API for the given search space
        """
        self.search_space = search_space
        self.dataset_api = dataset_api

    def should_use_surrogate(self):
        """
        Determine whether to use the surrogate model for evaluation based on certain conditions.
        """
        return self.iteration > self.config.get('surrogate_threshold', 2)

    def update_surrogate_model(self, model, accuracy):
        # Convert the model's parameters to a PyTorch tensor
        model_parameters = list(model.parameters())

        if len(model_parameters) > 0:
            # Convert model parameters to a PyTorch tensor
            X = torch.cat([param.view(-1) for param in model_parameters])

            # Prepare the target (accuracy) as a tensor
            y = torch.tensor([accuracy]).numpy()

            # Update the surrogate model with the observed data
            self.surrogate_model_instance.fit(X.reshape(1, -1), y)
        else:
            print("Warning: Model parameters contain no elements. Skipping surrogate model update.")

    def update_history(self, model, accuracy=None):
        """
        Update the search history with the model.

        Parameters:
        - model: model.
        """
        self.history.add(model)
