import math
import os
import time
import torch

import logging

import yaml

from nas_bib.exp_manager.data_loader import DatasetLoader

from nas_bib.exp_manager.experiment import generate_experiment_id
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from matplotlib.ticker import MaxNLocator

from nas_bib.exp_manager.utils import (convert_number, DEFAULT_PRECISION, flops_to_string,
    flops_to_value, memory_usage_to_string, memory_usage_to_value, number_to_string,
    number_to_value, params_to_string, params_to_value, round_tick_interval, runtime_to_string,
    runtime_to_value)
save_dir = '././././save'
metric_units = {
            'model_size': 'params',
            'train_acc': '%',
            'valid_acc': '%',
            'runtime': 's',
            'memory_usage': 'B',
            'latency': 'ms',
            'FLOPs': 'FLOPs',
        }
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manage NAS experiments.

    Parameters
    ----------
    Attributes
    ----------
    config : str | None
        Experiment configuration.
    id : str
        Experiment ID.
    """


    def __init__(self, search_method , search_space , config=None, id=None):

        self.config = config

        self.id = id if id else generate_experiment_id()

        self.search_method = search_method

        self.search_space = search_space 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_loader = DatasetLoader(self.config)

        self.writerFolder =  self.config.get("outs_metrics", "writer")
        self.metric_data = {}

        self.setup_tensorboard(self.writerFolder)
    def start_experiment(self):
        """
        Start the experiment, before training process.
        """
        logger.info("Starting the experiment...")
        # TO DO Implementation of starting the experiment
    def stop_experiment(self):
        """
        Stop the experiment, after training process.
        """
        logger.info("Stopping the experiment...")

        self.close_tensorboard()
        # TO DO Implementation of stopping the experiment
    def run_experiment(self):
        """
        Run the experiment, start the architecture search.
        """

        self.start_experiment()

        logger.info("Running the experiment...")

        # Load data
        self.train_loader = self.data_loader.get_loader(train=True)
        self.eval_loader = self.data_loader.get_loader(train=False)
        
        # self.subset_train = self.data_loader.get_subset_loader()
        # self.subset_eval = self.data_loader.get_subset_loader(train=False)

        self.search_method.adapt_search_space(search_space = self.search_space)
        
        self.search_method.search(self, self.train_loader , self.eval_loader)

        resulted_metrics_values = self.search_method.get_metrics_values()
                # Log all histograms at once
        self.log_plots()
        self.log_combined_plots()
        self.log_all_scalars()

        self.stop_experiment()
        best_architectures = self.search_method.get_best_architectures(num_architectures=1)

        batch_size = 1
        channels = self.config.get('dataset_params',{}).get('channels', 3)
        hight =  self.config.get('dataset_params',{}).get('hight', 32)
        width = self.config.get('dataset_params',{}).get('width' , 32)
        input_shape = (batch_size, channels, hight , width)

        dummy_input = torch.randn(input_shape)

        self.log_graph(best_architectures[0], dummy_input)

        logger.info("Experiment completed successfully.")
    def save_checkpoint(self):
        """
        Save experiment checkpoint.
        """
        # Implementation of saving checkpoint
        pass
    def load_checkpoint(self):
        """
        Load experiment checkpoint.
        """
        # Implementation of loading checkpoint
        pass
    def visualize_model_architecture(self, model):
        """
        Visualize the model architecture.

        Parameters
        ----------
        model : object
            The model to visualize.
        """
    def set_device(device):
        """
        Set the device for the experiment.

        Parameters
        ----------
        device : str
            Device to be used ('cuda' or 'cpu').
        """
        ExperimentManager.device = device
    def setup_tensorboard(self, log_dir):
        """
        Setup TensorBoard for logging.

        Parameters
        ----------
        log_dir : str
            Directory path to save TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        # self.writer = tf.summary.create_file_writer(log_dir)
    def log_scalar(self, model_metrics, step):
        """
        Log scalar values to TensorBoard as discrete points.

        Parameters
        ----------
        model_metrics : dict
            Dictionary containing model metrics.
        step : int
            Step or iteration number.
        """
        for metric_name, metric_value in model_metrics.items():
            self.writer.add_scalar(f'{metric_name}', metric_value, step)
            if metric_name not in self.metric_data:
                self.metric_data[metric_name] = []
            self.metric_data[metric_name].append((step, metric_value))
    def log_all_scalars(self):
        """
        Log all accumulated metrics to TensorBoard as discrete points.
        """
        for metric_name, values in self.metric_data.items():
            scalar_dict = {f'step_{step}': value for step, value in values}
            self.writer.add_scalars(f'{metric_name}_plot', scalar_dict, global_step=0)
    def log_plots(self):
        """
        Log accumulated metrics as custom plots.
        """
        metric_units = {
            'model_size': 'params',
            'train_acc': '%',
            'valid_acc': '%',
            'runtime': 's',
            'memory_usage': 'B',
            'latency': 'ms',
            'FLOPs': 'FLOPs',
        }

        # Function to get formatted metric values
        def format_metric(metric_name, value):
            if metric_name == 'model_size':
                return params_to_value(value)
            elif metric_name == 'train_acc' or metric_name == 'valid_acc':
                print(value)
                return (value)
            elif metric_name == 'runtime':
                return runtime_to_value(value )
            elif metric_name == 'memory_usage':
                return memory_usage_to_value(value)
            elif metric_name == 'latency':
                return (value)
            elif metric_name == 'FLOPs':
                result =  convert_number(value)
                metric_units['FLOPs'] = result[1]+"FLOPs"
                return result[0]
            else:
                return f"{value:.{DEFAULT_PRECISION}f}"

        for metric_name, values in self.metric_data.items():

            steps, metrics = zip(*values)
            formatted_metrics = [format_metric(metric_name, v) for v in metrics]

            fig, ax = plt.subplots()
            ax.scatter(steps, formatted_metrics)
            # Add metric unit to the plot title
            unit = metric_units.get(metric_name, 'unit')
            ax.set_title(f"{metric_name}")
            ax.set_xlabel('Step')
            ax.set_ylabel(f"{metric_name} ({unit})")

            # Set integer ticks on the x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # Convert buffer to an image and remove alpha channel
            image = Image.open(buf).convert('RGB')
            image = np.array(image)
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0

            # Log the image
            self.writer.add_image(f'{metric_name}_plot', image[0], global_step=0)

            # Save the plot to the specified directory
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{metric_name}.png')
            with open(save_path, 'wb') as f:
                f.write(buf.getbuffer())
            print(f'Plot saved to {save_path}')
    def log_combined_plots(self):
        """
        Log accumulated metrics as a single custom plot with different colors for each metric,
        using a key map to indicate the unit for each metric.
        """
        fig, ax1 = plt.subplots(figsize=(10, 8))  # Adjust width and height as needed

        # Adjust the position of the key card within the larger plot
        key_card_x = 0.75  # Adjust the x-coordinate as needed
        key_card_y = 1.14  # Adjust the y-coordinate as needed

        dpi = 300  # Resolution in dots per inch, by default 300.

        # Dictionary to map metric names to their units
        metric_units = {
            'model_size': 'params',
            'train_acc': '%',
            'valid_acc': '%',
            'runtime': 's',
            'memory_usage': 'B',
            'latency': 'ms',
            'FLOPs': 'FLOPs',
        }

        # Function to get formatted metric values
        def format_metric(metric_name, value):
            if metric_name == 'model_size':
                return params_to_value(value)
            elif metric_name == 'train_acc' or metric_name == 'valid_acc':
                print(value)
                return (value)
            elif metric_name == 'runtime':
                return runtime_to_value(value )
            elif metric_name == 'memory_usage':
                return memory_usage_to_value(value)
            elif metric_name == 'latency':
                return number_to_value(value)
            elif metric_name == 'FLOPs':
                result =  convert_number(value)
                metric_units['FLOPs'] = result[1]+"FLOPs"
                return result[0]
            else:
                return f"{value:.{DEFAULT_PRECISION}f}"


        # Iterate over each metric and plot it
        for i, (metric_name, values) in enumerate(self.metric_data.items()):
            steps, metrics = zip(*values)
            
            # Convert and format the metric values
            formatted_metrics = [format_metric(metric_name, v) for v in metrics]

            # Plot the metric
            color = f"C{i}"  # Use different colors for each metric
            yticks = ax1.get_yticks()
            num_ticks = len(yticks)

            ax1.scatter(steps, formatted_metrics, color=color)  # No label here

            unit = yticks[1] if yticks[0] == 0 else yticks[0]
            note = "0" if yticks[0] == 0 else "1"
            # Add key card to indicate unit values for each metric
            key_card = f"{metric_name}: {num_ticks} 'ticks', {note} note, 1 unit = {unit} {metric_units.get(metric_name, 'unit')}"
            ax1.annotate(key_card, xy=(0.95, key_card_y - i * 0.05), xycoords='axes fraction', va='top', ha='right', color=color)

        # Set title and x-axis label
        ax1.set_title('Metrics Over Steps')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Metric Value')

        # Set integer ticks on the x-axis
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add legend to indicate units for each metric
        ax1.legend()

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        plt.close(fig)

        
        os.makedirs(save_dir, exist_ok=True)
        print(f'Directory {save_dir} created or already exists.')

        save_path = os.path.join(save_dir, 'combined_metrics_plot.png')
        with open(save_path, 'wb') as f:
            f.write(buf.getbuffer())
        print(f'Plot saved to {save_path}')
        
        # Convert buffer to an image and remove alpha channel
        image = Image.open(buf).convert('RGB')
        image = np.array(image)
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Log the image
        self.writer.add_image('combined_metrics_plot', image[0], global_step=0)
        print('Image logged to TensorBoard.')
    def log_graph(self, model, input):
        """
        Log scalar value to TensorBoard.

        Parameters
        ----------
        tag : str
            Name of the scalar data.
        value : float
            Value to be logged.
        step : int
            Step or iteration number.
        """
        self.writer.add_graph(model, input)
    def close_tensorboard(self):
        """
        Close TensorBoard writer.
        """
        self.writer.close()

#example 
# file_path = "./configFile.yaml"
# # Load configuration from file
# with open(file_path, 'r') as file:
#     config = yaml.safe_load(file)

# # # Création d'une instance de la classe Chain_Search_Space
# seach_space_params = config.get('search_space')

# conf_ss = seach_space_params.get('config') 


# search_space = CellSearchSpace(config = conf_ss , input_dim=3)

# trainer = FullTrain(config)

# evaluate = SimpleEvaluator(config)

# search_method = RegularizationEvaluation(config = config , trainer = trainer , evaluator = evaluate )

# # Create an instance of ExperimentManager
# experiment_manager = ExperimentManager(search_method , search_space ,  config)

# # Run the experiment
# experiment_manager.run_experiment()