import logging

from nas_bib.exp_manager.utils import convert_number, memory_usage_to_string, params_to_string, runtime_to_string
from nas_bib.nas_components.metrics_script import calculate_model_size, measure_net_latency


from nas_bib.nas_components.trainer.trainer import Trainer
from nas_bib.utils.registre import register_class
import time
import psutil  # For memory usage
import torch
import torch.optim as optim

from calflops import calculate_flops

#from nas_bib.exp_manager.experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)

@register_class(registry="train_methods")
class FullTrain(Trainer):
    def __init__(self, config={}):
        super().__init__(config)

    def train(self, model , train_loader , metric_values):
        
        start_time = time.time()
        
        self.optimizer = getattr(optim, self.optimizer_name)(model.parameters(), **self.optimizer_params)

        # Initialize other variables for tracking metrics
        memory_usage = []
        accuracy_every_epoch = []
        loss_every_epoch = []

        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}]")
            
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                running_loss += loss.item()

                # Calculate train accuracy
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

                # Track train loss
                loss_every_epoch.append(loss.item())

                # Check if training time exceeds the maximum limit
                # if time.time() - start_time > self.max_training_time:
                #     logger.info('Training time exceeded maximum time limit of 5 minutes.')
                #     return None

            # Calculate train accuracy for the epoch
            train_acc_epoch = 100 * correct_train / total_train
            logger.info('Accuracy on train set for epoch {}: {:.2f}%'.format(epoch + 1, train_acc_epoch))
            accuracy_every_epoch.append(train_acc_epoch)
        
        logger.info('Finished Training')



        if 'train_acc' in metric_values:
            final_accuracy = accuracy_every_epoch[-1]
            metric_values['train_acc'].append(final_accuracy)
            logger.info('Final validation accuracy: ', metric_values['train_acc'][-1], ' %')

        if 'train_loss' in metric_values : 
            final_loss = loss_every_epoch[-1]
            metric_values['train_loss'].append(final_loss)
            logger.info(('Final training loss: ', (metric_values['train_loss'][-1]), ' %'))

        # Calculate total training time
        end_time = time.time()
        total_time = end_time - start_time
        if 'runtime' in metric_values : 
            metric_values['runtime'].append(total_time)
            logger.info('Runtime: {:.2f} s'.format(total_time))

        # Calculate average memory usage
        if 'memory_usage' in metric_values : 
            avg_memory_usage = sum(memory_usage) / len(memory_usage)
            metric_values['memory_usage'].append(avg_memory_usage)
            logger.info(('Average memory usage: ', memory_usage_to_string(metric_values['memory_usage'][-1]),' B'))

        # Measure latency
        if 'latency' in metric_values : 
            latency, _ = measure_net_latency(model, l_type='cpu', fast=True)  # Adjust latency measurement parameters as needed
            metric_values['latency'].append(latency)
            logger.info(f'Latency: {latency:.2f} ms')

        # Calculate FLOPs for the model
        if 'FLOPs' in metric_values : 
            flops, macs, params = calculate_flops(model=model, 
                                        input_shape=self.input_shape,
                                        output_as_string=False,
                                        output_precision=4)
            metric_values['FLOPs'].append(flops)
            if 'MACs' in metric_values : 
                metric_values['MACs'].append(macs)
            if 'params' in metric_values : 
                metric_values['params'].append(params)
            flops_value, flops_unit = convert_number(flops)
            logger.info('fwd FLOPs: ' ,flops_value ,flops_unit)

        # Calculate model size
        if 'model_size' in metric_values : 
            model_size = calculate_model_size(model)
            metric_values['model_size'].append(model_size)
            logger.info(('Model Size: ', params_to_string(metric_values['model_size'][-1],' params')))

