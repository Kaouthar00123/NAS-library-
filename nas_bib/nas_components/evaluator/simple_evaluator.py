import logging

import torch


from nas_bib.utils.registre import register_class

from calflops import calculate_flops
from nas_bib.nas_components.evaluator.evaluator import Evaluator

#from nas_bib.exp_manager.experiment_manager import ExperimentManager


@register_class(registry="eval_methods")
class SimpleEvaluator(Evaluator):
    def __init__(self, config={}):
        super().__init__(config)


    def evaluate(self,model ,testloader, metric_values):

        # Initialize logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)


        correct = 0
        total = 0
        test_loss = 0
        logger.info('Start evaluating...')

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate test loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

        accuracy = 100 * correct / total
        
        if 'valid_acc' in metric_values :
            metric_values['valid_acc'].append(accuracy)

        # Calculate average test loss
        avg_test_loss = test_loss / len(testloader)
        if 'valid_loss' in metric_values :
            metric_values['valid_loss'].append(avg_test_loss)

        print(f"Validation Accuracy: {accuracy:.2f}%")

        #ExperimentManager.log_scalar('Accuracy/Valid', accuracy)
        logger.info('Accuracy of the network on the validation images: %d %%' % accuracy)

        return metric_values