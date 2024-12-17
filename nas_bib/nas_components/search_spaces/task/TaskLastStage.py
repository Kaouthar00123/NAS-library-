import torch.nn as nn
import torch

class TaskLastStage:
    def __init__(self, task_info, config_specific_task):
        self.task_info = task_info
        self.config_specific_task = config_specific_task

    def create_last_layers(self, search_space):
        raise NotImplementedError("La méthode create_last_layers doit être implémentée dans les sous-classes.")

class ClassificationLastStage(TaskLastStage):
    def create_last_layers(self, search_space):
        config_specific_task = self.config_specific_task
        input_dim = config_specific_task['input_dim']
        num_classes = config_specific_task['num_classes']
        hidden_units = config_specific_task['hidden_units']
        pool_size = config_specific_task['pool_size']
        num_linear_layers = config_specific_task['num_linear_layers']
        activation_function = config_specific_task['activation_function']
        output_activation = config_specific_task['output_activation']

        layers = [
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(in_features=input_dim * pool_size * pool_size, out_features=hidden_units),
            activation_function
        ]

        for _ in range(num_linear_layers - 1):
            layers.extend([
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                activation_function
            ])

        layers.append(nn.Linear(in_features=hidden_units, out_features=num_classes))

        if output_activation:
            layers.append(output_activation)

        return nn.Sequential(*layers)

# Exemple d'utilisation
# # Exemple de tenseur d'entrée
# batch_size = 1
# num_channels = 3
# height = 32
# width = 32
# input_tensor = torch.randn(batch_size, num_channels, height, width)

# # Configuration spécifique à la tâche
# config_specific_task = {
#     'in_dim': num_channels,
#     'num_classes': 10,
#     'hidden_units': 512,
#     'pool_size': 1,
#     'num_linear_layers': 1,
#     'activation_function': nn.ReLU(inplace=True),
#     'output_activation': nn.Softmax(dim=1)
# }

# # Créer le modèle linéaire généré
# classification_last_stage = ClassificationLastStage(None,  config_specific_task)
# last_layers = classification_last_stage.create_last_layers(None)

# # Faire passer le tenseur d'entrée à travers le modèle
# output = last_layers(input_tensor)
# print("Output shape:", output.shape)
# print("Output values:", output)