#---------Config de fichier finale

# task_params:
#   - type: classification
#   - objective_metric: accuracy
#   - num_classes: 10
#   - dataSet: CIFAR10

# arch_type: cnn

# search_space:
#   pattern: chain_structured
#   config: # to be defined

# search_strategy:
#   method: RE
#   config: # to be defined

# train_strategy:
#   method: Full_train
#   config:
#     optimizer: adam
#     learning_rate: 0.3

#----------------------------------Principale Code-------------------------

from nas_bib.exp_manager.experiment_manager import ExperimentManager
from nas_bib.nas_components.search_spaces.task.TaskLastStage import ClassificationLastStage
from nas_bib.utils.registre import get_registered_class, display_registries, display_registry
import os , importlib.util, yaml
import torch.nn as nn

import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level
console_handler = logging.StreamHandler()  # Create console handler
logger.addHandler(console_handler)  # Add console handler
logger.propagate = False  # Disable propagation

def execute_files_in_directory(directory):
    """
    Exécute tous les fichiers Python (.py) dans un répertoire donné et ses sous-répertoires.

    Args:
        directory (str): Le chemin du répertoire à parcourir.
    """
    # Parcourir récursivement le répertoire et ses sous-répertoires
    current_file_dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(directory):
        # Filtrer les fichiers pour ne garder que les fichiers Python
        dirs[:] = [d for d in dirs if not d.startswith('__')]
        python_files = [f for f in files if f.endswith('.py')]
        # Exécuter chaque fichier Python trouvé
        for file in python_files:
            file_path = os.path.join(root, file)
            execute_file(file_path)

        # Appeler récursivement execute_files_in_directory pour chaque sous-répertoire
        for subdir in dirs:
            if not subdir.startswith('__'):  # Exclure les dossiers commençant par '__'
                subdir_path = os.path.join(root, subdir)
                # Appel récursif avec le sous-répertoire
                execute_files_in_directory(subdir_path)

def execute_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    exec(code, globals())

def import_all_from_file(file_name):
    # Obtenir le chemin absolu du fichier spécifié
    abs_file_path = os.path.abspath(file_name)
    # Obtenir le nom du module à partir du nom de fichier
    module_name = os.path.splitext(os.path.basename(abs_file_path))[0]
    try:
        # Trouver le spécificateur (specifier) du module
        spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
        if spec is None:
            logger.error(f"Impossible de trouver le spécificateur pour le fichier {file_name}")
            return
        # Charger le module depuis le spécificateur
        module = importlib.util.module_from_spec(spec)
        # Exécuter le module
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'importation du fichier {file_name}: {e}")

#------------------------------Exemple d'utilisation :

# Utilisation de la fonction pour exécuter les fichiers dans un dossier nommé 'scripts'

logger.info("Start analysing Files.....")
execute_files_in_directory('.\\nas_bib\\nas_components')
logger.info("End analysing Files.....")


#--------------------------START EXECUTION------------------
#Choose the configuration file to execute.
file_path = "test/scenario1.yaml"
# Load configuration from file
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

logger.info("---------Start execution")

search_space_class_name = config['search_space']['pattern']
search_method_class_name = config['search_method']['method']
train_method_class_name = config['train_method']['method']
eval_method_class_name = config['eval_method']['method']

# Instantiate the search space
search_space_class = get_registered_class("search_spaces", search_space_class_name)
if search_space_class:
    search_space_config = config['search_space']['config']

    input_dim = config.get('dataset_params',{}).get('channels', 3)

    if config.get('task_params',{}).get('type', 'classification') == 'classification':
        task_config = {
            'num_classes': config.get('dataset_params',{}).get('num_classes', 3),
            'hidden_units': 512,
            'pool_size': 1,
            'num_linear_layers': 1,
            'activation_function': nn.ReLU(inplace=True),
            'output_activation': nn.Softmax(dim=1)
        }
        task = ClassificationLastStage(None, task_config)

    else :
        raise ValueError("Other task types not implemented yet")
        
    search_space_instance = search_space_class(task = task , config=search_space_config, input_dim=input_dim)
    logger.info(f"Search space created with pattern: {search_space_class_name}")
else:
    print("Available search space patterns:")
    display_registry("search_spaces")
    raise ValueError("Search space '{}' not found in registry".format(search_space_class_name))

# Instantiate the train method
train_method_class = get_registered_class("train_methods", train_method_class_name)
if train_method_class:
    train_method_config = config
    train_method_instance = train_method_class(train_method_config)
    logger.info("Train method '{}' created successfully".format(train_method_class_name))
else:
    print("Available train methods:")
    display_registry("train_methods")
    raise ValueError("Train method '{}' not found in registry".format(train_method_class_name))

# Instantiate the eval method
eval_method_class = get_registered_class("eval_methods", eval_method_class_name)
if eval_method_class:
    eval_method_config = config
    eval_method_instance = eval_method_class(eval_method_config)
    logger.info("Eval method '{}' created successfully".format(eval_method_class_name))
else:
    print("Available eval methods:")
    display_registry("eval_methods")
    raise ValueError("Eval method '{}' not found in registry".format(eval_method_class_name))

# Instantiate the search method
if train_method_class and eval_method_class:
    search_method_class = get_registered_class("search_methods", search_method_class_name)
    search_method_config = config['search_method']['config']

    if search_method_class:
        search_method_instance = search_method_class(config=search_method_config, trainer=train_method_instance, evaluator=eval_method_instance)
        logger.info("Search method '{}' created successfully".format(search_method_class_name))

        # Create an instance of ExperimentManager
        experiment_manager = ExperimentManager(search_method_instance, search_space_instance, config)

        # Run the experiment
        logger.info("---------Start experiment")
        experiment_manager.run_experiment()
        logging.info("---------End experiment")
    else:
        print("Available search methods:")
        display_registry("search_methods")
        raise ValueError("Search method '{}' not found in registry".format(search_method_class_name))
else:
    logger.error("Train method and/or Eval method not created successfully. Exiting.")
    exit(1)

logger.info("---------End execution")

