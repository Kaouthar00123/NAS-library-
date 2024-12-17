import torch
import random
import inspect
from nas_bib.utils.registre import get_registered_class


def get_operation_index(operation, operation_names):
    """
    Récupère l'indice de l'opération dans la liste des noms d'opérations si elle existe.
    """
    operation_name = type(operation).__name__  # Nom de la classe de l'opération
    for i, name in enumerate(operation_names):
        if name.split('-')[0] == operation_name:
            return i
    return -1

# Définition de la fonction get_module_and_update_param
def get_module_and_update_param(module, new_param_name, new_param_value):
    # Obtenir le nom de la classe du module
    module_name = type(module).__name__
    expected_params = module.__init__.__code__.co_varnames[1:]  # Ignorer le premier paramètre 'self'

    param_values = {}
    for param_name in expected_params:
        if hasattr(module, param_name) and param_name not in ['bias', 'weight'] and not param_name.endswith('_'):
            param_values[param_name] = getattr(module, param_name)

    # # Obtenir les paramètres du module
    # Modifier le paramètre spécifique
    param_values[new_param_name] = new_param_value

    # Appeler la fonction pour obtenir une nouvelle instance du module
    return get_operation_instance(module_name, param_values)


def extract_operation_details(operation_str):
    # Vérifier si operation_str est vide
    if not operation_str:
        raise ValueError("La chaîne d'opération est vide.")

    # Séparer le nom de l'opération et les paramètres
    if "-" in operation_str:
        op_name, params_str = operation_str.split('-', 1)
    else:
        return operation_str, {}

    # Vérifier si params_str est vide
    params_dict = {}
    if  params_str:
        # Diviser les paramètres en une liste de paires clé-valeur
        params_list = params_str.split('&')
        
        # Créer un dictionnaire pour stocker les paramètres et leurs valeurs
        for param in params_list:
            key, value = param.split(':')
            if 'x' in value:
                # Diviser la valeur spéciale en deux parties
                value_before_x, value_after_x = value.split('x')
                # Ajouter un tuple au dictionnaire
                params_dict[key] = (int(value_before_x), int(value_after_x))
            else:
                try:
                    # Convertir en entier
                    params_dict[key] = int(value)
                except ValueError:
                    # Si la conversion en entier échoue, conserver la valeur en tant que chaîne de caractères
                    params_dict[key] = value
        
    return op_name, params_dict

def create_params_dict(params_dict, params_copy):
    for key, value in params_dict.items():
        # Si le paramètre existe déjà dans params_copy, écraser sa valeur
        if key in params_copy:
            params_copy[key] = value
        else:
            params_copy.update({key: value})
    return params_copy

def randomize_parameters(params):
    randomized_params = {}
    for key, values in params.items():
        if isinstance(values, list):
            if isinstance(values[0], list):
                randomized_params[key] = tuple(random.choice(values))
            else:
                randomized_params[key] = random.choice(values)
        else:
            randomized_params[key] = values
    return randomized_params

def check_init_param(operation_cls, param):
    
    init_params = inspect.signature(operation_cls.__init__).parameters
    
    if param in init_params:
        # Vérifier si le paramètre est un attribut de __init__
        return True
    else:
        return False
    
def get_operation_class(operation_name):
    """
    Retrieve the operation class based on the operation name.

    Args:
        operation_name (str): The name of the operation.

    Returns:
        class: The operation class.
    """
    # Vérifier si l'opération est None
    if operation_name is None:
        print("Erreur : Le nom de l'opération est None.")
        return None

    # Vérifier si l'opération est une opération de torch
    if hasattr(torch.nn, operation_name):
        return getattr(torch.nn, operation_name)
    elif hasattr(torch, operation_name):
        return getattr(torch, operation_name)
    else:
        return get_registered_class("operations", operation_name)


def instantiate_operation(operation_cls, parameters={}):
    """
    Instantiate the operation class with the given parameters.

    Args:
        operation_cls (class): The operation class.
        parameters (dict): The parameters for initializing the operation.

    Returns:
        torch.nn.Module: The created operation instance.
    """
    # Vérifier si la fonction d'initialisation (__init__) a des paramètres
    if len(inspect.signature(operation_cls.__init__).parameters) > 1:
        # Sélectionner les paramètres requis en fonction de la signature de __init__
        valid_params = {param_name: parameters[param_name] for param_name in inspect.signature(operation_cls.__init__).parameters if
                        param_name in parameters}
        # Instancier l'opération avec les paramètres sélectionnés
        return operation_cls(**valid_params)
    else:
        # Si la fonction d'initialisation n'a pas de paramètres autres que 'self', instancier l'opération sans paramètres
        return operation_cls()
    
def get_operation_instance(operation_name, parameters={}):
    # Vérifier si l'opération est None
    if operation_name is None:
        print("Erreur : Le nom de l'opération est None.")
        return None

    # Vérifier si l'opération est une opération de torch
    if hasattr(torch.nn, operation_name):
        operation_cls = getattr(torch.nn, operation_name)
    elif hasattr(torch, operation_name):
        operation_cls = getattr(torch, operation_name)
    else:
        operation_cls = get_registered_class("operations", operation_name)

    if operation_cls is None:
        print(f"Erreur : Opération '{operation_name}' non trouvée dans les opérations torch ou le registre.")
        return None

    # Vérifier si la fonction d'initialisation (__init__) a des paramètres
    if len( (inspect.signature(operation_cls.__init__)).parameters) > 1:
        # Sélectionner les paramètres requis en fonction de la signature de __init__
        valid_params = {param_name: param_value for param_name, param_value in parameters.items() if
                        param_name in operation_cls.__init__.__code__.co_varnames}
        # Instancier l'opération avec les paramètres sélectionnés
        return operation_cls(**valid_params)
    else:
        # Si la fonction d'initialisation n'a pas de paramètres autres que 'self', instancier l'opération sans paramètres
        return operation_cls()


#**********************************

def create_and_assign_operation(operation_name, parameters, in_dim):
    """
    Creates an operation instance from the given operation name and parameters,
    and then assigns it to the specified node in the graph.

    Args:
        graph (nx.DiGraph): The graph to assign the operation to.
        node (int): The node identifier in the graph.
        operation_name (str): The name of the operation to create.
        parameters (dict): The parameters of the operation.
        in_dim (int): The number of input channels.

    Returns:
        torch.nn.Module: The created operation instance.
    """
    # Copy parameters to avoid modifying them
    params_copy = parameters.copy()
    # Extraire les détails de l'opération
    operation, operation_details = extract_operation_details(operation_name)

    # Get an instance of the operation
    operation_cls = get_operation_class(operation)

    # Handle different operation types and their input_dim parameter mapping
    if check_init_param(operation_cls, 'in_channels'):
        params_copy["in_channels"] = in_dim
    if check_init_param(operation_cls, 'num_features'):
        params_copy["num_features"] = in_dim
    if check_init_param(operation_cls, 'input_size'):
        params_copy["input_size"] = in_dim
    if check_init_param(operation_cls, 'encoder_hidden_size'):
        params_copy["encoder_hidden_size"] = in_dim
    if check_init_param(operation_cls, 'decoder_hidden_size'):
        params_copy["decoder_hidden_size"] = in_dim

    create_params_dict(operation_details, params_copy)
    # Randomize parameters
    randomized_params = randomize_parameters(params_copy)
    # Get an instance of the operation
    operation_instance = instantiate_operation(operation_cls, randomized_params)

    return operation_instance
