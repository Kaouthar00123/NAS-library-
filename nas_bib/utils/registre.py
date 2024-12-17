import torch
import torch.nn as nn
from torch.nn import functional as F

DEFAULT_REGISTRY = "default_registry"

# Vérifie si registres est déjà défini
if 'registres' not in globals():
    registres = {}

def register_class(registry=DEFAULT_REGISTRY):
    def wrapper(cls):
        if registry not in registres:
            registres[registry] = {}
        class_name = cls.__name__
        if class_name not in registres[registry]:
            registres[registry][class_name] = cls
        return cls
    return wrapper

def get_registered_class(registry, class_name):
    return registres.get(registry, {}).get(class_name)

def display_registries():
    """
    Display all existing registries and their registered classes.
    """
    for registry, classes in registres.items():
        print(f"Registry: {registry}")
        for class_name in classes:
            print(f"\t{class_name}")

def display_registry(registry):
    """
    Display the names of classes in the specified registry.
    """
    if registry in registres:
        print(f"Registry: {registry}")
        for class_name in registres[registry]:
            print(f"\t{class_name}")
    else:
        print(f"No registry named '{registry}' found.")