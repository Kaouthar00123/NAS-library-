from abc import ABC, abstractmethod
import torch
import networkx as nx
import torch.nn as nn
from nas_bib.utils.registre import register_class, get_registered_class
from nas_bib.nas_components.search_spaces.core.GraphModule  import GraphModule

#Classe de base d'espace de recherche Neural network, contient les méthodes de bases et neccesaires pour manipuler espace et architcture NN, qui sont:
    #verfication de validiter de configuration de fichier de configuration, compare des architctures,
    #sampling random, modifier arch, encoder, prepare_evaluation, get et set performance
    #Justification: metter tous les oper neccesaires, car generalment, les méthodes qui s'existe sont pas indéadnats, à une archi seulle mais lié à espace et d'aillers va etres lui appliquer de plus cette atchis crre des transformations, donc, de pref les metter dans une classe specialser de ça, et pas dans classe à part d'architecture,
    #de plus: saper selon les paradigmes de constructures, car chaque'une est tout une represntation et à ces restrenction et unité de represnations, c vrai meme strcture, meme op, meme chose, mais logique et limitation de manipluation se differt, parille pour CNN, RNN
    #justification de choix des classe(CNN, tache, ), et puis generaliser strcture (noued: comb_op, operation, edge: soit rien: skip_cnx ou operation), genraliser chainé, et regle hearchique, puis changer type d'op, add module, train,
#construct ou init avec parse(car sont lié et modfier doit se faire pareils), compile, forward(car au liéu de faire structe puis convert, c'es trop car majourité pas besion de structure)
#Idée, d'avoir un espace générale, puis espace_graphe qui contine les trois diffrents paradigme(sont tous font pratiqument emel chose avec meme restriction, mais séparer selon ce qui reconnue, pour orienté user et ne pas charger d'autres espace avec plus de vérfication non-neccesaires, et eventuelle qui soit bcp plus speciliser et restrint pour cellulaire ou plus generale pour hearchique""on voit pas comme manipulation de structures mais plutot ou pattern de represntation et ces approches lié"), les fonction lié au nlogique d'espace son inlcut, mais structure de archs et ses infos sont stockes à part, ce qui plus correct en terme de charge et division de reposablit et nous permet de paralilser meme espace crre plusirs archis
#Un rollout qui continet archis(selon type cree) et infos lié (métrique et valeur, type d'espec et tt, et à chaque manipulation on apple les méthode d'espace avec cette archis, ou plutot on apple rollout qui va elle meme s'occupé d'apple espce selon méthodes)

@register_class(registry="search_spaces")
class SearchSpace():
    """
    Abstract class defining a search space for neural architecture.
    """
    def __init__(self, config):
        """
        Initialize the search space with a configuration.
        """
        self.config = config

    def sample_random_architecture(self):
        """
        Abstract method to sample a random architecture from the search space.
        """
        pass

    def mutate(self, architecture, mutation_params):
        """
        Abstract method to mutate a given architecture.
        :param architecture: The architecture to mutate.
        :param mutation_params: Parameters related to the mutation operation.
        """
        pass

    #************Commmun entre structure de graphe
    def encode(self, architecture, encoding_type):
        """
        Abstract method to encode a given architecture.
        :param architecture: The architecture to encode.
        :param encoding_type: Type of encoding to use(les plus connus: génotype, matrice d'adjasence).
        """
        pass

    def decode(self, encoded_architecture, encoding_type):
        """
        Abstract method to decode an encoded architecture.
        :param encoded_architecture: The encoded architecture.
        """
        pass

    def compile(self, architecture):
        """
        Abstract method to compile a given architecture.
        :param architecture: The architecture to compile.
        """
        pass

    def forward(self, architecture, input_data):
        """
        Abstract method to perform a forward pass through a given architecture(we stock intermedire result in cache after with desctructed cause he gonna relead processu and our objective is to get OutPut).
        :param architecture: The architecture to perform forward pass on.
        :param input_data: The input data for the forward pass.
        :return: The output of the forward pass.
        """
        pass

    def plot(self, architecture):
        """
        plot architecture
        """
        pass

    #***********Operation entre pluisieurs architectures
    def crossover(self, architecture1, architecture2, crossover_params):
        """
        Abstract method to perform crossover between two given architectures.
        :param architecture1: The first architecture.
        :param architecture2: The second architecture.
        :param crossover_params: Parameters related to the crossover operation.
        """
        pass

    def distance(self, architecture1, architecture2, metrique):
        """
        Abstract method to compute the distance between two given architectures.
        :param architecture1: The first architecture.
        :param architecture2: The second architecture.
        """
        pass

#*************Fonction de manipulation de structures de graphe specfier (add_node edge, update_node, edge, delete_node,edge)

# Classe représentant le SearchSpace_Graph
@register_class(registry="search_spaces")
class SearchSpaceGraph(SearchSpace):
    """
    Class representing a search space for graph-based neural architectures.
    Each graph in this search space is a graph (of type nx as it facilitates
    manipulation and has predefined functions) and a neural network module, 
    where each node has attributes: combine_op and operation, and each edge
    has an attribute: operations.
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, architecture, input_data):
        """
        Performs a forward pass through the architecture.
        """
        # Your implementation for forward pass through a graph-based architecture
        pass
    def encode(self, architecture, encoding_type):
        """
        Encodes a given architecture.
        """
        # Your implementation for encoding a graph-based architecture
        pass
    def decode(self, encoded_architecture, encoding_type):
        """
        Decodes an encoded architecture.
        """
        # Your implementation for decoding an encoded graph-based architecture
        pass
    # Methods for manipulating graph structures
    def add_node(self, architecture, node_idx, attributes):
        """
        Adds a node to the graph-based architecture.
        """
        # Convert attributes to concrete operations
        combine_op_name = attributes.get("combine_op", None)
        operation_name = attributes.get("operation", None)
        params = attributes.get("params", {})

        # Lookup combine_op and operation in the registry
        combine_op_cls = None
        operation_cls = None

        # Check if the combine_op exists in torch
        if hasattr(torch, combine_op_name):
            combine_op_cls = getattr(torch, combine_op_name)
        elif combine_op_name:
            combine_op_cls = get_registered_class("operations", combine_op_name)

        # Check if the operation exists in torch.nn
        if hasattr(nn, operation_name):
            operation_cls = getattr(nn, operation_name)
        elif operation_name:
            operation_cls = get_registered_class("operations", operation_name)

        # Instantiate combine_op and operation with parameters
        if combine_op_cls:
            combine_op = combine_op_cls
        else:
            combine_op = None

        if operation_cls:
            # Filter parameters to match the required arguments of the operation class
            valid_params = {param_name: param_value for param_name, param_value in params.items() if
                            param_name in operation_cls.__init__.__code__.co_varnames}
            operation = operation_cls(**valid_params)
        else:
            raise ValueError(f"Operation {operation_name} not found in the registry or torch.nn")

        # Add the node with the instantiated operations as attributes
        architecture.add_node(node_idx, combine_op=combine_op, operation=operation)
    def add_edge(self, architecture, node_idx1, node_idx2, attributes):
      """
      Adds an edge between two nodes in the graph-based architecture.
      """
      # Extract operation name and parameters from attributes
      operation_name = attributes.get("operation", None)
      params = attributes.get("params", {})

      if operation_name is None:
          return
      elif operation_name == "Identity":
          operation = nn.Identity()  # Identity function
      else:
          # Lookup operation in torch.nn
          operation_cls = getattr(nn, operation_name, None)
          if operation_cls is None:
              # Lookup operation in the registry
              operation_cls = get_registered_class("operations", operation_name)

          if operation_cls is not None:
              # Instantiate operation with parameters
              valid_params = {param_name: param_value for param_name, param_value in params.items() if param_name in operation_cls.__init__.__code__.co_varnames}
              operation = operation_cls(**valid_params)
          else:
              raise ValueError(f"Operation {operation_name} not found in torch.nn or the registry.")

      # Add the edge with the instantiated operation as attribute
      architecture.add_edge(node_idx1, node_idx2, operation=operation)
    def delete_node(self, architecture, node_idx):
        """
        Deletes a node from the graph-based architecture.
        """
        if architecture.has_node(node_idx):
            architecture.remove_node(node_idx)
        else:
            raise ValueError(f"Node {node_idx} does not exist in the architecture.")
    def delete_edge(self, architecture, node_idx1, node_idx2):
        """
        Deletes an edge between two nodes from the graph-based architecture.
        """
        if architecture.has_edge(node_idx1, node_idx2):
            architecture.remove_edge(node_idx1, node_idx2)
        else:
            raise ValueError(f"Edge between nodes {node_idx1} and {node_idx2} does not exist in the architecture.")
    def update_node(self, architecture, node_idx, attributes):
        """
        Updates the attributes of a node in the graph-based architecture.
        """
        if architecture.has_node(node_idx):
            current_attributes = architecture.nodes[node_idx]
            combine_op_name = attributes.get("combine_op", current_attributes.get("combine_op"))
            operation_name = attributes.get("operation", current_attributes.get("operation"))
            params = attributes.get("params", current_attributes.get("params", {}))

            # Lookup combine_op and operation in the registry
            combine_op_cls = None
            operation_cls = None

            # Check if the combine_op exists in torch
            if hasattr(torch, combine_op_name):
                combine_op_cls = getattr(torch, combine_op_name)
            elif combine_op_name:
                combine_op_cls = get_registered_class("operations", combine_op_name)

            # Check if the operation exists in torch.nn
            if hasattr(nn, operation_name):
                operation_cls = getattr(nn, operation_name)
            elif operation_name:
                operation_cls = get_registered_class("operations", operation_name)

            # Instantiate combine_op and operation with parameters
            if combine_op_cls:
                combine_op = combine_op_cls
            else:
                combine_op = None

            if operation_cls:
                # Filter parameters to match the required arguments of the operation class
                valid_params = {param_name: param_value for param_name, param_value in params.items() if
                                param_name in operation_cls.__init__.__code__.co_varnames}
                operation = operation_cls(**valid_params)
            else:
                raise ValueError(f"Operation {operation_name} not found in the registry or torch.nn")

            # Update the node with the instantiated operations as attributes
            architecture.nodes[node_idx].update({"combine_op": combine_op, "operation": operation, "params": params})
        else:
            raise ValueError(f"Node {node_idx} does not exist in the architecture.")
    def update_edge(self, architecture, node_idx1, node_idx2, attributes):
        """
        Updates the attributes of an edge between two nodes in the graph-based architecture.
        """
        if architecture.has_edge(node_idx1, node_idx2):
            current_attributes = architecture[node_idx1][node_idx2]
            operation_name = attributes.get("operation", current_attributes.get("operation"))
            params = attributes.get("params", current_attributes.get("params", {}))

            if operation_name is None:
                # Supprimer l'arête s'il n'y a pas d'opération spécifiée
                architecture.remove_edge(node_idx1, node_idx2)
                return
            elif operation_name == "Identity":
                operation = nn.Identity()  # Identity function
            else:
                # Lookup operation in torch.nn
                operation_cls = getattr(nn, operation_name, None)
                if operation_cls is None:
                    # Lookup operation in the registry
                    operation_cls = get_registered_class("operations", operation_name)

                if operation_cls is not None:
                    # Instantiate operation with parameters
                    valid_params = {param_name: param_value for param_name, param_value in params.items() if
                                    param_name in operation_cls.__init__.__code__.co_varnames}
                    operation = operation_cls(**valid_params)
                else:
                    raise ValueError(f"Operation {operation_name} not found in torch.nn or the registry.")

            # Update the edge with the instantiated operation as attribute
            architecture[node_idx1][node_idx2].update({"operation": operation, "params": params})
        else:
            raise ValueError(f"Edge between nodes {node_idx1} and {node_idx2} does not exist in the architecture.")
    def compute_input_dim(self, graph, node):
        raise NotImplementedError("the implemntation found un sub classes specifique")
    
#*********************TEST de la méthode add_node
# def test_node_and_edge():
#     config = None
#     search_space = SearchSpace_Graph(config)
#     architecture = GraphModule()

#     # Ajout de nœuds avec différentes configurations
#     search_space.add_node(architecture, 0, {"combine_op": "concat_features_maps", "operation": "Conv2d", "params": {"in_channels": 10, "out_channels": 32, "kernel_size": 3} } )
#     search_space.add_node(architecture, 1, {"combine_op": "sum_features_maps", "operation": "SeparableConv2d", "params": {"in_channels": 10, "out_channels": 32, "kernel_size": 3}})
#     search_space.add_node(architecture, 2, {"combine_op": "concat_features_maps", "operation": "MaxPool2d", "params": {"kernel_size": 2}})
#     search_space.add_node(architecture, 3, {"combine_op": "sum_features_maps", "operation": "BatchNorm2d", "params": {"num_features": 64}})

#     # Affichage de la structure après l'ajout des nœuds
#     print("Structure après ajout des nœuds :")
#     print(architecture.nodes(data=True))

#     # Ajout d'arêtes avec différentes configurations
#     search_space.add_edge(architecture, 0, 1, {"operation": None})
#     search_space.add_edge(architecture, 1, 2, {"operation": "Identity"})
#     search_space.add_edge(architecture, 2, 3, {"operation": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3}})

#     # Affichage de la structure après l'ajout des arêtes
#     print("Structure après ajout des arêtes :")
#     print(architecture.edges(data=True))

#     # Test de mise à jour d'un nœud
#     print("\nTest de mise à jour d'un nœud :")
#     search_space.update_node(architecture, 0, {"combine_op": "mean", "operation": "MaxPool2d", "params": {"kernel_size": 3}})
#     print("Structure après mise à jour du nœud 0 :")
#     print(architecture.nodes(data=True))

#     # Test de mise à jour d'une arête
#     print("\nTest de mise à jour d'une arête :")
#     search_space.update_edge(architecture, 1, 2, {"operation": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3}})
#     print("Structure après mise à jour de l'arête entre les nœuds 0 et 1 :")
#     print(architecture.edges(data=True))

#     # Test de suppression d'une arête
#     print("\nTest de suppression d'une arête :")
#     search_space.delete_edge(architecture, 1, 2)
#     print("Structure après suppression de l'arête entre les nœuds 1 et 2 :")
#     print(architecture.edges(data=True))

#     # Test de suppression d'un nœud
#     print("\nTest de suppression d'un nœud :")
#     search_space.delete_node(architecture, 2)
#     print("Structure après suppression du nœud 2 :")
#     print(architecture.nodes(data=True))



# # Exécution du test
# test_node_and_edge()
