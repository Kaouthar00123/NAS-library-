import yaml
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpaceGraph 
from nas_bib.nas_components.search_spaces.core.operations import concat_features_maps, sum_features_maps, SeparableConv2d 
from nas_bib.nas_components.search_spaces.core.GraphModule import GraphModule
from nas_bib.utils.fct_utils import create_and_assign_operation, randomize_parameters, get_module_and_update_param, get_operation_index
from nas_bib.utils.registre import register_class, get_registered_class
from nas_bib.nas_components.search_spaces.task.TaskLastStage import TaskLastStage, ClassificationLastStage

# Configuration de la tâche
configTask = {
    'num_classes': 10,
    'hidden_units': 512,
    'pool_size': 1,
    'num_linear_layers': 1,
    'activation_function': nn.ReLU(inplace=True),
    'output_activation': nn.Softmax(dim=1)
}

task = ClassificationLastStage(None, configTask)

@register_class(registry="search_spaces")
class CellSearchSpace(SearchSpaceGraph):
    default_config = {
        "num_cells": 3,
        "num_cell_groups": {0: "normal", 1: "reduction"},
        "cells_layout": [0, 0, 1],
        "cell_groups_details": {
            "num_nodes_per_cell": 4,
            "num_init_nodes": 1,
            "num_init_inputs": 2,
            "primitive_operations": ['AvgPool2d', 'MaxPool2d', 'SeparableConv2d', 'Conv2d', 'Identity'],
            "other_configs": {
                "out_channels": [32, 64],
                "kernel_size": [(1, 1), (2, 2)]
            }
        }
    }

    def __init__(self, task=task, config={}, input_dim=3):
        """
        Initialize the Cell Search Space.

        Args:
            task (Task): Task object representing the current task.
            config (dict): Configuration dictionary.
            input_dim (int): Dimension of the input.
        """
        self.task = task
        self.input_dim = input_dim

        self.num_cells = config.get("num_cells", self.default_config["num_cells"])
        self.num_cell_groups = config.get("num_cell_groups", self.default_config["num_cell_groups"])
        self.cells_layout = config.get("cells_layout", self.default_config["cells_layout"])

        self.num_nodes_per_cell = config.get("cell_groups_details", {}).get("num_nodes_per_cell", self.default_config["cell_groups_details"]["num_nodes_per_cell"])
        self.num_init_nodes = config.get("cell_groups_details", {}).get("num_init_nodes", self.default_config["cell_groups_details"]["num_init_nodes"])
        self.num_init_inputs = config.get("cell_groups_details", {}).get("num_init_inputs", self.default_config["cell_groups_details"]["num_init_inputs"])
        self.primitive_operations = config.get("cell_groups_details", {}).get("primitive_operations", self.default_config["cell_groups_details"]["primitive_operations"])
        self.other_configs = config.get("cell_groups_details", {}).get("other_configs", self.default_config["cell_groups_details"]["other_configs"])

        self.list_cell_infos, self.template_arch = self.create_template_arch()

    def __str__(self):
        """
        Return a string representation of the Cell Search Spaces Configuration.
        """
        return f"Cell Search Spaces Configuration:\n" \
               f"num_cells: {self.num_cells}\n" \
               f"num_cell_groups: {self.num_cell_groups}\n" \
               f"cells_layout: {self.cells_layout}\n" \
               f"num_nodes_per_cell: {self.num_nodes_per_cell}\n" \
               f"num_init_nodes: {self.num_init_nodes}\n" \
               f"num_init_inputs: {self.num_init_inputs}\n" \
               f"primitive_operations: {self.primitive_operations}\n" \
               f"other_configs: {self.other_configs}"

    def fill_cell_group_infos(self):
        """
        Fill cell group information.

        Returns:
            list: List of cell group information dictionaries.
        """
        cell_group_infos = []
        num_cell_groups = len(self.num_cell_groups)

        for i in range(num_cell_groups):
            group_info = {
                "id": i,
                "type": self.num_cell_groups[i],
                "num_nodes_per_cell": self.num_nodes_per_cell[i] if isinstance(self.num_nodes_per_cell, dict) else self.num_nodes_per_cell,
                "num_init_nodes": self.num_init_nodes[i] if isinstance(self.num_init_nodes, dict) else self.num_init_nodes,
                "num_init_inputs": self.num_init_inputs[i] if isinstance(self.num_init_inputs, dict) else self.num_init_inputs,
                "primitive_operations": self.primitive_operations[i] if isinstance(self.primitive_operations, dict) else self.primitive_operations,
                "other_configs": self.other_configs[i] if isinstance(self.other_configs, dict) and isinstance(next(iter(self.other_configs.values())), dict) else self.other_configs
            }
            cell_group_infos.append(group_info)

        return cell_group_infos
    
    def create_cell_graph(self, cell_info):
        """
        Crée un graphe NetworkX basé sur les informations d'une cellule spécifique.
        """
        num_nodes = cell_info["num_nodes_per_cell"]

        G = GraphModule()
        G.add_nodes_from(range(num_nodes))

        # Ajout de l'attribut combine_op à chaque nœud avec la valeur "sum"
        nx.set_node_attributes(G, {node: {"combine_op": sum_features_maps()} for node in G.nodes()})

        # Ajout d'un nœud supplémentaire à la fin avec l'attribut combine_op défini sur "concat"
        G.add_node(num_nodes, combine_op = concat_features_maps())

        # Parcours des nœuds précédents
        for j in range(num_nodes):
            # Ajout d'une arête avec l'opération "Identity"
            G.add_edge(j, num_nodes, operation = nn.Identity())

        return G

    def create_template_arch(self):
        # Création du graphe template_arch
        template_arch = GraphModule()

        # Création du premier nœud avec l'attribut input_dim
        template_arch.add_node(0, input_dim=self.input_dim)

        # Création des cellules
        list_cell_infos = self.fill_cell_group_infos()
        list_cell_graphs = [self.create_cell_graph(cell_info) for cell_info in list_cell_infos]

        # Création et liaison des autres nœuds
        for i in range(1, self.num_cells +1):
            id = self.cells_layout[i-1]
            template_arch.add_node(i, id=id, typeCell = list_cell_infos[id]["type"]+"-cell")
            # Lien entre le nœud précédent et le nœud actuel
            if(i != 0):
                template_arch.add_edge(i-1, i, operation = nn.Identity())

            # Attribution de la cellule correspondante au nœud
            cell_graph = list_cell_graphs[self.cells_layout[i-1]]
            template_arch.nodes[i]["operation"] = cell_graph.copy()

        # Création du premier nœud avec l'attribut input_dim
        last_node = self.num_cells

        template_arch.add_node(max(template_arch.nodes) + 1) # va contenir le nœud final du classificateur
        template_arch.add_edge(last_node, last_node +1 , operation = nn.Identity())

        return list_cell_infos, template_arch

    def compute_input_dim(self, node, G, in_channel_graph_previous):
        """
        Computes the input dimension of a node in the architecture graph.

        Args:
            node (int): Node identifier.
            G (networkx.Graph): Architecture graph.
            in_channel_graph_previous (int): Input dimension of the previous graph.

        Returns:
            int: Input dimension of the node.
        """
        if node == 0:
            return in_channel_graph_previous
        else:
            input_dim = []
            # Loop over the predecessors of the current node
            for predecessor in G.predecessors(node):
                edge_data = G.get_edge_data(predecessor, node)
                operation = edge_data.get("operation")
                if operation:
                    # If the operation has an out_channels parameter, use it
                    if 'out_channels' in operation.__dict__:
                        input_dim.append(operation.out_channels)
                    else:
                        # Otherwise, use the input_dim of the previous node
                        input_dim.append(G.nodes[predecessor].get("input_dim", 0))
                else:
                    # Use the input_dim of the previous node
                    input_dim.append(G.nodes[predecessor].get("input_dim", 0))

            # Calculating input_dim based on the combination
            combine_op = G.nodes[node].get("combine_op")
            if isinstance(combine_op, concat_features_maps):
                result = sum(input_dim)
            elif isinstance(combine_op, sum_features_maps):
                result = min(input_dim)
            else:
                result = input_dim[0]  # Default value if no combination is specified
            return result

    def sample_random_architecture(self):
        """
        Creates a random architecture from the initialized search space.

        Returns:
            networkx.Graph: Random architecture graph.
        """
        # Create a copy of the initial graph
        arch_random = self.template_arch.copy()
        ensemble_graphes = []

        # Iterate over all cell groups
        for cell_group_id, cell_info in enumerate(self.list_cell_infos):
            # Extract information from the cell group
            num_nodes_per_cell = cell_info["num_nodes_per_cell"]
            primitive_operations = cell_info["primitive_operations"]
            num_nodes_inputs = cell_info["num_init_nodes"]
            params = cell_info["other_configs"]

            # Create the cell graph from the cell group information
            cell_graph = self.create_cell_graph(cell_info)

            # Modify connections and primitive operations for each cell stage
            for idx in range(1, num_nodes_per_cell):
                preceding_nodes = list(range(idx))
                for _ in range(num_nodes_inputs):
                    node_id = random.choice(preceding_nodes)
                    operation = random.choice(primitive_operations)
                    cell_graph.add_edge(node_id, idx, operation=operation, params=params)

            ensemble_graphes.append(cell_graph)

        input_dim_prev = self.input_dim  # Initialize the input_dim variable
        
        # Iterate over each node in arch_random
        for i in range(1, len(self.cells_layout) + 1):
            node_id = arch_random.nodes[i]["id"]
            operation_graph = ensemble_graphes[node_id].copy()  # Copy the corresponding sub-graph

            # Update the "operation" attribute of node i in arch_random
            arch_random.nodes[i]["operation"] = operation_graph

            # Iterate over each node in operation_graph
            for node in list(operation_graph.nodes())[:-1]:
                for predecessor in operation_graph.predecessors(node):
                    edge_data = operation_graph.get_edge_data(predecessor, node)
                    operation_name = edge_data["operation"]
                    params = edge_data["params"]
                    in_parm = operation_graph.nodes[predecessor]["input_dim"]
                    del operation_graph[predecessor][node]['operation']
                    del operation_graph[predecessor][node]['params']
                    operation_graph[predecessor][node]["operation"] = create_and_assign_operation(operation_name, params, in_parm)

                operation_graph.nodes[node]["input_dim"] = self.compute_input_dim(node, operation_graph, input_dim_prev)

            last_node = list(operation_graph.nodes())[-1]
            operation_graph.nodes[last_node]["input_dim"] = self.compute_input_dim(last_node, operation_graph, input_dim_prev)
            input_dim_prev = operation_graph.nodes[last_node].get("input_dim")

        last_node = list(arch_random.nodes())[-1]
        self.task.config_specific_task['input_dim'] = input_dim_prev
        arch_random.nodes[last_node]["operation"] = self.task.create_last_layers(arch_random)
        
        arch_random.parse()
        return arch_random

    def mutate(self, architecture, mutate_node_prob=0.5):
        """
        Mutates the given architecture.

        Args:
            architecture (networkx.Graph): Architecture graph.
            mutate_node_prob (float): Probability of mutating a node.

        Returns:
            networkx.Graph: Mutated architecture graph.
        """
        arch_mutated = architecture.copy()
        num_cell_groups = len(self.num_cell_groups)

        # Randomly select a cell group to mutate
        mutate_i_cg = np.random.randint(0, num_cell_groups - 1)
        # Choose the first node of this type of group from cell_layout (first appearance index of the value mutate_i_cg)
        id_first_cell_graph = self.cells_layout.index(mutate_i_cg) + 1
        cell_graph = arch_mutated.nodes[id_first_cell_graph]["operation"]

        # Recover information from the previous node
        if id_first_cell_graph != 1:
            cell_before_last = arch_mutated.nodes[id_first_cell_graph - 1]["operation"]
            input_dim_prev = cell_before_last.nodes[max(cell_before_last.nodes)].get("input_dim")
        else:
            input_dim_prev = arch_mutated.nodes[0].get("input_dim")

        # Get information about the chosen cell group
        cell_info = self.fill_cell_group_infos()[mutate_i_cg]
        num_nodes_per_cell = cell_info["num_nodes_per_cell"]
        primitive_operations = cell_info["primitive_operations"]
        params = cell_info["other_configs"]
        num_primitives = len(primitive_operations)

        node_mutated = np.random.randint(1, num_nodes_per_cell - 1)
        cnx_mutated = np.random.randint(0, cell_graph.in_degree(node_mutated))

        # print(f"in mutate, mutate_i_cg: {mutate_i_cg}, node_mutated: {node_mutated}, cnx_mutated: {cnx_mutated}")

        if np.random.random() < mutate_node_prob:  # Mutation of the connection
            # print("Connection mutation case")
            # Select the connection related to node_mutated with cnx cnx_mutated and attribute value (in class name) of this cnx
            # Change connection between mutated node and its predecessor which are linked by cnx_mutated, change predecessor of this node
            edge = list(cell_graph.in_edges(node_mutated, data=True))[cnx_mutated]
            specific_predecessor = list(cell_graph.predecessors(node_mutated))[cnx_mutated]
            operation_name = random.choice(primitive_operations)
            new_params = randomize_parameters(params)
            # print(f"new_params: {new_params}")
            # Create a list of predecessors, excluding the specific one
            list_exclude = [i for i in range(node_mutated - 1) if i != specific_predecessor]
            if not list_exclude:
                return self.mutate(architecture, mutate_node_prob=random.random())

            new_predecessor_id = random.choice(list_exclude)

            # Update graph from modified group to the end, as it will impact the whole graph since nodes are connected and changing one parameter will affect other_configs
            for i in range(id_first_cell_graph, len(self.cells_layout) + 1):
                cell = arch_mutated.nodes[i]["operation"]  # Get the sub-graph of node i
                node_list = list(cell.nodes())  # Extract a list of nodes from the cell
                for node_id in node_list:  # Traverse the nodes of the cell with this list
                    if (arch_mutated.nodes[i]["id"] == mutate_i_cg) and (node_id == node_mutated):
                        # Special treatment for the node with an identifier matching idx_cell_grp
                        cell.remove_edge(specific_predecessor, node_mutated)
                        cell.add_edge(new_predecessor_id, node_mutated, operation=create_and_assign_operation(operation_name, new_params, cell.nodes[new_predecessor_id]["input_dim"]))
                    cell.nodes[node_id]["input_dim"] = self.compute_input_dim(node_id, cell, input_dim_prev)
                    # Update outgoing edges of the current node
                    for succ_node_id in cell.successors(node_id):
                        # Update the operation of the outgoing edge
                        cell.edges[node_id, succ_node_id]["operation"] = get_module_and_update_param(cell.edges[node_id, succ_node_id]["operation"], "input_dim", cell.nodes[node_id]["input_dim"])

                # Update for the next one
                input_dim_prev = cell.nodes[max(cell.nodes)].get("input_dim")

        else:  # Mutation of the operation
            # print("Operation mutation case")
            offset = np.random.randint(1, num_primitives)
            specific_predecessor = list(cell_graph.predecessors(node_mutated))[cnx_mutated]
            # print(f"specific_predecessor: {specific_predecessor}")
            old_op_idx = get_operation_index(cell_graph.edges[(specific_predecessor, node_mutated)]["operation"], primitive_operations)
            new_op = primitive_operations[(old_op_idx + offset) % num_primitives]
            new_params = randomize_parameters(params)
            # print(f"new_params: {new_params}")
            for i in range(id_first_cell_graph, len(self.cells_layout) + 1):
                cell = arch_mutated.nodes[i]["operation"]  # Get the sub-graph of node i
                node_list = list(cell.nodes())  # Extract a list of nodes from the cell
                for node_id in node_list:  # Traverse the nodes of the cell with this list
                    if (arch_mutated.nodes[i]["id"] == mutate_i_cg) and (node_id == node_mutated):
                        # Special treatment for the node with an identifier matching idx_cell_grp
                        input_dim = cell.nodes[specific_predecessor]["input_dim"]
                        # print(f"new_op: {new_op}, params:{params}, input_dim:{input_dim}")
                        cell.edges[(specific_predecessor, node_mutated)]["operation"] = create_and_assign_operation(new_op, new_params, cell.nodes[specific_predecessor]["input_dim"])
                    cell.nodes[node_id]["input_dim"] = self.compute_input_dim(node_id, cell, input_dim_prev)
                    # Update outgoing edges of the current node
                    for succ_node_id in cell.successors(node_id):
                        # Update the operation of the outgoing edge
                        cell.edges[node_id, succ_node_id]["operation"] = get_module_and_update_param(cell.edges[node_id, succ_node_id]["operation"], "input_dim", cell.nodes[node_id]["input_dim"])

                # Update for the next one
                input_dim_prev = cell.nodes[max(cell.nodes)].get("input_dim")

        last_node = list(arch_mutated.nodes())[-1]
        self.task.config_specific_task['input_dim'] = input_dim_prev
        arch_mutated.nodes[last_node]["operation"] = self.task.create_last_layers(arch_mutated)

        arch_mutate_new = arch_mutated.unparse()
        arch_mutate_new.parse()

        return arch_mutate_new

#********************************** Test from AW_NAS adapte: 
#which is compatible to the ENAS, DARTS, FBNet search spaces, as well as many baseline networks

# config_yaml = """
# num_cells: 8
# num_cell_groups:
#     0: normal
#     1: reduction
# cells_layout: [ 0,0, 1,0, 0,1, 0,0 ]
# cell_groups_details:
#     num_nodes_per_cell: 4
#     # num_init_nodes: 2
#     # num_init_inputs: 2
#     primitive_operations:
#         - Zero
#         - MaxPool2d-kernel_size:3x3
#         - AvgPool2d-kernel_size:3x3
#         - Identity
#         - SeparableConv2d-kernel_size:3x3
#         - SeparableConv2d-kernel_size:5x5
#         - DilatedConv2d-kernel_size:3x3
#         - DilatedConv2d-kernel_size:5x5
#     other_configs:
#         out_channels:
#             - 32
#             - 64
# """
# config = yaml.safe_load(config_yaml)

# config = config if isinstance(config, dict) else {}

# search_space = CellSearchSpace(config = config,input_dim= 3)
# print(search_space)

# # print("*******************RANDOM************************")
# # Appeler la fonction random_sample_arch pour obtenir une architecture aléatoire
# architecture_aleatoire1 = search_space.sample_random_architecture()
# Affichage de l'architecture générée
# print("***************Print d'architecture*************")
# architecture_aleatoire1.print_recursive()

# config_yaml = """
# num_cells: 3
# num_cell_groups:
#     0: normal
#     1: reduction
# cells_layout: [0,0, 1]
# cell_groups_details:
#   num_nodes_per_cell: 4
# #   num_init_nodes: 2
# #   num_init_inputs: 2
#   primitive_operations:
#     0:
#         - Zero
#         - Identity
#         - Conv2d-kernel_size:3x3&padding:1
#         - SepConvDARTS-kernel_size:3x3
#         - DilConvDARTS-kernel_size:3x3
#     1:
#         - Zero
#         - Identity
#         - MaxPool2d-kernel_size:3x3
#         - AvgPool2d-kernel_size:3x3 
#         - Conv2d-kernel_size:2x2
# """


# config_yaml = """
# num_cells: 3
# num_cell_groups:
#   0: normal
#   1: reduction
# cells_layout: [0, 0, 1]
# cell_groups_details:
#   num_nodes_per_cell:
#     0: 4
#     1: 3
#   num_init_nodes:
#     0: 2
#     1: 2
#   num_init_inputs:
#     0: 2
#     1: 2
#   primitive_operations:
#     0:
#       - AvgPool2d_3x3
#       - MaxPool2d_3x3
#       - SeparableConv2d_3x3
#       - Conv2d_3x3
#       - Identity
#     1:
#       - SeparableConv2d_3x3
#       - Conv2d_5x5
#       - Identity
# """



# config_yaml = """"""

# config = yaml.safe_load(config_yaml)

# config = config if isinstance(config, dict) else {}

# search_space = CellSearchSpace(config = config,input_dim= 3)
# print(search_space)

# print("*******************RANDOM************************")
# # Appeler la fonction random_sample_arch pour obtenir une architecture aléatoire
# architecture_aleatoire1 = search_space.sample_random_architecture()
# # Affichage de l'architecture générée
# print("***************Print d'architecture*************")
# architecture_aleatoire1.print_recursive()




# full_train = full_train()
# accurancy = full_train.train(architecture_aleatoire1)
# print(f"finale accurancy arch1 : {accurancy}")

# architecture_aleatoire2 = search_space.sample_random_architecture()
# accurancy = full_train.train(architecture_aleatoire2)
# print(f"finale accurancy arch2 : {accurancy}")

# architecture_aleatoire3 = search_space.sample_random_architecture()
# accurancy = full_train.train(architecture_aleatoire3)
# print(f"finale accurancy arch3 : {accurancy}")

# # print(architecture_aleatoire)
# architecture_aleatoire1.print_recursive()

# print("*******************MUTATION************************")
# architecture_mutated1 = search_space.mutate(architecture_aleatoire1, mutate_node_prob = random.random())
# accurancy = full_train.train(architecture_mutated1)
# print(f"finale accurancy mutate_arch1 : {accurancy}")
# architecture_mutated1.print_recursive()

# architecture_mutated2 = search_space.mutate(architecture_mutated1, mutate_node_prob = random.random())
# accurancy = full_train.train(architecture_mutated2)
# print(f"finale accurancy mutate_arch2 : {accurancy}")

