import yaml
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpaceGraph 
from nas_bib.nas_components.search_spaces.core.operations import concat_features_maps, sum_features_maps, SeparableConv2d 
from nas_bib.nas_components.search_spaces.core.GraphModule import GraphModule
from nas_bib.utils.fct_utils import get_operation_instance, create_and_assign_operation
from nas_bib.utils.registre import register_class, get_registered_class
from nas_bib.nas_components.search_spaces.task.TaskLastStage import  ClassificationLastStage

config_task = {
    'num_classes': 10,
    'hidden_units': 512,
    'pool_size': 1,
    'num_linear_layers': 1,
    'activation_function': nn.ReLU(inplace=True),
    'output_activation': nn.Softmax(dim=1)
}
# Creating an instance of the ClassificationLastStage class
task = ClassificationLastStage(None, config_task)

@register_class(registry="search_spaces")
class HierarchicalSearchSpace(SearchSpaceGraph):

    DEFAULT_CONFIG = {
        "primitive_operations": ['AvgPool2d-kernel_size:2x2', 'MaxPool2d-kernel_size:3x3', 'SeparableConv2d-kernel_size:3x3', 'Conv2d-kernel_size:3x3', 'Identity'],
        "hierarchical_params": {
            "num_levels": 2,
            "details_levels": [{'num_level': 1, 'num_nodes': 3},  {'num_level': 2, 'num_nodes': 4}]
        },
        "other_configs": {
            "out_channels": [32, 64],
            "kernel_size": [(1, 1), (2, 2)]
        },
    }

    def __init__(self, task=task, config={}, input_dim=3):
        """
        Initializes the HierarchicalSearchSpace class with the parameters from the configuration file.

        Args:
            config (dict): A dictionary containing the configuration parameters.
        """
        self.task = task
        self.input_dim = input_dim

        self.primitive_operations = config.get('primitive_operations', self.DEFAULT_CONFIG["primitive_operations"])
        hierarchical_params = config.get('hierarchical_params', self.DEFAULT_CONFIG["hierarchical_params"])
        self.num_levels = hierarchical_params.get('num_levels')
        self.details_levels = hierarchical_params.get('details_levels', [])
        self.other_configs = config.get('other_configs', self.DEFAULT_CONFIG["other_configs"])

    def __str__(self):
        return (f"Primitive Operations: {self.primitive_operations}\n"
                f"Number of Levels: {self.num_levels}\n"
                f"Details Levels: {self.details_levels}\n"
                f"Other Configurations: {self.other_configs}")
    
    def calculate_input_dim(self, graph, from_node, to_node, in_dim):
        """
        Calculate the input dimension for the given edge in the graph.

        Args:
            graph (nx.DiGraph): The NetworkX graph.
            from_node (int): The source node of the edge.
            to_node (int): The target node of the edge.
            in_dim (int): The input dimension.

        Returns:
            int: The calculated input dimension for the edge.
        """
        if graph.nodes[to_node]["input_dim"] is not None:
            if isinstance(graph.nodes[to_node]["combine_op"], concat_features_maps):
                return graph.nodes[to_node]["input_dim"] + in_dim
            elif isinstance(graph.nodes[to_node]["combine_op"], sum_features_maps):
                return min(graph.nodes[to_node]["input_dim"], in_dim)
            else:
                return in_dim
        else:
            return in_dim

    def _create_graph_level(self, level, num_nodes):
        """
        Creates a simple graph representing a level motif graph.

        Args:
            self: Instance of the HierarchicalSearchSpace class.
            level (int): Level of the motif.
            num_nodes (int): Number of nodes in the motif.

        Returns:
            A NetworkX graph representing the level motif.
        """
        graph = GraphModule()
        graph.graph['motif_level'] = level
        graph.graph["id"] = ""
        for i in range(num_nodes):
            if i == 0:
                graph.add_node(0, input_dim=None)
            else:
                if i == (num_nodes - 1):
                    graph.add_node(i, combine_op=concat_features_maps(), input_dim=None)
                else:
                    graph.add_node(i, combine_op=sum_features_maps(), input_dim=None)
                for j in range(i):
                    graph.add_edge(j, i)
                    graph.edges[j, i]['operation'] = None
        return graph
    
    def _fill_edges_recursively(self, graph, input_dim, motifs_level, details_levels, current_level):
        """
        Recursively fill edges of a graph with motifs.

        Args:
            graph (nx.DiGraph): The NetworkX graph whose edges need to be filled.
            motifs_level (list): A list of NetworkX graphs representing motifs of each level.
            details_levels (list): A list of details of each level.
            current_level (int): The current level.
        """

        # Extract details of the current level
        level_details = next((detail for detail in details_levels if detail.get('num_level') == current_level), None)

        # Assign input channels to the first node of the graph
        graph.nodes[0]["input_dim"] = input_dim

        for edge in graph.edges:
            from_node, to_node = edge

            edge_ops = level_details.get('edge_operations', [])

            specific_edge_op = next((edge_op for edge_op in edge_ops if edge_op.get('from') == from_node and edge_op.get('to') == to_node), None)

            if specific_edge_op:
                operation_name = random.choice(specific_edge_op.get('allowed_operations', []))
                if operation_name == "none":
                    graph.edges[edge]['operation'] =  None
                else:
                    in_ch = graph.nodes[from_node]['input_dim']
                    graph.edges[edge]['operation'] = create_and_assign_operation(operation_name, self.other_configs, in_ch)
                    in_ch_s = graph.edges[edge]['operation'].out_channels if 'out_channels' in graph.edges[edge]['operation'].__dict__ else in_ch
                    graph.nodes[to_node]["input_dim"] = self.calculate_input_dim(graph, from_node, to_node, in_ch_s)
            else:
                niveau_choisi = random.randint(0, graph.graph.get('motif_level', 1) - 1)
                in_ch = graph.nodes[from_node]['input_dim']
                if niveau_choisi == 0:
                    operation_name = random.choice(self.primitive_operations)
                    if operation_name == "none":
                        graph.edges[edge]['operation'] =  None
                    else:
                        graph.edges[edge]['operation'] = create_and_assign_operation(operation_name, self.other_configs, in_ch)
                        in_ch_s = graph.edges[edge]['operation'].out_channels if 'out_channels' in graph.edges[edge]['operation'].__dict__ else in_ch
                        graph.nodes[to_node]["input_dim"] = self.calculate_input_dim(graph, from_node, to_node, in_ch_s)
                else:
                    motif_copie = motifs_level[niveau_choisi - 1].copy()
                    motif_copie.graph["id"] = f"motif_{graph.graph.get('id', '')}_{edge}"
                    self._fill_edges_recursively(motif_copie, in_ch, motifs_level, details_levels, niveau_choisi)
                    graph.edges[edge]['operation'] = motif_copie
                    id_last_node_sous_graphe , data = list(motif_copie.nodes(data=True))[-1]
                    graph.nodes[to_node]["input_dim"] = self.calculate_input_dim(graph, from_node, to_node, data['input_dim'])
        
        edges_to_remove = [(u, v) for u, v, data in graph.edges(data=True) if data.get('operation') is None]
        graph.remove_edges_from(edges_to_remove)
    
    def sample_random_architecture(self):
        """
        Function to randomly generate an architecture from the search space.

        Args:
            self: Instance of the HierarchicalSearchSpace class.

        Returns:
            A NetworkX graph representing the generated architecture.
        """

        # Creating and initializing level motifs
        motifs_level = []
        for i in range(1, self.num_levels + 1):
            motif_level = self._create_graph_level(i, self.details_levels[i - 1]["num_nodes"])
            motifs_level.append(motif_level)

        # Creating a GraphModule object
        final_graph = motifs_level[-1].copy()
        final_graph.graph["id"] = "Global_arch"
        # Recursive filling of the edges of the last level graph
        self._fill_edges_recursively(final_graph, self.input_dim, motifs_level, self.details_levels, self.num_levels)
        
        last_node, data = list(final_graph.nodes(data=True))[-1]

        self.task.config_specific_task['input_dim'] =  data["input_dim"]
        final_graph.nodes[last_node]["operation"] = self.task.create_last_layers(final_graph)

        final_graph.parse()
        return final_graph

    def visualize_architecture(self, G, motif_colors):
        def recursive_draw(G, pos, motif_colors, ax=None):
            if ax is None:
                ax = plt.gca()
            # Generating the ID for the graph
            graph_id = G.graph['id']

            graph_center = nx.spring_layout(G, center=(0.5, 0.5))  # Using spring_layout with a specified center
            #ax.annotate(graph_id, xy=(graph_center[0], graph_center[1]), xytext=(graph_center[0], graph_center[1] + 0.05),ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Creating a new figure for each recursive call
            plt.figure(figsize=(10, 8))

            # Generating the ID for the graph
            graph_id = G.graph['id']

            # Retrieving the motif_level of the graph
            motif_level = G.graph["motif_level"]

            plt.title( graph_id)
            # Drawing the nodes
            for node, data in G.nodes(data=True):
                if 'combine_op' in data:
                    # Node size based on the motif_level
                    node_size = 300 + motif_level * 200
                    # Motif_level color
                    node_color = motif_colors.get(motif_level, 'gray')
                    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
                    nx.draw_networkx_labels(G, pos, labels={node: f"{node}\nType: {data['combine_op']}"},
                                            font_color='black', font_size=10, font_weight='bold', ax=ax)

            # Drawing the edges and storing the subgraph edges
            subgraph_edges = []
            for u, v, data in G.edges(data=True):
                if isinstance(data['operation'], str):
                    # Drawing edges and labels for primitive operations
                    edge_color = motif_colors.get(0, 'green')
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5, connectionstyle='arc3,rad=0.5' ,arrows=False, edge_color=edge_color, ax=ax)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels= f"({u}, {v}):operation = {data['operation']}", font_color=edge_color,
                                                font_size=10, font_weight='bold', ax=ax)
                elif isinstance(data['operation'], nx.Graph):
                    # Drawing edges and labels for subgraphs
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5,connectionstyle='arc3,rad=0.5' ,arrows=False, edge_color='black', ax=ax)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=f"({u}, {v}):operation = graph", font_color='black',
                                                font_size=10, font_weight='bold', ax=ax)
                    # Storing subgraph edges
                    subgraph_edges.append((u, v))

            # Displaying the figure
            plt.axis('off')
            plt.show()

            # Drawing subgraphs from stored edges
            for idx, (u, v) in enumerate(subgraph_edges):
                # Drawing the subgraph
                subgraph = G[u][v]['operation']
                pos =  nx.spring_layout(subgraph, **self.calculate_layout_params(G), seed=42)
                recursive_draw(subgraph, pos, motif_colors)


        # Calculating node layout
        pos = nx.spring_layout(G, **self.calculate_layout_params(G), seed=42)
        # Calling the recursive function to draw the graphs
        recursive_draw(G, pos, motif_colors)
    # Function to calculate spring layout parameters
    def calculate_layout_params(self, G):
        # Add your logic to calculate parameters here
        return {'center': (0.5, 0.5), 'k': 0.15, 'scale': 2.0, 'iterations': 50}













#--------------TEST: from NASLib: Hierarchical search space as defined in Liu et al.: Hierarchical Representations for Efficient Architecture Search

# config_yaml = """
# primitive_operations:
#   - SepConvDARTS-stride:1x1
#   - Zero #for none operation
#   - Identity 
#   - MaxPool2d-kernel_size:2x2
#   - AvgPool2d
#   - DepthwiseConv
#   - ConvBNReLU

# hierarchical_params:
#   num_levels: 3
#   details_levels:
#     - num_level: 1 #motif
#       num_nodes: 4

#     - num_level: 2 #cell
#       num_nodes: 5

#     - num_level: 3 #external_graph
#       num_nodes: 14
#       edge_operations:
#         - from: 0
#           to: 1
#           allowed_operations:
#             - StemNASLib
#         - from: 2
#           to: 3
#           allowed_operations:
#             - SepConvDARTS-out_channels:16&kernel_size:3&stride:1&padding:1
#         - from: 4
#           to: 5
#           allowed_operations:
#             - SepConvDARTS-out_channels:32&kernel_size:3&stride:2&padding:1
#         - from: 6
#           to: 7
#           allowed_operations:
#             - SepConvDARTS-out_channels:32&kernel_size:3&stride:1&padding:1

#         - from: 8
#           to: 9
#           allowed_operations:
#             - StemNASLib
#         - from: 10
#           to: 11
#           allowed_operations:
#             - SepConvDARTS-out_channels:64&kernel_size:3&stride:2&padding:1
#         - from: 12
#           to: 13
#           allowed_operations:
#             - SepConvDARTS-out_channels:64&kernel_size:3&stride:1&padding:1

# other_configs:
#   out_channels:
#     - 16
#     - 32
#     - 64
#   stride:
#     - 1
#     - 2
#   kernel_size:
#     - 3
#     - 5
#   expansion_ratio:
#     - 1
#     - 6
# """
# config = yaml.safe_load(config_yaml)

# config = config if isinstance(config, dict) else {}

# # Créer une instance de la classe Hierarchical_Search_Space
# search_space = HierarchicalSearchSpace(config = config, input_dim=3)

# # Appeler la fonction random_sample_arch pour obtenir une architecture aléatoire
# architecture_aleatoire1 = search_space.sample_random_architecture()
# Affichage de l'architecture générée
# print("***************Print d'architecture*************")
# architecture_aleatoire1.print_recursive()









# config_yaml = """
# primitive_operations:
#   - SepConvDARTS-stride:1x1
#   - Zero
#   - Identity
#   - MaxPool2d-kernel_size:2x2
#   - AvgPool2d
#   - DepthwiseConv
#   - ConvBNReLU

# hierarchical_params:
#   num_levels: 3
#   details_levels:
#     - num_level: 1 #motif
#       num_nodes: 4

#     - num_level: 2 #cell
#       num_nodes: 5

#     - num_level: 3 #external_graph
#       num_nodes: 4
#       edge_operations:
#         - from: 0
#           to: 1
#           allowed_operations:
#             - StemNASLib
#         - from: 2
#           to: 3
#           allowed_operations:
#             - SepConvDARTS-stride:2x2
#         # - from: 4
#         #   to: 5
#         #   allowed_operations:
#         #     - SepConvDARTS-stride:2x2
#         # - from: 6
#         #   to: 7
#         #   allowed_operations:
#         #     - SepConvDARTS-stride:1x1
# other_configs:
#   out_channels:
#     - 16
#     - 32
#     - 64
#   stride:
#     - 1
#     - 2
#   kernel_size:
#     - 3
#     - 5
#   expansion_ratio:
#     - 1
#     - 6
# """

# # config_yaml = """
# # primitive_operations:
# #   - SepConvDARTS-stride:1x1
# #   - Zero
# #   - Identity
# #   - MaxPool2d-kernel_size:2x2
# #   - AvgPool2d-kernel_size:2x2
# #   - DepthwiseConv
# #   - ConvBNReLU

# # # hierarchical_params:
# # #   num_levels: 3
# # #   details_levels:
# # #     - num_level: 1 #motif
# # #       num_nodes: 2

# # #     - num_level: 2 #cell
# # #       num_nodes: 2

# # #     - num_level: 3 #external_graph
# # #       num_nodes: 4
# # #       edge_operations:
# # #         - from: 0
# # #           to: 1
# # #           allowed_operations:
# # #             - StemNASLib
# # #         - from: 2
# # #           to: 3
# # #           allowed_operations:
# # #             - SepConvDARTS-stride:2x2
# # # other_configs:
# # #   out_channels:
# # #     - 16
# # #     - 32
# # #     - 64
# # #     - 128
# # #   stride:
# # #     - 1
# # #     - 2
# # #   kernel_size:
# # #     - [3,3]
# # #   expansion_ratio:
# # #     - 1
# # #     - 3
# # #     - 6
# # # """



# config_yaml = """"""

# config = yaml.safe_load(config_yaml)

# config = config if isinstance(config, dict) else {}

# # Créer une instance de la classe Hierarchical_Search_Space
# search_space = HierarchicalSearchSpace(config = config, input_dim=3)

# # Appeler la fonction random_sample_arch pour obtenir une architecture aléatoire
# architecture_aleatoire1 = search_space.sample_random_architecture()
# # Affichage de l'architecture générée
# print("***************Print d'architecture*************")
# architecture_aleatoire1.print_recursive()


# full_train = full_train()
# accurancy = full_train.train(architecture_aleatoire1)
# print(f"finale accurancy arch1 : {accurancy}")

# architecture_aleatoire2 = search_space.sample_random_architecture()
# accurancy = train(architecture_aleatoire2)
# print(f"finale accurancy arch2 : {accurancy}")

# architecture_aleatoire3 = search_space.sample_random_architecture()
# accurancy = train(architecture_aleatoire3)
# print(f"finale accurancy arch3 : {accurancy}")

# # print(architecture_aleatoire)
# search_space.print_recursive(architecture_aleatoire1)

# # random_architecture.forward( X = torch.randn(1, 3, 20, 20) )
# # trainloader, testloader = load_data('CIFAR10', train_size=500, test_size=100, batch_size=50)
# # model = random_architecture
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# # train_model(trainloader, testloader, model, criterion, optimizer, epochs=2)

# # #affichage et visualisation de graphe
# # pos = nx.spring_layout(random_architecture)
# # motif_colors = {0: 'green', 1: 'blue', 2: 'orange'}

# # hierarchical_space.visualize_architecture(random_architecture, motif_colors)