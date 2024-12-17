import yaml
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from nas_bib.nas_components.search_spaces.core.search_space_base import SearchSpaceGraph 
from nas_bib.nas_components.search_spaces.core.GraphModule import GraphModule
from nas_bib.utils.fct_utils import  get_operation_instance, create_and_assign_operation, get_module_and_update_param
from nas_bib.utils.registre import register_class, get_registered_class, display_registries
from nas_bib.nas_components.search_spaces.task.TaskLastStage import TaskLastStage, ClassificationLastStage

task_config = {
    'num_classes': 10,
    'hidden_units': 512,
    'pool_size': 1,
    'num_linear_layers': 1,
    'activation_function': nn.ReLU(inplace=True),
    'output_activation': nn.Softmax(dim=1)
}

# Creating an instance of the ClassificationLastStage class
task = ClassificationLastStage(None, task_config)

@register_class(registry="search_spaces")
class LayerSearchSpace(SearchSpaceGraph):
    # Assign default values
    DEFAULT_PRIMITIVE_OPERATIONS = ['AvgPool2d-kernel_size:3x3', 'MaxPool2d-kernel_size:3x3', 'SeparableConv2d-kernel_size:3x3', 'Conv2d-kernel_size:5x5', 'Conv2d-kernel_size:3x3']
    DEFAULT_CHAIN_SIZE = [3, 5]
    DEFAULT_NODE_OPERATIONS = []
    DEFAULT_EDGE_OPERATIONS = []
    DEFAULT_NUM_FILTERS = [32, 64]
    DEFAULT_KERNEL = [(1, 1), (2, 2)]

    def __init__(self, task=task, config={}, input_dim=3):
        #### Validation Section

        # Initialize attributes with default values or those provided in the configuration
        self.task = task
        self.input_dim = input_dim

        self.primitive_operations = config.get('primitive_operations', self.DEFAULT_PRIMITIVE_OPERATIONS)
        self.chain_size = config.get('chain_size', self.DEFAULT_CHAIN_SIZE)
        self.node_operations = config.get('node_operations', [])
        self.edge_operations = config.get('edge_operations', [])
        self.other_configs = {}

        # Assign default values for parameters in other_configs
        self.other_configs = config.get('other_configs', {'out_channels': self.DEFAULT_NUM_FILTERS, 'kernel_size': self.DEFAULT_KERNEL})
        self.other_configs['combine_op'] = config.get('other_configs', {}).get('combine_op', ['sum_features_maps'])
        ### Handle case if user doesn't specify these parameters

    def __str__(self):
        return f"LayerSearchSpace(primitive_operations={self.primitive_operations}, chain_size={self.chain_size}, node_operations={self.node_operations}, edge_operations={self.edge_operations}, other_configs={self.other_configs})"


    #***********************Function utlity
    def compute_input_dim(self, graph, node):
        predecessors = list(graph.predecessors(node))
        out_channels = []

        # Parcourir les prédécesseurs
        if(node != 0):
            for predecessor in predecessors:
                if 'operation' not in graph.nodes[predecessor]:
                    out_channels.append(graph.nodes[predecessor]["input_dim"])
                else:
                    # Faites quelque chose en cas d'absence de l'attribut 'operation'
                    # Par exemple, définissez operation à None ou effectuez une autre action
                    operation = graph.nodes[predecessor]['operation']
                    # Si l'opération est un pooling, appliquer récursivement la fonction pour ce nœud
                    if hasattr(operation, 'out_channels'):
                        # Si l'opération a à la fois input_dim et out_channels, on ajoute out_channels
                        out_channels.append(operation.out_channels)
                    elif hasattr(operation, 'in_channels'):
                        # Si l'opération a input_dim mais pas out_channels, on ajoute input_dim
                        out_channels.append(operation.input_dim)
                    else:
                        # Si l'opération n'a ni input_dim ni out_channels, on calcule input_dim récursivement
                        out_channels.append(self.compute_input_dim(graph, predecessor))

        # Vérifier s'il y a un seul prédécesseur
        if len(predecessors) == 1:
            # Si oui, les canaux d'entrée sont égaux aux canaux de sortie de ce prédécesseur
            input_dim = out_channels[0]
        else:
            # Calculer les canaux d'entrée selon la méthode de combinaison
            combine_op = graph.nodes[node].get('combine_op')
            if 'sum' in combine_op.__class__.__name__:
                input_dim = min(out_channels)
            elif 'concat' in combine_op.__class__.__name__:
                input_dim = sum(out_channels)
            else:
                print(f"Erreur: Méthode de combinaison '{combine_op}' non reconnue.")
                return None

        return input_dim
    
    def replace_max_with_index(self, node_operations, edge_operations, num_chains):
        """
        Replace "max" with the actual node index based on the number of chains
        in node_operations and edge_operations.
        """
        node_operations_copy = []
        edge_operations_copy = []

        for node_spec in node_operations:
            node_spec_copy = node_spec.copy()
            if isinstance(node_spec_copy['node_id'], str) and "max" in node_spec_copy['node_id']:
                node_spec_copy['node_id'] = (eval(node_spec_copy['node_id'].replace("max", str(num_chains))))
            node_operations_copy.append(node_spec_copy)

        for edge_spec in edge_operations:
            edge_spec_copy = edge_spec.copy()
            if isinstance(edge_spec_copy['to'], str) and "max" in edge_spec_copy['to']:
                edge_spec_copy['to'] = (eval(edge_spec_copy['to'].replace("max", str(num_chains))))
            if isinstance(edge_spec_copy['from'], str) and "max" in edge_spec_copy['from']:
                edge_spec_copy['from'] = (eval(edge_spec_copy['from'].replace("max", str(num_chains))))
            edge_operations_copy.append(edge_spec_copy)

        return node_operations_copy, edge_operations_copy
    

    #*****************************Real function
    def sample_random_architecture(self):
        # Create an empty graph using NetworkX
        G = GraphModule()

        # Determine the number of chains based on the configuration
        num_chains = random.randint(self.chain_size[0], self.chain_size[1])

        # Add a special node for the start
        G.add_node(0, input_dim=self.input_dim)

        # Replace "max" with the actual node index based on the number of chains in node_operations and edge_operations
        node_operations_copy, edge_operations_copy = self.replace_max_with_index(self.node_operations, self.edge_operations, num_chains)

        # Create nodes and connect them based on the layout pattern
        for i in range(1, num_chains + 1):  # Start from 1 since 0 is reserved for the start node
            # Determine the operation for the current node
            similar_to_node = None
            node_similair_instance = None
            if node_operations_copy:
                node_spec = next((node_spec for node_spec in node_operations_copy if node_spec['node_id'] == i), None)
                if node_spec:
                    if 'similar_to' in node_spec:
                        similar_to_node = node_spec['similar_to']
                        if similar_to_node:
                            node_similair_instance = G.nodes[similar_to_node]["operation"]
                    elif 'allowed_operations' in node_spec:
                        operation_name = random.choice(node_spec['allowed_operations'])
                else:
                    # Node is not specified, choose operation randomly from primitive_operations
                    print("in node: ", i, "we chose operation")
                    operation_name = random.choice(self.primitive_operations)

            # Add operation and parameters to the current node
            G.add_node(i)
            # Check if the current node has specified connections in edge_operations
            if any(edge_spec['to'] == i for edge_spec in edge_operations_copy):
                # Retrieve the list of indices of nodes from which to connect to the current node
                connected_nodes_indices = [edge_spec['from'] for edge_spec in edge_operations_copy if edge_spec['to'] == i]
                # Connect the specified nodes to the current node
                for node_index in connected_nodes_indices:
                    G.add_edge(node_index, i, operation=nn.Identity())
            else:
                # Connect the current node with the previous one
                G.add_edge(i-1, i, operation=nn.Identity())

            if len(list(G.predecessors(i))) > 1:
                # If yes, assign a random combine_op from self.other_config["combine_op"]
                if "combine_op" in node_operations_copy[i]:
                    G.nodes[i]["combine_op"] = get_operation_instance(node_operations_copy[i]["combine_op"])
                else:
                    G.nodes[i]["combine_op"] = get_operation_instance(random.choice(self.other_configs["combine_op"]))
            
            if(node_similair_instance):
                G.nodes[i]['operation'] = get_module_and_update_param(node_similair_instance, "in_channels" , self.compute_input_dim(G, i))
            else: 
                G.nodes[i]['operation'] = create_and_assign_operation(operation_name, self.other_configs , self.compute_input_dim(G, i))

        # Connect the last node with "end" node
        end = num_chains + 1
        G.add_node(end)
        if any(edge_spec['to'] == "end" for edge_spec in edge_operations_copy):
            # Retrieve the list of indices of nodes from which to connect to the current node
            connected_nodes_indices = [edge_spec['from'] for edge_spec in edge_operations_copy if edge_spec['to'] == "end"]
            # Connect the specified nodes to the current node
            for node_index in connected_nodes_indices:
                G.add_edge(node_index, end, operation=nn.Identity())
        else:
            # Connect the current node with the previous one
            G.add_edge(num_chains, end, operation=nn.Identity())

        if len(list(G.predecessors(end))) > 1:
            # If yes, assign a random combine_op from self.other_config["combine_op"]
            if "combine_op" in node_operations_copy[end]:
                G.nodes[end]["combine_op"] = get_operation_instance(node_operations_copy[end]["combine_op"])
            else:
                G.nodes[end]["combine_op"] = get_operation_instance("concat_features_maps")

        
        G.nodes[end]["input_dim"] = self.compute_input_dim(G, end)
        
        last_node, data = list(G.nodes(data=True))[-1]

        self.task.config_specific_task['input_dim'] =  data["input_dim"]
        G.nodes[last_node]["operation"] = self.task.create_last_layers(G)

        G.parse()
        return G

    def visualize_architecture(self, G):
        # Visualize the graph
        pos = nx.spring_layout(G)  # Layout for visualization
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)

        labels = {}
        for node in G.nodes():
            label = f"Node {node}\n"
            if 'operation' in G.nodes[node]:
                label += f"Operation: {G.nodes[node]['operation']}\n"
            labels[node] = label
        nx.draw_networkx_labels(G, pos, labels, font_color='black')

        plt.title('NAS Chain Search Architecture')
        plt.show()




# #---****************************TEST: MobileNet seach space, url: 
# #https://www.researchgate.net/publication/361260658_Design_Space_Exploration_of_a_Sparse_MobileNetV2_Using_High-Level_Synthesis_and_Sparse_Matrix_Techniques_on_FPGAs/figures?lo=1

# # Charger la configuration à partir du fichier YAML

# #MobileNetV2-space, lien: https://dl.acm.org/doi/pdf/10.1145/3524500 
# config_yaml = """
# primitive_operations:

# chain_size:
#   - 19
#   - 19

# other_configs:
#   kernel_size:
#     - 3
#     - 5 
#   expansion_ratio:
#     - 1
#     - 3
#     - 6

# node_operations:
#   - node_id: 1
#     allowed_operations:
#         - Conv2d-kernel_size:3x3&out_channels:32 

#   - node_id: 2
#     allowed_operations:
#         - MBConv-out_channels:16

#   - node_id: 3
#     allowed_operations:
#         - MBConv-out_channels:24
#   - node_id: 4
#     similar_to: 3

#   - node_id: 5
#     allowed_operations:
#         - MBConv-out_channels:32
#   - node_id: 6
#     similar_to: 5
#   - node_id: 7
#     similar_to: 5

#   - node_id: 8
#     allowed_operations:
#         - MBConv-out_channels:64
#   - node_id: 9
#     similar_to: 8
#   - node_id: 10
#     similar_to: 8
#   - node_id: 11
#     similar_to: 8

#   - node_id: 12
#     allowed_operations:
#         - MBConv-out_channels:96
#   - node_id: 13
#     similar_to: 12
#   - node_id: 14
#     similar_to: 12

#   - node_id: 15
#     allowed_operations:
#         - MBConv-out_channels:160
#   - node_id: 16
#     similar_to: 15
#   - node_id: 17
#     similar_to: 15

#   - node_id: 18
#     allowed_operations:
#         - MBConv-out_channels:320
#   - node_id: max
#     allowed_operations:
#         - AvgPool2d-kernel_size:1x1
# """

# # config_yaml = """"""
# config = yaml.safe_load(config_yaml)

# config = config if isinstance(config, dict) else {}

# # Création d'une instance de la classe Chain_Search_Space
# search_space = LayerSearchSpace(config=config, input_dim=3)
# print("layer_search_space instance cree: ", search_space, "\n")

# # Exemple d'utilisation : Génération d'une architecture aléatoire
# architecture_aleatoire1 = search_space.sample_random_architecture()

# # Affichage de l'architecture générée
# # print("***************Print d'architecture*************")
# # architecture_aleatoire1.print_recursive()

# conf = """
# primitive_operations:
#     #none in this case
# chain_size: [19, 19] #[min, max]
# node_operations:
#     - node_id: 1
#     allowed_operations:
#         - Conv2d-kernel_size:3x3&out_channels:32 
#     ...
#     - node_id: 3
#     allowed_operations:
#         - MBConv-out_channels:24
#     - node_id: 4
#     similar_to: 3 #similar operation and parametres
#     - node_id: 5
#     allowed_operations:
#         - MBConv-out_channels:32
#     - node_id: 6
#     similar_to: 5
#     - node_id: 7
#     similar_to: 5
#     ...
#     - node_id: 18
#     allowed_operations:
#         - MBConv-out_channels:320
#     - node_id: max
#     allowed_operations:
#         - AvgPool2d-kernel_size:1x1
# other_configs:
#     kernel_size: [3,5]
#     expansion_ratio: [1,3,6]
    
# """









# Appeler la fonction random_sample_arch pour obtenir une architecture aléatoire
# full_train = full_train()
# accurancy = full_train.train(architecture_aleatoire1)
# print(f"finale accurancy arch1 : {accurancy}")

# architecture_aleatoire2 = search_space.sample_random_architecture()

# accurancy = full_train.train(architecture_aleatoire2)
# print(f"finale accurancy arch2 : {accurancy}")

# architecture_aleatoire3 = search_space.sample_random_architecture()
# accurancy = full_train.train(architecture_aleatoire3)
# print(f"finale accurancy arch3 : {accurancy}")


# trainloader, testloader = load_data('CIFAR10', train_size=500, test_size=100, batch_size=50)
# model = random_architecture
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_model(trainloader, testloader, model, criterion, optimizer, epochs=2)

#test mutate
# mutated_architecture = search_space.mutate(random_architecture, mutation_prob=0.1, operation_mutation_prob=0.5, node_deletion_prob=0.2)
# Visualiser l'architecture générée
# search_space.visualize_architecture(mutated_architecture)


#***********************D'autres config examples

# config_yaml = """
# primitive_operations:
#   - AvgPool2d-kernel_size:2x2
#   - MaxPool2d-kernel_size:3x3
#   - SeparableConv2d-kernel_size:3x3
#   - Conv2d-kernel_size:5x5&padding:1
#   - Conv2d-kernel_size:3x3&padding:1
#   - Bottleneck
#   - AvgPool2d-kernel_size:2x2
#   - MaxPool2d-kernel_size:3x3
#   - Conv2d-kernel_size:5x5&padding:1
#   - Conv2d-kernel_size:3x3&padding:1
#   - MBConv

# chain_size:
#   - 7
#   - 8

# other_configs:
#   out_channels:
#     - 32
#     - 64
#   kernel_size:
#     - 3
#     - 5 
#   expansion_ratio:
#     - 1
#     - 3
#     - 6

# node_operations:
#   - node_id: 1
#     allowed_operations:
#         - Conv2d-kernel_size:3x3
#   - node_id: 2
#     allowed_operations:
#         - Bottleneck
#   - node_id: 3
#     combine_op: sum_features_maps
#   - node_id: 4
#     similar_to: 1
#   - node_id: max
#     allowed_operations:
#         - Conv2d-kernel_size:1x1
#   - node_id : end
#     combine_op: concat_features_maps

# edge_operations:
#   - from: 1
#     to: 3
#     allowed_operations:
#       - Identity
#   - from: 1
#     to: 2
#     allowed_operations:
#       - Identity
#   - from: 2
#     to: 3
#     allowed_operations:
#       - Identity
# """


# config_yaml = """
# primitive_operations:
#   - AvgPool2d-kernel_size:2x2
#   - MaxPool2d-kernel_size:3x3
#   - SeparableConv2d-kernel_size:3x3
#   - Conv2d-kernel_size:5x5
#   - Conv2d-kernel_size:5x5
#   - ConvTranspose2d-kernel_size:3x3
#   - Identity
#   - ReLU
#   - BatchNorm2d
#   - Bottleneck
#   - DilConvDARTS 


# chain_size:
#   - 6
#   - 9

# other_configs:
#   out_channels:
#     - 32
#     - 64
#   kernel_size:
#     - 2
#     - 3
#   stride:
#     - 1
#     - 2
#   expansion_ratio:
#     - 1
#     - 3
#     - 6

# # node_operations:
# #   - node_id: 1
# #     allowed_operations:
# #         - StemNASLib
# #   - node_id: 2
# #     allowed_operations:
# #         - DilConvDARTS
# #   - node_id: 3
# #     allowed_operations:
# #         - ConvBNReLU
# #   - node_id: 4
# #     allowed_operations:
# #         - DepthwiseConv
# #   - node_id: 4
# #     allowed_operations:
# #         - SepConvDARTS
# #   - node_id: max - 1
# #     allowed_operations:
# #         - AvgPool2d-kernel_size:2x2
# # """





# config_yaml = """
# primitive_operations: #il specfier ici les operations qui veut autorisé et peut auusi pour opeation secfier ces paramtres
#   - Conv2d-kernel_size:5x5 #this is pytorch primitive op
#   - SeparableConv2d-kernel_size:3x3&out_channels:32 #this is block d'operation definie par user avec ces paramtres
#   - DilConvDARTS #this is personlied block d'operations

# chain_size: [#min, #max]

# other_configs:
#   out_channels: #list of psible valeur
#     - 32
#     - 64
#   oher_paramtres: ..
# node_operations:
#   - node_id: 
#     allowed_operations:
#         - operation
#     OR
#     similar_to: node id 

#     combine_op: type of concat operation: for eampleconcat_features_maps

# edge_operations:
#   - from: surce id
#     to: resceiveur id