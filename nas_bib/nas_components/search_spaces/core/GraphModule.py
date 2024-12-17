import torch
import torch.nn as nn
import networkx as nx
import json

from nas_bib.nas_components.search_spaces.core.operations import sum_features_maps, concat_features_maps

class GraphModule(nx.DiGraph, nn.Module):
    def __init__(self, name=None):
        super(GraphModule, self).__init__()
        nn.Module.__init__(self)
        self.name = name

    def parse(self):
        for node_idx in nx.lexicographical_topological_sort(self):
            if "operation" in self.nodes[node_idx]:
                operation = self.nodes[node_idx]["operation"]
                if isinstance(operation, GraphModule):
                    operation.parse()
                    self.add_module(
                        "{}-node-subgraph(motif)_at({})".format(self.name, node_idx),
                        operation,
                    )
                elif isinstance(operation, torch.nn.Module):
                    self.add_module(
                        "{}-node-module_at({})".format(self.name, node_idx),
                        operation,
                    )
            for successor_idx in self.successors(node_idx):
                edge_data = self.get_edge_data(node_idx, successor_idx)
                if edge_data and "operation" in edge_data:
                    edge_op = edge_data["operation"]
                    if edge_op is not None:
                        if isinstance(edge_op, GraphModule):
                            edge_op.parse()
                            self.add_module(
                                "{}-edge-subgraph(motif)_at({},{})".format(self.name, node_idx, successor_idx),
                                edge_op,
                            )
                        elif isinstance(edge_op, torch.nn.Module):
                            self.add_module(
                                "{}-edge-module_at({},{})".format(self.name, node_idx, successor_idx),
                                edge_op,
                            )

    def unparse(self):
        g = self.__class__()
        g.clear()

        graph_nodes = self.nodes(data=True)
        graph_edges = self.edges(data=True)

        for node_idx, data in graph_nodes:
            if "operation" in data:
                if isinstance(data["operation"], GraphModule):
                    data["operation"] = data["operation"].unparse()
        for u, v, data in graph_edges:
            if "operation" in data:
                if isinstance(data["operation"], GraphModule):
                    data["operation"] = data["operation"].unparse()

        g.add_nodes_from(graph_nodes)
        g.add_edges_from(graph_edges)
        g.graph.update(self.graph)
        g.name = self.name

        return g

    def forward(self, X):
        node_input = {}
        for node in nx.topological_sort(self):
            if len(list(self.predecessors(node))) == 0:
                node_input[node] = X
            else:
                inputs = []
                for predecessor_idx in self.predecessors(node):
                    edge_data = self.get_edge_data(predecessor_idx, node)
                    if edge_data is not None:
                        edge_operation = edge_data.get('operation', nn.Identity())
                    else:
                        edge_operation = nn.Identity()
                    if isinstance(edge_operation, GraphModule):
                        inputs.append(edge_operation.forward(node_input[predecessor_idx]))
                    else:
                        inputs.append(edge_operation(node_input[predecessor_idx]))
                if len(inputs) > 1:
                    comb_op = self.nodes[node].get('combine_op', sum_features_maps())
                    node_input[node] = comb_op(inputs)
                else:
                    node_input[node] = inputs[0]

            if 'operation' in self.nodes[node]:
                operation = self.nodes[node]['operation']
                if isinstance(operation, GraphModule):
                    node_input[node] = operation.forward(node_input[node])
                else:
                    node_input[node] = operation(node_input[node])

        output_node = list(self.nodes)[-1]
        output = node_input[output_node]
        return output

    def display_graph_details(self, level=1):
      for node in self.nodes(data=True):
          node_id, node_data = node
          print(f"{'  ' * level}Node {node_id} details:")
          print(f"{'  ' * level}{node_data}")

          out_edges = list(self.out_edges(node_id))
          if out_edges:
              print(f"{'  ' * (level)}Incoming Edges details for Node {node_id}:")
              for source, target in out_edges:
                  edge_data = self.get_edge_data(source, target)
                  print(f"{'  ' * (level+1)}From Node {source} to Node {target}: {edge_data}")

          if "operation" in node_data and isinstance(node_data["operation"], GraphModule):
              print(f"{'  ' * level}Subgraph operation details:")
              subgraph = node_data["operation"]
              subgraph.display_graph_details(level = level * 2)

    def graphmodule_to_json(self, model_id={}, metrics = {}):
        graph_data = {
            "id": model_id,
            "metrics": metrics,
            "graph": {
                "name": self.name,
                "nodes": {},
                "edges": []
            }
        }

        for node_idx, node_data in self.nodes(data=True):
            node_info = {}
            if "operation" in node_data:
                operation = node_data["operation"]
                if isinstance(operation, GraphModule):
                    node_info["operation"] = (operation.graphmodule_to_json())["graph"]
                elif isinstance(operation, nn.Module):
                    node_info["operation"] = self.module_to_dict(operation)

            if "combine_op" in node_data:
                node_info["combine_op"] = str(node_data["combine_op"])
            graph_data["graph"]["nodes"][str(node_idx)] = node_info


        for source, target, edge_data in self.edges(data=True):
            edge_info = {}
            if "operation" in edge_data:
                operation = edge_data["operation"]

                if isinstance(operation, GraphModule):
                    edge_info= (operation.graphmodule_to_json())["graph"]

                elif isinstance(operation, nn.Module):
                    edge_info = self.module_to_dict(operation)

            graph_data["graph"]["edges"].append({
                "source": str(source),
                "target": str(target),
                "operation": edge_info
            })


        return graph_data

    def write_json_to_file(self, json_data, filename):
        with open(filename, "a") as f:
            json.dump(json_data, f, indent=2)
            f.write("\n")

    def tensor_to_list(self, tensor):
        return tensor.tolist() if isinstance(tensor, torch.Tensor) else None

    def module_to_dict(self, module):
        module_dict = {
            "module_name": module.__class__.__name__,
            "attributes": {},
            "parameters": {}
        }

        # Vérifier si le module est un module de base ou un module composé
        if isinstance(module, nn.Module):
            # Si c'est un module de base
            if not isinstance(module, nn.Sequential):
                # Récupérer les attributs du module de base
                for attr_name, attr_value in module.__dict__.items():
                    if not attr_name.startswith('_') and not callable(attr_value):
                        module_dict["attributes"][attr_name] = attr_value

                # Récupérer les paramètres du module de base
                for name, param in module.named_parameters():
                    module_dict["parameters"][name] = param.tolist()
            else:
                # Si c'est un module composé
                # Décomposer le module séquentiel
                module_dict["submodules"] = []
                for sub_module in module:
                    module_dict["submodules"].append(self.module_to_dict(sub_module))

        return module_dict

    def print_recursive(self, indent=0):
        for node_idx in nx.lexicographical_topological_sort(self):
            node_data = self.nodes[node_idx]
            print("       " * indent, f"Node {node_idx}: {node_data}", end="\n")

            if isinstance(node_data.get('operation'), GraphModule):
                print("       " * indent, f"GraphModule found at Node {node_idx}.")
                node_data['operation'].print_recursive(indent + 1)

            out_edges = list(self.out_edges(node_idx, data=True))
            for u, v, data in out_edges:
                print("       " * indent, f"{u} -> {v}, et data = {data}", end="\n")
                if isinstance(data.get('operation'), GraphModule):
                    data['operation'].print_recursive(indent + 1)




# # Exemple d'utilisation :
# graph1 = GraphModule(name="TestGraph1")
# # Ajouter des nœuds et des arêtes au graphique

# # Ajout de quelques opérations au graphe
# graph1.add_node(1, operation=torch.nn.Conv2d(3, 6, 3))
# graph1.add_node(2, operation=torch.nn.MaxPool2d(2, 2))
# subgraph = GraphModule(name="SousGraphe1")
# subgraph.add_node(1, operation=torch.nn.MaxPool2d(2, 2))
# subgraph.add_node(2, operation=torch.nn.MaxPool2d(2, 2))
# subgraph.add_edge(1, 2, operation=torch.nn.BatchNorm1d(5))
# graph1.add_node(3, operation=subgraph)
# graph1.add_edge(1, 2, operation=torch.nn.BatchNorm1d(5))
# graph1.add_edge(2, 3, operation=torch.nn.BatchNorm1d(5))

# # Transformer le modèle en format JSON
# json_data1 = graph1.graphmodule_to_json( model_id=1, metrics={"accuracy": 0.95, "loss": 0.1})
# print("******************json_data1**************")
# print(json_data1)

# # Écrire les données JSON dans un fichier
# graph1.write_json_to_file(json_data1, filename="models.json")

# graph2 = GraphModule(name="TestGraph2")
# # Ajouter des nœuds et des arêtes au graphique

# # Ajout de quelques opérations au graphe
# graph2.add_node(1, operation=torch.nn.Conv2d(16, 32, 3))
# graph2.add_node(2, operation=nn.Sequential(
#   nn.Linear(32, 10),
#   nn.ReLU(),
#   nn.Linear(10, 10),
#   nn.ReLU(),
#   nn.Linear(10, 10),
#   nn.Softmax(dim=1)
# )
#                 )
# subgraph = GraphModule(name="SousGraphe2")
# subgraph.add_node(1, operation=torch.nn.MaxPool2d(2, 2))
# subgraph.add_node(2, operation=torch.nn.MaxPool2d(2, 2))
# subgraph.add_edge(1, 2, operation=torch.nn.BatchNorm1d(5))
# graph2.add_node(3, operation=subgraph)
# graph2.add_edge(1, 2, operation=torch.nn.BatchNorm1d(5))
# graph2.add_edge(2, 3, operation=torch.nn.BatchNorm1d(5))

# # Transformer le modèle en format JSON
# json_data2 = graph2.graphmodule_to_json( model_id=2, metrics={"accuracy": 0.85, "loss": 0.2})

# # Écrire les données JSON dans un fichier
# graph2.write_json_to_file(json_data2, filename="models.json")