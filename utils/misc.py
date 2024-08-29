import json
import torch
import yaml
import random
import wandb
import numpy as np
import open3d as o3d
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import Tuple

# custom function import
from pointnet2_ops import pointnet2_utils

def set_random_seed(logger: str, seed: int=42, deterministic=False) -> None:
    """
    Set a random seed for reproducability.

    :param seed: The seed to be used.
    :param deterministic: Whether to set the deterministic option for CUDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Random seed set to: {seed},'
                    f'Deterministic: {deterministic}')

    
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def fps(data: torch.Tensor, npoint: int):
    fps_idx = pointnet2_utils.furthest_point_sample(data, npoint)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

           
def save_point_cloud(np_array: np.ndarray, file_path: Path):
    """
    Save a numpy array as a point cloud file.

    :param np_array: NumPy array representing point cloud data.
    :param file_path: File path to save the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    o3d.io.write_point_cloud(str(file_path), pcd)
 

def abs_cum_distance_in_patch(patch: np.ndarray) -> float:
    """
    Calculate the absolute cumulative distance between each combination of points in the patch using vectorized operations.

    :param patch: A numpy array of shape (group_size, 3), representing "group_size" points in 3D space (x, y, z).
    :return: The sum of absolute distances between each pair of points.
    """
    # Subtract every point from every other point (broadcasting)
    diff = patch[:, np.newaxis, :] - patch[np.newaxis, :, :]
    
    # Calculate the squared distance for each pair
    squared_dist = np.sum(diff**2, axis=-1)
    
    # Take the square root to get the actual distances and sum them up
    return np.sum(np.sqrt(squared_dist[np.triu_indices_from(squared_dist, k=1)]))


def modified_coefficient_of_variation(predicted_point_patches: np.ndarray) -> float:
    """
    Calculate a modified coefficient of variation (M-CV) for a list of distances.
    M-CV is high when all points differ a lot from one another and 0 when all points are close to or identical.

    :param predicted_point_patches: all point patches predicted by the model
    :return: The modified coefficient of variation.
    """
    distances = [abs_cum_distance_in_patch(patch) for patch in predicted_point_patches]
    if not distances:
        return 0
    
    distances_array = np.array(distances)
    mean = np.mean(distances_array)
    sd = np.std(distances_array)

    if mean == 0:
        return 0 if sd == 0 else 1  # Return 0 if all values are 0, else 1

    # Normalizing by the mean, adjusting for cases where the mean is very small
    cv = sd / mean
    return cv



def to_cpu(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """
    Move a list of tensors to CPU and convert them to NumPy arrays.

    :param tensors: Variable number of PyTorch tensors.
    :return: Tuple of NumPy arrays corresponding to the input tensors.
    """
    return tuple(t.detach().cpu().numpy() for t in tensors)


def save_as_json(data, file_path: Path):
    """
    Save data to a JSON file.

    :param data: Data to save.
    :param file_path: Path to the JSON file.
    """
    with file_path.open('w') as f:
        json.dump(data, f)


def save_graph(nodes: np.array, hierarchy_edges: np.array, path: Path) -> None:
    """
    Create a folder for the graph then save the entire graph in two seperate files: 
    nodes.npy, hierarchy_edges.npy.
    
    :param nodes: nodes
    :param hierarchy_edges: hierarchy edges
    param path: the path to the folder
    """
    # create the folder
    path.mkdir(parents=True, exist_ok=True)
    # save the nodes and edges each in a seperate file
    np.save(path / 'nodes.npy', nodes)
    np.save(path / 'hierarchy_edges.npy', hierarchy_edges)

def load_default_graph(path: Path):
    """
    Load a yaml file from a given directory
    :param file: the file path
    :return: yaml file 
    """
    # Load the YAML file
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def add_node_feature_partnet(nodes, label, feature, base_type: int = None): 
    """
    Adds a feature to a specific node identified by its label in a graph. If the node already 
    contains a feature, this function duplicates the node and adds the new feature to the duplicate,
    preserving the original node's feature.

    :param graph: A dictionary representing a graph, whith nodes and edges. 
    :param node_label: The label of the node to which the feature should be added or duplicated.
    :param feature: patch encoding from PointMAE encoder. 
    :return: The updated graph dictionary with the feature added or node duplicated.
    """    
    shared_leaf_nodes = [1, 7, 8, 11] 
    
    base_types = {
            "regular_leg_base": 28,
            "drawer_base": 26,
            "pedestal_base": 25,
            "star_leg_base": 29
    }

    base_type = base_types[base_type]


    # Iterate through the nodes
    for node in nodes:
        if node['label'] == label:
            # find the correct node among duplicate nodes. 
            if label in shared_leaf_nodes and node['parent_node'] == base_type:
                new_node = node.copy()
                node['parent_node'] = base_type
                new_node['feature'] = feature
                nodes.append(new_node)
                break # added feature
            else:
                new_node = node.copy()
                new_node['feature'] = feature
                nodes.append(new_node)
                break # added feature
    return nodes

def add_node_feature_ceasar(nodes, label, feature): 
    """
    Adds a feature to a specific node identified by its label in a graph. If the node already 
    contains a feature, this function duplicates the node and adds the new feature to the duplicate,
    preserving the original node's feature.

    :param graph: A dictionary representing a graph, whith nodes and edges. 
    :param node_label: The label of the node to which the feature should be added or duplicated.
    :param feature: patch encoding from PointMAE encoder. 
    :return: The updated graph dictionary with the feature added or node duplicated.
    """    
    # Iterate through the nodes
    for node in nodes:
        if node['label'] == label:
            new_node = node.copy()
            new_node['feature'] = feature
            nodes.append(new_node)
            break # added feature
    return nodes


def nodes_to_numpy(nodes, feature_dim: int, cfg=None):
    """
    Converts a list of node dictionaries to a structured numpy array, updating the feature of the node with label 8 to be the average of non-zero features.

    :param nodes: List of node dictionaries.
    :param feature_dim: Dimension of the feature vectors.
    :return: Structured numpy array of nodes.
    """
    type_to_int = {"part": 0, "sub_structure": 1, "object": 2}
    nodes_dtype = [('feature', np.float32, (feature_dim,)), ('label', np.float32), ('parent_node', np.float32), ('center', np.float32, (3,)), ('pos_embedding', np.float32, (feature_dim,))]
    
    # Initialize an empty list to store node data
    new_nodes = []
    
    # Store features separately for calculating the average
    features = []
    pos_embeddings = []

    # Loop through each node in the input
    for node in nodes:
        feature = node['feature']
        pos_embedding = np.zeros(feature_dim, dtype=np.float32) if node['pos_embedding'] is None else np.array(node['pos_embedding'][0])
        if np.any(feature):  # add to list to calculate avg for master node
            features.append(feature)  # Only append non-zero features
            pos_embeddings.append(pos_embedding) # Only append non-zero embeddings
        new_node = (feature, node['label'], node['parent_node'], np.zeros(3, dtype=np.float32), pos_embedding)  # Assume center is zero if not provided
        new_nodes.append(new_node)

    # Convert list of nodes to a structured numpy array
    structured_nodes = np.array(new_nodes, dtype=nodes_dtype)

     # TODO: old code, does not apply: Optionally remove nodes with all-zero features if the configuration specifies
    if not cfg.use_default_graph:
        ValueError("setting use_default_graph to False is not allowed!")
        structured_nodes = np.array([node for node in structured_nodes if np.any(node['feature'])], dtype=nodes_dtype)

    return structured_nodes

def nodes_to_numpy_ceasar(nodes, feature_dim: int, cfg=None):
    """
    Converts a list of node dictionaries to a structured numpy array, updating the feature of the node with label 8 to be the average of non-zero features.

    :param nodes: List of node dictionaries.
    :param feature_dim: Dimension of the feature vectors.
    :return: Structured numpy array of nodes.
    """
    nodes_dtype = [('feature', np.float32, (feature_dim,)), ('label', np.float32), ('parent_node', np.float32)]
    
    # Initialize an empty list to store node data
    new_nodes = []
    
    # Store features separately for calculating the average
    features = []

    # Loop through each node in the input
    for node in nodes:
        feature = node['feature']
        if np.any(feature):  # add to list to calculate avg for master node
            features.append(feature)  # Only append non-zero features
        new_node = (feature, node['label'], node['parent_node'])  # Assume center is zero if not provided
        new_nodes.append(new_node)

    # Convert list of nodes to a structured numpy array
    structured_nodes = np.array(new_nodes, dtype=nodes_dtype)

     # TODO: old code, does not apply: Optionally remove nodes with all-zero features if the configuration specifies
    if not cfg.use_default_graph:
        ValueError("setting use_default_graph to False is not allowed!")
        structured_nodes = np.array([node for node in structured_nodes if np.any(node['feature'])], dtype=nodes_dtype)

    return structured_nodes


def generate_edges(nodes) -> np.ndarray:
    """
    Generate an edge index list based on the parent_node value of each node.
    This function makes sure to create one edge for each leaf node, resolving the case where some nodes (1, 7, 8, 11) are connected to multiple parents.

    :param nodes: A list of nodes, each node is a dictionary with 'feature', 'label', 'parent_node'.
    :param edge_attribute: A boolean flag to decide whether to include edge attributes directly in the edge tuples.
    :return: An array of tuples, where each tuple (i, j) represents an edge from node i to node j.
    """
    
    # Create a mapping from node identifiers to their indices
    node_index_mapping = {node['label']: idx for idx, node in enumerate(nodes)}
    
    edge_index = []
    for idx, node in enumerate(nodes):
        parent_node = node['parent_node']
        if parent_node is not None:
            parent_index = node_index_mapping.get(parent_node)
            if parent_index is not None:
                edge_index.append((parent_index, idx))

    return np.array(edge_index).T
        

### Dummy Classifier ###


def run_dummy_classifier(data_module, cfg):
    epochs = cfg.epochs
    if getattr(cfg, 'dummy_classifier', False) and getattr(cfg.model, 'noise_percentage', 'None') != 'None':
        raise ValueError("When running the dummy classifier, you need to set noise_percentage to None otherwise the results will not show up correctly in wandb.")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    X_train, y_train = [], []
    for batch in train_loader:
        X_train.append(batch.x.numpy())
        y_train.append(batch.y.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_val, y_val = [], []
    for batch in val_loader:
        X_val.append(batch.x.numpy())
        y_val.append(batch.y.numpy())
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train)

    y_pred = dummy_clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    #Connot set baseline in WandB so we simulate multiple epochs to make it show up as we want
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}, Dummy Classifier Accuracy: {acc}")
        wandb.log({'epoch': epoch + 1, 'Validation Accuracy': acc})




