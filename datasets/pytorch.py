import numpy as np
from pathlib import Path

import json
import logging
import torch
import torch.utils.data as data

from torch_geometric.data import HeteroData, Data
from torch_geometric.data import Dataset as GeomDataset

from .io import IO
from .build import DATASETS
from utils.logger import print_log


@DATASETS.register_module()
class PartNet(data.Dataset):
    """
    Dataset class for PartNet data. Inherits from PyTorch's Dataset class.
    Map-style datasets

    :param config: Configuration object with dataset parameters.
    """

    def __init__(self, cfg):
        self.data_root = Path(cfg.DATA_PATH)
        self.labels = cfg.LABELS
        self.base_type = cfg.BASE_TYPE
        self.dir_list = [d for d in self.data_root.iterdir() if d.is_dir()]
        print_log(f'[DATASET] {len(self.dir_list)} instances were loaded', logger='PartNet')
        # get some statistics about the dataset:
        self.available_samples = self.__len__()

        # TODO: Aangepast op 5 jan , dit is de origine PointMae na te doen
        self.N_POINTS = cfg.N_POINTS
        self.sample_points_num = cfg.npoints
        self.permutation = np.arange(self.N_POINTS)

    def pc_norm(self, pc):
        """
        Normalize a point cloud. 

        :param pc: Input point cloud.
        :return: Normalized point cloud.
        """

        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        """
        Randomly sample points from the point cloud along with indices.

        :param pc: Input point cloud.
        :param num: Number of points to sample.
        :return: Tuple of sampled points and indices.
        """
        np.random.shuffle(self.permutation)
        indices = self.permutation[:num]
        sampled_pc = pc[indices]
        return sampled_pc, indices

    def farthest_point_sample(self, xyz: np.ndarray) -> np.ndarray:
        """
        Efficiently sample points from a point cloud using the farthest point sampling algorithm.

        :param xyz: Pointcloud data, [N, D].
        :param sample_points_num: Number of samples.
        :return: Sampled pointcloud index, [sample_points_num, D].
        """
        N, D = xyz.shape
        centroids = np.zeros(self.sample_points_num, dtype=np.int64)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(self.sample_points_num):
            centroids[i] = farthest
            centroid = xyz[farthest, :3]
            dist = np.sum((xyz[:, :3] - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return xyz[centroids]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset. 
        Error handling is done by IO

        :param idx: Index of the sample.
        :return: Sampled data.
        """
        sample_dir = self.data_root / self.dir_list[idx]
        # assuming the point cloud has the same name in each dir
        pcd_file = sample_dir / 'point_sample' / 'ply-10000.ply'

        if self.labels:
            labels_file = sample_dir / 'point_sample' / 'unique_labels.csv'
            pcd = IO.get(pcd_file).astype(np.float32)
            labels = IO.get(labels_file)

            # Sub sample both pointcloud and labels
            pcd, indices = self.random_sample(pcd, self.sample_points_num)
            labels = labels[indices]  # Reorder labels

            pcd = self.pc_norm(pcd)
            pcd = torch.from_numpy(pcd).float()
            labels = torch.tensor(labels, dtype=torch.long)

            # to construct the default Table graph, we need to know the base type of each sample. 
            if self.base_type:
                json_file = sample_dir / 'result.json'
                data = IO.get(json_file)
                base_type = self.get_base_type(data)
                return pcd, labels, base_type
            else:
                return pcd, labels
        else:
            pcd = IO.get(pcd_file).astype(np.float32)
            pcd, _ = self.random_sample(pcd, self.sample_points_num)
            # data = self.farthest_point_sample(data)
            pcd = self.pc_norm(pcd)
            pcd = torch.from_numpy(pcd).float()

            return pcd

    def get_base_type(self, data):
        """
        Retrieve the base type from results.json.

        :param data: JSON data.
        :return: Base type if found, otherwise None.
        """
        search_values = [
            "regular_leg_base",
            "drawer_base",
            "pedestal_base",
            "star_leg_base"
        ]

        def search_json(data):
            if isinstance(data, dict):
                # Check if the 'name' key contains one of the base types
                if 'name' in data and data['name'] in search_values:
                    return data['name']
                # Recurse into children if they exist
                if 'children' in data:
                    for child in data['children']:
                        result = search_json(child)
                        if result:
                            return result
            elif isinstance(data, list):
                for item in data:
                    result = search_json(item)
                    if result:
                        return result
            return None

        return search_json(data)

    def __len__(self):
        return len(self.dir_list)


@DATASETS.register_module()
class Ceasar(data.Dataset):
    """
    Dataset class for Ceasar data. Inherits from PyTorch's Dataset class.
    Map-style datasets

    :param config: Configuration object with dataset parameters.
    """

    def __init__(self, cfg):
        self.data_root = Path(cfg.DATA_PATH)
        self.labels = cfg.LABELS
        self.base_type = cfg.BASE_TYPE
        self.dir_list = [d for d in self.data_root.iterdir() if d.is_dir()]
        print_log(f'[DATASET] {len(self.dir_list)} instances were loaded', logger='Ceasar')
        # get some statistics about the dataset:
        self.available_samples = self.__len__()

        # TODO: Aangepast op 5 jan , dit is de origine PointMae na te doen
        self.N_POINTS = cfg.N_POINTS
        self.sample_points_num = cfg.npoints
        self.permutation = np.arange(self.N_POINTS)

    def pc_norm(self, pc):
        """
        Normalize a point cloud. 

        :param pc: Input point cloud.
        :return: Normalized point cloud.
        """

        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        """
        Randomly sample points from the point cloud along with indices.

        :param pc: Input point cloud.
        :param num: Number of points to sample.
        :return: Tuple of sampled points and indices.
        """
        np.random.shuffle(self.permutation)
        indices = self.permutation[:num]
        sampled_pc = pc[indices]
        return sampled_pc, indices

    def farthest_point_sample(self, xyz: np.ndarray) -> np.ndarray:
        """
        Efficiently sample points from a point cloud using the farthest point sampling algorithm.

        :param xyz: Pointcloud data, [N, D].
        :param sample_points_num: Number of samples.
        :return: Sampled pointcloud index, [sample_points_num, D].
        """
        N, D = xyz.shape
        centroids = np.zeros(self.sample_points_num, dtype=np.int64)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(self.sample_points_num):
            centroids[i] = farthest
            centroid = xyz[farthest, :3]
            dist = np.sum((xyz[:, :3] - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return xyz[centroids]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset. 
        Error handling is done by IO

        :param idx: Index of the sample.
        :return: Sampled data.
        """
        sample_dir = self.data_root / self.dir_list[idx]
        # assuming the point cloud has the same name in each dir
        pcd_file = sample_dir / 'point_cloud.ply'

        if self.labels:
            labels_file = sample_dir / 'labels.csv'
            pcd = IO.get(pcd_file).astype(np.float32)
            labels = IO.get(labels_file)

            # Sub sample both pointcloud and labels
            pcd, indices = self.random_sample(pcd, self.sample_points_num)
            labels = labels[indices]  # Reorder labels

            pcd = self.pc_norm(pcd)
            pcd = torch.from_numpy(pcd).float()
            labels = torch.tensor(labels, dtype=torch.long)

            return pcd, labels

        else:
            pcd = IO.get(pcd_file).astype(np.float32)
            pcd, _ = self.random_sample(pcd, self.sample_points_num)
            # data = self.farthest_point_sample(data)
            pcd = self.pc_norm(pcd)
            pcd = torch.from_numpy(pcd).float()

            return pcd

    def __len__(self):
        return len(self.dir_list)


@DATASETS.register_module()
class GraphDataset(GeomDataset):
    """
    Dataset class for Graph data, builds a heterogenious graph dataset.
    Inherits from PyTorchGeomtric's Dataset class, map-style datasets.

    :param config: Configuration object with dataset parameters.
    """

    def __init__(self, cfg):
        # TODO check if required:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg['DATA_PATH'])
        self.samples = [d for d in self.data_root.iterdir() if d.is_dir()]
        print_log(f'[DATASET] {len(self.samples)} instances were loaded', logger='GraphData')

    def __getitem__(self, idx: int) -> Data:
        """
        Get a sample from the dataset.

        :param idx: Index of the sample.
        :return: Sampled data.
        """
        sample_path = self.data_root / self.samples[idx]
        nodes = IO.get(sample_path / 'nodes.npy')
        features = torch.from_numpy(nodes['feature'])
        labels = torch.from_numpy(nodes['label'])

        if getattr(self.cfg, 'return_raw_data', False):
            # I use this to create and save the k-folds 
            edges = IO.get(sample_path / 'hierarchy_edges.npy')
            data = {'nodes': nodes, 'edges': edges}
            return data

        if self.cfg.use_hierarchy_edges:
            edges = IO.get(sample_path / 'hierarchy_edges.npy')
            edge_indices = torch.from_numpy(edges).to(torch.long)
            edge_indices = torch.from_numpy(edges[0:2]).to(torch.long)
        else:
            ValueError("No or incorrect edge type defined in config")

        # Create a Data object which is used for homogeneous graphs in PyG
        graph = Data(x=features, y=labels, edge_index=edge_indices)
        return graph

    def __len__(self):
        return len(self.samples)


@DATASETS.register_module()
class PartNetEmbeddings(GeomDataset):
    """
    Data loader for the raw embeddings created by the PointMAE encoder.
    This class is meant to be used for the simple MLP that tries to directly predict the labels from the embeddings. 

    :param config: Configuration object with dataset parameters.
    """

    def __init__(self, cfg):
        # TODO check if required:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg['DATA_PATH'])
        self.samples = [d for d in self.data_root.iterdir() if d.is_dir()]
        print_log(f'[DATASET] {len(self.samples)} instances were loaded', logger='PartNetEmbeddings')

    def __getitem__(self, idx: int) -> Data:
        """
        Get a sample from the dataset.

        :param idx: Index of the sample.
        :return: Sampled data.
        """
        sample_path = self.data_root / self.samples[idx]
        embeddings = IO.get(sample_path / 'embeddings.npy')
        labels = IO.get(sample_path / 'labels.npy')

        # Ensure consistent shuffling
        shuffled_embeddings, shuffled_labels = self._shuffle_embeddings_labels(embeddings, labels)

        return shuffled_embeddings, shuffled_labels

    def _shuffle_embeddings_labels(self, embeddings: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Shuffle embeddings and labels consistently.

        :param embeddings: Embeddings array.
        :param labels: Labels array.
        :return: Shuffled embeddings and labels.
        """
        assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have the same number of samples."
        indices = np.arange(embeddings.shape[0])
        np.random.shuffle(indices)
        return embeddings[indices], labels[indices]

    def __len__(self):
        return len(self.samples)
