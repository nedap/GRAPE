import h5py
import logging
import numpy as np
import open3d as o3d
import pandas as pd
import pickle
import json

from pathlib import Path
from typing import Dict, Any

class IO:
    """
    Serves as a utility for reading data from files with different extensions:
    Supported extensions: .npy, .h5, .ply
    """

    @classmethod
    def get(cls, file_path: Path):
        file_extension = file_path.suffix

        try:
            if file_extension in ['.npy']:
                return cls._read_npy(file_path)
            elif file_extension in ['.h5']:
                return cls._read_h5(file_path)
            elif file_extension in ['.ply']:
                return cls._read_ply(file_path)
            elif file_extension in ['.csv']:
                return cls._read_csv(file_path)
            elif file_extension in ['.pickle']:
                return cls._read_pickle(file_path)
            elif file_extension in ['.json']:
                return cls._read_json(file_path)
            else:
                supported_extensions = ['.npy', '.h5', '.ply', '.csv', '.pickle', 'json']
                raise Exception(f'Unsupported file extension {file_extension}.'
                                f'Supported exstesions: {supported_extensions}')
            
        except Exception as e:
            raise logging.error(f'Error occurred while reading {file_path} file: {e}')

        
    @classmethod
    def _read_npy(cls, file_path: Path) -> np.ndarray:
        """
        Reads a .npy file.

        :param file_path: Path to the .npy file.
        :return: Data as a NumPy array.
        """ 
        try:
            return np.load(file_path)
        except Exception as e:
            raise logging.error(f'Error while reading .npy file {e}')
    
    @classmethod
    def _read_h5(cls, file_path: Path, dataset: str = 'data') -> np.ndarray:
        """
        Reads a dataset from an HDF5 file.

        :param file_path: Path to the HDF5 file.
        :param dataset: Name of the dataset to be read from the file. Defaults to 'data'.
        :return: Data from the specified dataset as a NumPy array.
        """        
        try:
            with h5py.File(file_path, 'r') as f:
                if dataset in f:
                    return f[dataset][()]
                else:
                    raise KeyError(f'Dataset {dataset} not in the file')
        except Exception as e:
            raise logging.error(f'Error while reading .h5 file: {e}')

    @classmethod
    def _read_ply(cls, file_path: Path) -> np.ndarray:
        """
        reads a .ply file using open3d

        :param file_path: Path to the .ply file.
        :return: Data as open3D
        """
        try:
            return np.asarray(o3d.io.read_point_cloud(str(file_path)).points)
        except Exception as e:
            raise logging.error(f'Error while reading .ply file: {e}')
        
    @classmethod
    def _read_csv(cls, file_path: Path, cols=['Integer Label']) -> np.ndarray:
        """
        reads a .csv file using Pandas

        :param file_path: Path to the .ply file.
        :return: Data as np.array
        """
        try:
            df = pd.read_csv(str(file_path), usecols=cols)
            return df['Integer Label'].to_numpy()
        except Exception as e:
            raise logging.error(f'Error while reading .csv file: {e}')
        
    @classmethod
    def _read_pickle(cls, file_path: Path) -> Dict[str, Any]:
        """
        reads a .pickle file

        :param file_path: Path to the .ply file.
        :return: dict 
        """
        try:
            with open(file_path, 'rb') as f:
                dictionary = pickle.load(f)
            return dictionary
        except Exception as e:
            raise logging.error(f'Error while reading .pickle file: {e}')
        
    @classmethod
    def _read_json(cls, file_path: Path) -> Dict[str, Any]:
        """
        Reads a .json file.

        :param file_path: Path to the .json file.
        :return: Data as a dictionary.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise logging.error(f'Error while reading .json file: {e}')
        
        
