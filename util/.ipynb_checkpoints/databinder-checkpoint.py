import torch
import json
import os
import logging
from matplotlib.figure import Figure
from datetime import datetime


logging.basicConfig(level=logging.INFO)

class DataBinder:
    """
    A class used to bind data to a specific directory.

    ...

    Attributes
    ----------
    dir : str
        a formatted string to print out the directory where the data will be saved
    info : dict
        a dictionary to save the information of the data

    Methods
    -------
    __save():
        Saves the current state of the info dictionary into a json file in the directory.
    __load():
        Loads the info dictionary from the json file in the directory.
    add(key:str, value):
        Adds a new key-value pair to the info dictionary.
    remove(key:str):
        Removes an existing key-value pair from the info dictionary.
    get(key:str):
        Gets the value of a specific key in the info dictionary.
    keys():
        Returns all keys in the info dictionary.
    """

    def __init__(self, target_dir:str):
        """
        Constructs all the necessary attributes for the data binder object.

        Parameters
        ----------
            target_dir : str
                directory where the data will be saved
        """

        self.dir = target_dir
            
        if not os.path.exists(target_dir):
            os.mkdir(self.dir)
            logging.info(f'Create {target_dir}')
            self.info = {}
        elif not os.path.exists(f'{target_dir}/info.json'):
            self.info = {}
            logging.info(f'Create {target_dir}/info.json')
            logging.warning(f'{target_dir} may not be for Data Binder.')
        else:
            self.info = self.__load()
            
    def __save(self):
        """
        Saves the current state of the info dictionary into a json file in the directory.
        """

        saved_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.info['saved_datetime'] = saved_datetime
        
        with open(f'{self.dir}/info.json', 'w') as fout:
            json.dump(self.info, fout, indent=4)

        logging.info(f'Saved info at {saved_datetime}')

    def __load(self):
        """
        Loads the info dictionary from the json file in the directory.
        """

        with open(f'{self.dir}/info.json', 'r') as fin:
            info = json.load(fin)

        logging.info(f'Loaded info from {self.dir}/info.json')

        return info
            
    def add(self, key:str, value):
        """
        Adds a new key-value pair to the info dictionary.

        Parameters
        ----------
            key : str
                key of the new key-value pair
            value : int, float, str, torch.Tensor
                value of the new key-value pair
        """

        if not isinstance(key, str):
            raise Exception('The key should be string')

        if isinstance(value, (int, float, str)):
            self.info[key] = value
            self.__save()
            return value
        elif isinstance(value, Figure):
            f_save = key.replace(' ', '_')
            f_path = f'{self.dir}/{f_save}.png'
            
            self.info[key] = f_path
            value.savefig(f_path)

            self.__save()
        else:
            f_save = key.replace(' ', '_')
            f_path = f'{self.dir}/{f_save}.pt'

            self.info[key] = f_path
            torch.save(value, f_path)

            self.__save()
            
            return f_path

    def remove(self, key:str):
        """
        Removes an existing key-value pair from the info dictionary.

        Parameters
        ----------
            key : str
                key of the key-value pair to be removed
        """

        if key not in self.info.keys():
            raise Exception(f'{key} dose not exist.')

        value = self.info[key]

        if isinstance(value, str) and os.path.exists(value):
            os.remove(value)

        del self.info[key]

        self.__save()

        return key

    def get(self, key:str):
        """
        Gets the value of a specific key in the info dictionary.

        Parameters
        ----------
            key : str
                key of the key-value pair to be fetched
        """

        if key not in self.info.keys():
            raise Exception(f'the key "{key}" dose not exist.')

        value = self.info[key]

        if isinstance(value, str) and self.dir in value \
           and os.path.exists(value) and not value.endswith('.png'):
            return torch.load(value)
        else:
            return value

    def keys(self):
        """
        Returns all keys in the info dictionary.
        """

        return list(self.info.keys())
