import os
import logging
import copy
import numpy as np
from typing import Dict, List, Union
from tqdm import tqdm
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
import torch
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class KnowledgeGraphEmbedding:

    def __init__(self, dir_save:str=None):
        """
        Args:
            dir_save (str) :
                A directory where embedding models of knowledge graph are saved.
        """

        self.dir_save = dir_save
        self.random_seeds = []

        if not os.path.exists(self.dir_save):
            os.mkdir(self.dir_save)
        else:
            random_seeds_path = f'{self.dir_save}/random_seeds.json'
            if os.path.exists(random_seeds_path):
                with open(random_seeds_path, 'r') as f:
                    self.random_seeds = json.load(f)

    
    @staticmethod
    def from_directory(cls,dir_save):
        """
        Create an instance of KnowledgeGraphEmbedding from a directory.

        Args:
            dir_save (str):
                The directory where the embedding models of the knowledge graph are saved.

        Returns:
            KnowledgeGraphEmbedding:
                An instance of the KnowledgeGraphEmbedding class.
        """
        return cls(dir_save=dir_save)
    
    
    def train(self, dict_args:Dict, random_seeds:List):
        """
        Args:
            dict_args (dict):
                A dictionary of parameters passed to pykeen pipeline 
                except random seed. 
        """

        logger.info('start learning ...')
        
        for random_seed in tqdm(random_seeds):
            
            _dict_args = copy.deepcopy(dict_args)
            _dict_args['random_seed'] = random_seed
            pipeline_result = pipeline(**_dict_args)

            _dir_save = f'{self.dir_save}/{random_seed}'
            if not os.path.exists(_dir_save):
                os.mkdir(_dir_save)
            
            pipeline_result.save_to_directory(f'{_dir_save}/result')

        self.random_seeds = random_seeds
        # Save self.random_seeds in self.dir_save
        random_seeds_path = f'{self.dir_save}/random_seeds.json'
        with open(random_seeds_path, 'w') as f:
            json.dump(self.random_seeds, f)
    
    def score_hrt(self, 
                  triples_factory:TriplesFactory, 
                  random_seeds:Union[int,List]=None) -> Dict:
        """
        calculate scores of triples for triples_factory.
        If random_seeds is a int, calculate score with model 
        trained with corresponding random_seeds.
        """

        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        elif random_seeds == None:
            random_seeds = self.random_seeds
        elif isinstance(random_seeds,List):
            pass
        else:
            Exception('random_seeds is either int, list or None')
        
        dict_scores = {}
        for random_seed in random_seeds:
            
            if random_seed not in self.random_seeds:
                logger.warning(f'model corresponding to random seed {random_seed} \
                               has not learned.')

            model_path = f'{self.dir_save}/{random_seed}/result/trained_model.pkl'
            model = torch.load(model_path)
            dict_scores[random_seed] = model.score_hrt(triples_factory.mapped_triples).cpu().detach().numpy()
        
        return dict_scores

    def get_random_seeds(self):
        """
        Returns:
            List[int]: The list of random seeds used for learning.
        """
        return self.random_seeds
        
    def get(self, selection:str, random_seeds:Union[int,List]=None):
        """
        Args:
            random_seeds (Union[int,List]):
                The random seeds used for learning the models.
            selection (str):
                The selection of what to return. 
                Possible values are 'model', 'model_info', and 'dir_path'.

        Returns:
            The selected result(s) based on the given selection as a dictionary with random seed as the key.
        """
        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        elif isinstance(random_seeds, list):
            pass
        elif random_seeds == None:
            random_seeds = self.random_seeds
        else:
            raise ValueError("random_seeds must be either an integer or a list of integers.")
        
        results = {}
        for random_seed in random_seeds:
            if random_seed not in self.random_seeds:
                raise Exception(f"Random seed {random_seed} is not in the list of random seeds used for learning.")
            
            model_path = f'{self.dir_save}/{random_seed}/result/trained_model.pkl'
            json_path = f'{self.dir_save}/{random_seed}/result/results.json'
            dir_path = f'{self.dir_save}/{random_seed}/result'
            
            model = torch.load(model_path)
            
            with open(json_path, 'r') as f:
                model_info = json.load(f)
            
            if selection == 'model':
                results[random_seed] = model
            elif selection == 'model_info':
                results[random_seed] = model_info
            elif selection == 'dir_path':
                results[random_seed] = dir_path
            else:
                raise ValueError("Invalid selection. Possible values are 'model', 'model_info', and 'dir_path'.")
        
        return results

            

if __name__ == "__main__":
    # Instantiate the KnowledgeGraphEmbedding class
    kg_embedding = KnowledgeGraphEmbedding(dir_save="../models/20240812/test")

    # Define the parameters for the pykeen pipeline
    dict_args = {
        "dataset": "fb15k237",
        "model": "TransE",
        'training_kwargs': {"num_epochs":2}
    }

    # Define the random seeds for learning
    random_seeds = [42, 123, 456]

    # Learn the knowledge graph embeddings
    kg_embedding.train(dict_args, random_seeds)

    # Get the list of random seeds used for learning
    seeds = kg_embedding.get_random_seeds()
    print("Random Seeds:", seeds)

    # Get the trained model for a specific random seed
    random_seed = 42
    model = kg_embedding.get(random_seed, "model")
    print("Trained Model:", model)

    # Get the model information for a specific random seed
    model_info = kg_embedding.get(random_seed, "model_info")
    print("Model Information:", model_info)

    # Get the directory path for a specific random seed
    dir_path = kg_embedding.get(random_seed, "dir_path")
    print("Directory Path:", dir_path)

    
    