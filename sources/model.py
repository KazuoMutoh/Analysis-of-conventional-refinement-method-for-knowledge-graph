import os
import logging
import copy
from typing import Dict, List, Union
from pykeen.pipeline import pipeline
from pykeen.triples.triples_factory import TriplesFactory
import torch
import json
import yaml
import textwrap
import time
import shutil

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class KnowledgeGraphEmbedding:
    def __init__(self, dir_save: str = None):
        """
        Initialize the KnowledgeGraphEmbedding class.

        Args:
            dir_save (str):
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
    def from_directory(cls, dir_save):
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

    def train(self, 
              dict_args: Dict, 
              random_seeds: List,
              delete_previous_results:bool=False):
        """
        Train the knowledge graph embeddings.

        Args:
            dict_args (dict):
                A dictionary of parameters passed to pykeen pipeline
                except random seed.
            random_seeds (List):
                A list of random seeds used for training.
            delete_previous_results (bool):
                If True, delete previous results.
        """

        if self._result_exists(random_seeds):
            if delete_previous_results:
                logger.info('delete previous results')
                self._delete_previous_results()
            else:
                raise Exception(f'Results have already existed. Change random seed or start new.\
                                 Results corresponding to {self.random_seeds} have existed.')

        for random_seed in random_seeds:

            _dict_args = copy.deepcopy(dict_args)
            _dict_args['random_seed'] = random_seed
            pipeline_result = pipeline(**_dict_args)

            _dir_save = f'{self.dir_save}/{random_seed}'
            if not os.path.exists(_dir_save):
                os.mkdir(_dir_save)

            pipeline_result.save_to_directory(f'{_dir_save}/result')

        self.random_seeds += random_seeds
        
        # Save self.random_seeds in self.dir_save
        random_seeds_path = f'{self.dir_save}/random_seeds.json'
        with open(random_seeds_path, 'w') as f:
            json.dump(self.random_seeds, f)

    def _result_exists(self, random_seeds):

        for random_seed in random_seeds:
            _dir_save = f'{self.dir_save}/{random_seed}'
            if os.path.exists(f'{_dir_save}/result'):
                logger.info(f'{_dir_save} exists ...')
                return True
        return False
            
    def _delete_previous_results(self):

        for random_seed in self.random_seeds:
            result_dir = f'{self.dir_save}/{random_seed}/result'
            if os.path.exists(result_dir):
                logger.info(f"Deleting previous results for random seed {random_seed}")
                shutil.rmtree(f'{self.dir_save}/{random_seed}')

    def score_hrt(self, triples_factory: TriplesFactory, random_seeds: Union[int, List] = None) -> Dict:
        """
        Calculate scores of triples for triples_factory.

        Args:
            triples_factory (TriplesFactory):
                The triples factory object containing the triples.
            random_seeds (Union[int,List], optional):
                The random seeds used for training the models. If None, uses the random seeds used during training.
                If an integer, calculates the score with the model trained with the corresponding random seed.
                If a list of integers, calculates the scores with the models trained with the corresponding random seeds.

        Returns:
            dict:
                A dictionary containing the scores of triples for each random seed.
        """
        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        elif random_seeds is None:
            random_seeds = self.random_seeds
        elif isinstance(random_seeds, List):
            pass
        else:
            raise ValueError('random_seeds is either int, list or None')

        dict_scores = {}
        for random_seed in random_seeds:
            if random_seed not in self.random_seeds:
                logger.warning(f'model corresponding to random seed {random_seed} has not learned.')

            model_path = f'{self.dir_save}/{random_seed}/result/trained_model.pkl'
            model = torch.load(model_path)
            dict_scores[random_seed] = model.score_hrt(triples_factory.mapped_triples).cpu().detach().numpy()

        return dict_scores

    def get_random_seeds(self):
        """
        Get the list of random seeds used for learning.

        Returns:
            List[int]:
                The list of random seeds used for learning.
        """
        return self.random_seeds

    def get(self, selection: str, random_seeds: Union[int, List] = None):
        """
        Get the selected result(s) based on the given selection.

        Args:
            selection (str):
                The selection of what to return. Possible values are 'model', 'model_info', and 'dir_path'.
            random_seeds (Union[int,List], optional):
                The random seeds used for learning the models. If None, uses the random seeds used during training.
                If an integer, gets the result for the model trained with the corresponding random seed.
                If a list of integers, gets the results for the models trained with the corresponding random seeds.

        Returns:
            dict:
                The selected result(s) based on the given selection as a dictionary with random seed as the key.
        """
        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        elif isinstance(random_seeds, list):
            pass
        elif random_seeds is None:
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


class KnowledgeGraphEmbeddingOnABCI(KnowledgeGraphEmbedding):

    def __init__(self, dir_save):
        super().__init__(dir_save)

    def submit_jobs(self, 
                    dict_args: Dict, 
                    random_seeds: List,
                    delete_previous_results:bool=False,
                    h_rt:str='01:00:00',
                    resource_type:str='rt_G.small') -> None:
        """
        Submit jobs for training knowledge graph embedding to ques.

        Args:
            dict_args (Dict): A dictionary containing the arguments for training.
            random_seeds (List): A list of random seeds for training.
            h_rt (str, optional): The maximum runtime for each job. Defaults to '01:00:00'.
        Returns:
            None
        """

        if self._result_exists(random_seeds):
            if delete_previous_results:
                logger.info('delete previous results')
                self._delete_previous_results()
            else:
                raise Exception(f'Results have already existed. Change random seed or start new.\
                                 Results corresponding to {self.random_seeds} have existed.')
        
        for random_seed in random_seeds:
            
            _dict_args = copy.deepcopy(dict_args)
            _dict_args['random_seed'] = random_seed

            _dir_save = f'{self.dir_save}/{random_seed}'
            if not os.path.exists(_dir_save):
                os.mkdir(_dir_save)

            yaml_path = f'{_dir_save}/params.yml'
            with open(yaml_path, 'w') as f:
                yaml.dump(_dict_args, f)


            script = textwrap.dedent(f"""
            #!/bin/bash
            
            # options for batch job execution.
            # for all options, please check https://docs.abci.ai/ja/job-execution/#job-execution-options 
            
            #$ -l {resource_type}=1
            #$ -l h_rt={h_rt}
            #$ -m a
            #$ -m b
            #$ -m e
            #$ -j y
            #$ -o {_dir_save}/log.txt
            #$ -cwd
            
            source /etc/profile.d/modules.sh
            source /home/acg16558pn/kg_20240423/bin/activate
            module load cuda/12.1
            module load python/3.10
            python script_pykeen_pipeline.py -i {_dir_save}/params.yml -o {_dir_save}/result
            """)
        
            with open(f'{_dir_save}/script.sh', 'w') as fout:
                fout.write(script)
        
            os.system(f'qsub -g gcc50441 {_dir_save}/script.sh') 


        self.random_seeds = random_seeds
        
        # Save self.random_seeds in self.dir_save
        random_seeds_path = f'{self.dir_save}/random_seeds.json'
        with open(random_seeds_path, 'w') as f:
            json.dump(self.random_seeds, f)


    def monitor_jobs(self, monitoring_period:float=60.0):
        """
        Monitoring if jobs finished or not.
        When a job has finished, a directory "result" will be created in 
        self.dir_save/{random_seed}. 

        Args:
            monitoring_period:
                a interval to check if jobs finished or not.

        Returns:
            message:
        """
        while True:
        
            finished_jobs = []
            for random_seed in self.random_seeds:
                _dir_save = f'{self.dir_save}/{random_seed}'
                result_dir = f'{_dir_save}/result'
                if os.path.exists(result_dir):
                    finished_jobs.append(random_seed)
            if len(finished_jobs) == len(self.random_seeds):
                message = "All jobs finished."
                logger.info(message)
                break
            else:
                message = f"Jobs {finished_jobs} finished. Waiting for other jobs to finish."
                logger.info(message)
                time.sleep(monitoring_period)
        
        return message



if __name__ == "__main__":
    """
    # Instantiate the KnowledgeGraphEmbedding class
    kg_embedding = KnowledgeGraphEmbedding(dir_save="../models/20240812/test")

    # Define the parameters for the pykeen pipeline
    dict_args = {
        "dataset": "fb15k237",
        "model": "TransE",
        'training_kwargs': {"num_epochs": 2}
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
    """

    # Instantiate the KnowledgeGraphEmbedding class
    kg_embedding = KnowledgeGraphEmbeddingOnABCI(dir_save="../models/20240813/test2")

    # Define the parameters for the pykeen pipeline
    dict_args = {
        "dataset": "fb15k237",
        "model": "TransE",
        'training_kwargs': {"num_epochs": 2}
    }

    # Define the random seeds for learning
    random_seeds = [10, 20, 456]

    # Learn the knowledge graph embeddings
    #kg_embedding.train(dict_args, random_seeds)
    kg_embedding.submit_jobs(dict_args, random_seeds, h_rt='00:01:00', delete_previous_results=True)
    kg_embedding.monitor_jobs(monitoring_period=10)

    # Get the list of random seeds used for learning
    seeds = kg_embedding.get_random_seeds()
    print("Random Seeds:", seeds)

    # Get the trained model for a specific random seed
    random_seed = 10
    model = kg_embedding.get(random_seeds=random_seed, selection="model")
    print("Trained Model:", model)

    # Get the model information for a specific random seed
    model_info = kg_embedding.get(random_seeds=random_seed, selection="model_info")
    print("Model Information:", model_info)

    # Get the directory path for a specific random seed
    dir_path = kg_embedding.get(random_seeds=random_seed, selection="dir_path")
    print("Directory Path:", dir_path)
    


