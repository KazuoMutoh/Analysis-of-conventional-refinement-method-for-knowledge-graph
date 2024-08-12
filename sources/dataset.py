import numpy as np
import pandas as pd
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory
from pykeen.triples.triples_factory import TriplesFactory
from typing import List, Dict, Union
import os



try:
    from util.databinder import DataBinder
except ModuleNotFoundError:
    import sys
    sys.path.append('../util')
    from databinder import DataBinder


class KnowledgeGraph:

    def __init__(self, 
                 dataset_name:str,
                 training:TriplesFactory=None,
                 testing:TriplesFactory=None,
                 validation:TriplesFactory=None,
                 **kwargs):
        
        self.dataset_name = dataset_name

        if dataset_name != None:
            dataset = get_dataset(dataset=dataset_name, **kwargs)
            self.dataset_name = dataset.get_normalized_name()
            self.training = dataset.training
            self.testing = dataset.testing
            self.validation = dataset.validation
        elif training != None and testing != None:
            self.dataset = None
            self.training = training
            self.testing = testing
            self.validation = validation
        else:
            raise Exception('either dataset_name of training/testing/validation should be specified')

    def create_false_dataset(self, 
                             ratio:int=0.1, 
                             random_seeds:List=[0],
                             dir_save=None) -> Dict:
        """
        create false triples factor by replacing 
        head and tail of triples randomly.

        Args:
            ratio (int)
            random_seeds(list)

        Returns:
            dict_false_triples (Dict):
                a dictionary whose keys are random seeds
                and values are dictionary. 
                The dictionary has two key: triples factory and dataframe.
        """

        df_tt_features = self.create_triples_feature_table(self.testing)

        dict_false_triples = {}
        for random_seed in random_seeds:

            tf_tf, false_indices = self.create_false_triples(self.testing, ratio, random_seed)

            df_tf_features = self.create_triples_feature_table(tf_tf)
        
            df1 = df_tt_features.copy(deep=True)
            df1.rename(columns={
                'head':'head(org)',
                'relation':'relation(org)',
                'tail':'tail(org)',
                'head_degree':'head_degree(org)',
                'tail_degree':'tail_degree(org)'},
                inplace=True)
            df2 = df_tf_features.copy(deep=True)
            df_tt_tf_features = pd.concat([df1, df2], axis=1)
            df_tt_tf_features['is-error'] = [(True) if (idx in false_indices) 
                                             else (False) for idx in df_tt_tf_features.index]
            df_tt_tf_features['degree'] = df_tt_tf_features['head_degree'] + df_tt_tf_features['tail_degree']

            dict_false_triples[random_seed] = {'triples_factory':tf_tf, 'dataframe':df_tt_tf_features}

        if dir_save !=None:
            
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)

            db = DataBinder(target_dir=dir_save)
            db.add('description', 'false triples')
            db.add('false_ratio', ratio)
            db.add('random_seeds', random_seeds)
            db.add('dataset_name', self.dataset_name)
            db.add('dict_false_triples', dict_false_triples)

        return dict_false_triples

    @staticmethod
    def create_false_triples(tf:TriplesFactory, 
                             ratio:float=0.1, 
                             random_seed:int=0) -> Union[TriplesFactory,np.array]:

        # fix random seed
        np.random.seed(random_seed)
        
        num_false = int(tf.num_triples * ratio)
        false_triples = tf.mapped_triples.clone()
        false_indices = np.random.choice(tf.num_triples, num_false)
        for i in false_indices:
            if np.random.random() < 0.5:  # Replace head
                false_triples[i, 0] = np.random.choice(tf.num_entities)
            else:  # Replace tail
                false_triples[i, 2] = np.random.choice(tf.num_entities)

        false_tf = TriplesFactory(false_triples, tf.entity_to_id, tf.relation_to_id, 
                                  create_inverse_triples=tf.create_inverse_triples)

        return false_tf, false_indices
    
    @staticmethod
    def create_triples_feature_table(tf:TriplesFactory) -> pd.DataFrame:
        
        triples = tf.mapped_triples.numpy()
        head_labels = [tf.entity_id_to_label[h] for h in triples[:, 0]]
        relation_labels = [tf.relation_id_to_label[r] for r in triples[:, 1]]
        tail_labels = [tf.entity_id_to_label[t] for t in triples[:, 2]]
        
        # Calculate degrees
        head_degrees = np.array([np.sum(triples[:, 0] == h) for h in triples[:, 0]])
        tail_degrees = np.array([np.sum(triples[:, 2] == t) for t in triples[:, 2]])
        
        return pd.DataFrame({
            'head': head_labels,
            'relation': relation_labels,
            'tail': tail_labels,
            'head_degree': head_degrees,
            'tail_degree': tail_degrees,
        })

if __name__ == "__main__":

    # Create an instance of the KnowledgeGraph class
    kg = KnowledgeGraph(dataset_name="fb15k237", 
                        dataset_kwargs={'create_inverse_triples':True})

    # Create a false dataset
    false_dataset = kg.create_false_dataset(ratio=0.1, 
                                            random_seeds=[0, 1, 2],
                                            dir_save='test')

    # Print the false dataset for each random seed
    for random_seed, data in false_dataset.items():
        print(f"Random Seed: {random_seed}")
        print(data['dataframe'])
        print(data['triples_factory'])
        print()
        

            
        
        
