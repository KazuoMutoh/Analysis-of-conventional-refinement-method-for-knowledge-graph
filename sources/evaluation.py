import numpy as np
from typing import Dict
from sklearn.metrics import precision_recall_curve, auc
import os
import sys

# Check if the module is being executed as a script
if __name__ == "__main__":
    # Get the directory where this module is saved
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the module directory to the Python path
    sys.path.append(module_dir)

    # Import the necessary modules using relative path
    from model import KnowledgeGraphEmbedding
    from dataset import KnowledgeGraph

else:
    # Import the necessary modules using absolute path
    from .model import KnowledgeGraphEmbedding
    from .dataset import KnowledgeGraph


class Evaluator:
    """
    Class for evaluating the performance of a knowledge graph embedding model.
    
    Methods:
        evaluate(kge, dict_false_triples) -> Dict:
            Evaluate the model using the provided knowledge graph embedding (KGE) model and false triples.
            
            Args:
                kge (object): The knowledge graph embedding model.
                dict_false_triples (dict): A dictionary containing false triples for evaluation.
            
            Returns:
                dict_evaluation_results (dict): A dictionary containing the evaluation results.
        
        _get_hits_at_k(model) -> Dict:
            Calculate the hits@k metric for the given KGE model.
            
            Args:
                model (dict): The KGE model.
            
            Returns:
                dict_hits_at_k (dict): A dictionary containing the hits@k metric for each seed.
        
        _calculate_true_negative_ratio(scores, false_indices, top=[0.01]) -> float:
            Calculate the true negative ratio for the given scores and false indices.
            
            Args:
                scores (np.array): The scores of triples calculated by the KGE model.
                false_indices (np.array): A list of indices of scores whose corresponding triples are false.
                top (list, optional): The top percentage of scores to consider for calculating the threshold. Defaults to [0.01].
            
            Returns:
                true_negative_ratio (float): The true negative ratio.
        
        _calculate_precision_recall_curve(scores, false_indices) -> dict:
            Calculate the precision-recall curve and corresponding area under curve (AUC) for the given scores and false indices.
            
            Args:
                scores (np.array): The scores of triples calculated by the KGE model.
                false_indices (np.array): A list of indices of scores whose corresponding triples are false.
            
            Returns:
                dict_precision_recall_curve (dict): A dictionary containing the precision, recall, and AUC values.
    """
    def __init__(self):
        pass

    def evaluate(self, kge, dict_false_triples) -> Dict:
        """
        Evaluate the performance of a knowledge graph embedding model using the provided KGE model and false triples.
        
        Args:
            kge (object): The knowledge graph embedding model.
            dict_false_triples (dict): A dictionary containing false triples for evaluation.
        
        Returns:
            dict_evaluation_results (dict): A dictionary containing the evaluation results.
        """
        dict_evaluation_results = {}
        
        for model_random_seed, trained_model in kge.get(selection='model').items():
            
            dict_evaluation_results[model_random_seed] = {}
            
            for data_random_seed, _dict_false_triples in dict_false_triples.items():
                dict_evaluation_results[model_random_seed][data_random_seed] = {}
                df = _dict_false_triples['dataframe']
                false_indices = np.array(df[df['is-error']==True].index)
                scores = trained_model.score_hrt(_dict_false_triples['triples_factory'].mapped_triples).cpu().detach().numpy().flatten()
                dict_tnr = self._calculate_true_negative_ratio(scores,false_indices,top=[0.01,0.03,0.05,0.1])
                dict_pr_curve = self._calculate_precision_recall_curve(scores,false_indices)
                dict_evaluation_results[model_random_seed][data_random_seed]['true_negative_ratio'] = dict_tnr
                dict_evaluation_results[model_random_seed][data_random_seed]['precision_recall_curve'] = dict_pr_curve
                dict_evaluation_results[model_random_seed][data_random_seed]['model_info'] =\
                      self._get_hits_at_k(kge.get(random_seeds=model_random_seed,selection='model_info'))

        return dict_evaluation_results
                

    def _get_hits_at_k(self, dict_model_info) -> Dict:
        """
        Calculate the hits@k metric for the given KGE model.
        
        Args:
            model (dict): The KGE model.
        
        Returns:
            dict_hits_at_k (dict): A dictionary containing the hits@k metric for each seed.
        """
        dict_hits_at_k = {}
        
        for seed, _dict_model_info in dict_model_info.items():
            dict_hits_at_k[seed] = {}
            
            for i in [1,3,5,10]:
                dict_hits_at_k[seed][i] = _dict_model_info['metrics']['both']['realistic'][f'hits_at_{i}']
        
        return dict_hits_at_k
    
    def _calculate_true_negative_ratio(self, scores:np.array, false_indices:np.array, top=[0.01]) -> float:
        """
        Calculate the true negative ratio for the given scores and false indices.
        
        Args:
            scores (np.array): The scores of triples calculated by the KGE model.
            false_indices (np.array): A list of indices of scores whose corresponding triples are false.
            top (list, optional): The top percentage of scores to consider for calculating the threshold. Defaults to [0.01].
        
        Returns:
            true_negative_ratio (float): The true negative ratio.
        """
        # Sort the scores in descending order
        sorted_scores = np.sort(scores)[::-1]
        
        dict_true_negative_ratio = {}
        for t in top:
            # Calculate the threshold for true negatives
            threshold = sorted_scores[int(len(sorted_scores) * t)]
            
            # Count the number of false scores below the threshold
            num_false_below_threshold = np.sum(scores[false_indices] < threshold)
            
            # Calculate the true negative ratio
            true_negative_ratio = num_false_below_threshold / len(false_indices)

            dict_true_negative_ratio[t] = true_negative_ratio
        
        return dict_true_negative_ratio
    
    def _calculate_precision_recall_curve(self, scores:np.array, false_indices:np.array) -> Dict:
        """
        Calculate the precision-recall curve and corresponding area under curve (AUC) for the given scores and false indices.
        
        Args:
            scores (np.array): The scores of triples calculated by the KGE model.
            false_indices (np.array): A list of indices of scores whose corresponding triples are false.
        
        Returns:
            dict_precision_recall_curve (dict): A dictionary containing the precision, recall, and AUC values.
        """
        # Get the scores of true triples
        true_scores = scores[~false_indices]
        
        # Get the scores of false triples
        false_scores = scores[false_indices]
        
        # Create the labels for true and false triples
        true_labels = np.ones_like(true_scores)
        false_labels = np.zeros_like(false_scores)
        
        # Concatenate the scores and labels
        all_scores = np.concatenate([true_scores, false_scores])
        all_labels = np.concatenate([true_labels, false_labels])
        
        # Calculate the precision and recall values
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        
        # Calculate the area under the precision-recall curve
        auc_score = auc(recall, precision)
        
        # Create the dictionary with the precision, recall, and auc values
        dict_precision_recall_curve = {
            'precision': precision,
            'recall': recall,
            'auc': auc_score
        }
        
        return dict_precision_recall_curve
    
if __name__ == "__main__":
    """
    # Instantiate the KnowledgeGraphEmbedding class
    kg_embedding = KnowledgeGraphEmbedding(dir_save="../models/20240812/test")

    # Define the parameters for the pykeen pipeline
    dict_args = {
        "dataset": "fb15k237",
        "model": "TransE",
        'training_kwargs': {"num_epochs":2}
    }

    # Define the random seeds for learning
    random_seeds = [42, 123]

    # Learn the knowledge graph embeddings
    kg_embedding.train(dict_args, random_seeds)
    """


    kg_embedding = KnowledgeGraphEmbedding(dir_save="../models/20240812/test")

    # Instantiate the KnowledgeGraph class
    kg = KnowledgeGraph(dataset_name="fb15k237", 
                        dataset_kwargs={'create_inverse_triples':True})

    # Create a false dataset
    false_dataset = kg.create_false_dataset(ratio=0.1, 
                                            random_seeds=[0, 1, 2],
                                            dir_save='test')

    # Instantiate the Evaluator class
    evaluator = Evaluator()

    # Evaluate the KGE model using the false triples
    evaluation_results = evaluator.evaluate(kg_embedding, false_dataset)

    # Print the evaluation results
    print("Evaluation Results:")
    for model_random_seed, model_results in evaluation_results.items():
        print(f"Model Random Seed: {model_random_seed}")
        for data_random_seed, data_results in model_results.items():
            print(f"Data Random Seed: {data_random_seed}")
            print(f"True Negative Ratio: {data_results['true_negative_ratio']}")
            print(f"Precision-Recall Curve AUC: {data_results['precision_recall_curve']['auc']}")
            print(f"model-info:{data_results['model_info']}")
        print()