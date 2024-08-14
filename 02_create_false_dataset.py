from sources.dataset import KnowledgeGraph
import random

if __name__ == "__main__":

    """parameters"""
    base_dir_save = './data/processed/20240815'
    dataset_name = 'wn18rr'
    num_seeds = 10
    random_seed = 0
    """"""

    random.seed(random_seed)  # Fix random seed
    random_seeds = [random.randint(1, 1000) for _ in range(num_seeds)]
    
    kg = KnowledgeGraph(dataset_name=dataset_name)
    kg.create_false_dataset(random_seeds=random_seeds,
                            dir_save=f'{base_dir_save}/false_dataset_{dataset_name}_random_seed_{0}')