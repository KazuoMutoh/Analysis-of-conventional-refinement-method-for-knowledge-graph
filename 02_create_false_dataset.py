from sources.dataset import KnowledgeGraph

if __name__ == "__main__":

    random_seeds = [42, 123, 456, 789, 101112]
    dataset_name = 'kinships'
    
    kg = KnowledgeGraph(dataset_name=dataset_name)
    kg.create_false_dataset(random_seeds=random_seeds,dir_save=f'./data/processed/false_dataset_{dataset_name}')