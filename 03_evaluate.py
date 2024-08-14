from sources.model import KnowledgeGraphEmbeddingOnABCI
from sources.evaluation import Evaluator
from util.databinder import DataBinder

if __name__ == "__main__":
    """parameters"""
    dir_model = './models/20240813/fb15k237_transe_try1'
    dir_data = './data/processed/20240815/false_dataset_fb15k237_random_seed_0'
    dir_report = './reports/20240815/evaluation_fb15k237_transe_try1'

    # Create an instance of the KnowledgeGraphEmbeddingOnABCI class
    kge = KnowledgeGraphEmbeddingOnABCI(dir_save=dir_model)

    # Load the knowledge graph data
    db_dataset = DataBinder(target_dir=dir_data)
    false_dataset = db_dataset.get('dict_false_triples')
    print(false_dataset )

    # Create an instance of the Evaluator class
    evaluator = Evaluator()
    evaluation_results = evaluator.evaluate(kge,false_dataset)
    #print(evaluation_results)

    db_eval = DataBinder(target_dir=dir_report)
    db_eval.add('dir_model', dir_model)
    db_eval.add('dir_data',dir_data)
    db_eval.add('evaluation_results', evaluation_results)
    db_eval.add('false_dataset', false_dataset)
    

    

