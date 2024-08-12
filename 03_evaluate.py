from sources.model import KnowledgeGraphEmbeddingOnABCI
from sources.dataset import KnowledgeGraph
from sources.evaluation import Evaluator
from util.databinder import DataBinder

if __name__ == "__main__":
    # Create an instance of the KnowledgeGraphEmbeddingOnABCI class
    kge = KnowledgeGraphEmbeddingOnABCI(dir_save='./models/20240813/kinships_transe_try1')

    # Load the knowledge graph data
    db = DataBinder(target_dir='./data/processed/20240813/false_dataset_kinships')
    false_dataset = db.get('dict_false_triples')
    print(false_dataset)

    # Create an instance of the Evaluator class
    evaluator = Evaluator()
    evaluation_results = evaluator.evaluate(kge,false_dataset)

    print(evaluation_results)