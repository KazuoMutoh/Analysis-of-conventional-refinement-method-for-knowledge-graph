from sources.benchmark_study import get_best_params
from sources.model import KnowledgeGraphEmbeddingOnABCI
import sys

if __name__ == "__main__":

    model_name = 'transe'
    dataset_name = 'wn18rr'
    _id = 'try1'
    dir_save = f'./models/20240813/{dataset_name}_{model_name}_{_id}'
    random_seeds = [42, 123, 456, 789, 101112]

    dict_args = get_best_params('df_best_param.pkl', 
                                model_name=model_name,
                                dataset_name=dataset_name)
    
    print(dict_args)
    
    kge = KnowledgeGraphEmbeddingOnABCI(dir_save=dir_save)
    kge.submit_jobs(dict_args,random_seeds=random_seeds,h_rt='8:00:00')
    #kge.monitor_jobs()
