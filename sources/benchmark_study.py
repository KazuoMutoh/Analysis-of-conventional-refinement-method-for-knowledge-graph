import pandas as pd
import numpy as np

def convert_dtype(value):
    if isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    else:
        return value

def get_best_params(f_params:str, 
                    model_name:str, 
                    dataset_name:str,
                    use_label_smoothing:bool=False):
    """
    最適なハイパーパラメータを取得する．
    TODO:
        洗練されていないので修正すること．
    """
    df_best_params = pd.read_pickle(f_params).reset_index()
    df_best_params = df_best_params[df_best_params['model'].isin([model_name])]
    df_best_params = df_best_params[df_best_params['dataset'].isin([dataset_name])]
    
    dict_args = {}
    idx = df_best_params.index[0]
    
    dict_args['dataset'] = df_best_params.loc[idx,'dataset']
    
    dict_args['dataset_kwargs'] = {}
    for k, v in df_best_params.filter(regex='pipeline_config.pipeline.dataset_kwargs').loc[idx].items():
        if not np.isnan(v):
            dict_args['dataset_kwargs'][k.split('.')[-1]] = v

    dict_args['evaluator'] = df_best_params.loc[idx,'evaluator']

    dict_args['evaluator_kwargs'] = {}
    for k, v in df_best_params.filter(regex='pipeline_config.pipeline.evaluator_kwargs').loc[idx].items():
        if not np.isnan(v):
            dict_args['evaluator_kwargs'][k.split('.')[-1]] = v
    
    dict_args['model'] = df_best_params.loc[idx,'model']

    dict_args['loss'] = df_best_params.loc[idx, 'loss']

    dict_args['regularizer'] = df_best_params.loc[idx, 'regularizer']

    dict_args['optimizer'] = df_best_params.loc[idx, 'optimizer']

    dict_args['optimizer_kwargs'] = {}
    for k, v in df_best_params.filter(regex='pipeline_config.pipeline.optimizer_kwargs').loc[idx].items():
        if not np.isnan(v) and 'automatic_memory_optimization' not in k:
            dict_args['optimizer_kwargs'][k.split('.')[-1]] = convert_dtype(v)
    
    dict_args['model_kwargs'] = {}
    for k, v in df_best_params.filter(regex='pipeline_config.pipeline.model_kwargs').loc[idx].items():
        if not np.isnan(v) and 'automatic_memory_optimization' not in k:
            k = k.split('.')[-1]
            if k in ['output_channels', 'kernel_height', 'kernel_width']:
                v = int(v)    
            dict_args['model_kwargs'][k] = convert_dtype(v)

    dict_args['training_loop'] = df_best_params.loc[idx, 'training_loop']

    dict_args['training_kwargs'] = {}
    for k, v in df_best_params.filter(regex='pipeline_config.pipeline.training_kwargs').loc[idx].items():
        if not np.isnan(v):
            k = k.split('.')[-1]
            if k in ['batch_size', 'num_epochs']:
                v = int(v)

            if not use_label_smoothing:
                if k not in ['label_smoothing']:
                    dict_args['training_kwargs'][k.split('.')[-1]] = v
            else:
                dict_args['training_kwargs'][k.split('.')[-1]] = v

    return dict_args