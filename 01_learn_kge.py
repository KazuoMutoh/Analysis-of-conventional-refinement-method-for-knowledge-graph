#!/usr/bin/env python
# coding: utf-8

# # Knowledge Embedding with Pykeen
# ## 概要
# - 知識グラフから知識グラフの埋め込みモデルを学習する．
# - pykeenにあらかじめ入っているデータセットに関しては，この[GitHub](https://github.com/pykeen/benchmarking)のページを参考にハイパーパラメータを設定．
# ## 入力データ・パラメータ
# - 知識グラフ埋め込みモデル名
# - 知識グラフ
# - ランダムシード（optional）
# ## 出力データ
# - 各ランダムシードの知識グラフ
# - メタデータ

# ## modules

# In[1]:


import copy
import os
import logging
import shutil
import multiprocessing
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset

from util.databinder import DataBinder


# In[2]:


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# ## functions

# In[3]:


def convert_dtype(value):
    if isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    else:
        return value

def delete_all_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)

def get_best_params(f_params:str, model_name:str, dataset_name:str):
    """
    最適なハイパーパラメータを取得する．
    TODO:
        洗練されていないので修正すること．
    """
    df_best_params = pd.read_pickle(f_params).reset_index()
    df_best_params = df_best_params[df_best_params['model'].isin([model_name])]
    df_best_params = df_best_params[df_best_params['dataset'].isin([dataset_name])]
    
    dict_args = {}
    idx = 0
    
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

            #if k not in ['label_smoothing']:
            #    dict_args['training_kwargs'][k.split('.')[-1]] = v
            dict_args['training_kwargs'][k.split('.')[-1]] = v

    return dict_args


def learn_embedding(dict_args:dict, dir_save:str=None):
    """
    知識グラフ埋込モデルを計算する.
    
    Args:
        dict_args(dict):
            埋め込みモデル学習のためのパラメータ．
            デフォルトのパラメータを使う場合でも，
            datasetとmodelのキーワードは必須
        dir_save(str):
            None以外の時は指定されたディレクトリに学習結果を
            保存する．
    Returns:
        pipeline_result:
            pykeenのpipelineのresult
            [PipelineResult](https://pykeen.readthedocs.io/en/stable/api/pykeen.pipeline.PipelineResult.html)
    """

    """
    # 辞書型で渡せるので以下のコードは使わない
    pipeline_result = pipeline(
        dataset=dict_args['dataset'],
        dataset_kwargs = dict_args['dataset_kwargs'],
        evaluator=dict_args['evaluator'],
        evaluator_kwargs = dict_args['evaluator_kwargs'],
        loss = dict_args['loss'],
        model= dict_args['model'],
        model_kwargs = dict_args['model_kwargs'],    
        training_kwargs=dict_args['training_kwargs'],
        optimizer=dict_args['optimizer'],
        optimizer_kwargs=dict_args['optimizer_kwargs'],
        stopper='early',
        stopper_kwargs={'frequency':50, 'patience':2, 'relative_delta':0.002},
        random_seed=random_seed
    )
    """

    logger.info(f'start learning embedding in random seed {dict_args.get("random_seed")}')  
    
    pipeline_result = pipeline(**dict_args)

    hits_at_10 = pipeline_result.get_metric('hits_at_10')
    print('Hits@10\t' + str(hits_at_10))

    if dir_save != None:
        pipeline_result.save_to_directory(dir_save)

    logger.info(f'finish learning embedding in random seed {dict_args.get("random_seed")}') 
        
    return pipeline_result


# ## parameters

# In[4]:


# for input
# ----------------------------------------------------
## a path to the data fra
f_params = '../benchmarking/df_best_param.pkl'
## a name of knowledge graph embedding model
model_name = 'transe'
## a data set (knowledge graph)
dataset_name = 'fb15k237'
## a list of random seeds which should be unique for each other
list_random_seeds = [0, 1, 2, 3]
## a number of epochs (if None, using best value)
num_epochs = 2
##  a maiximum number of multi processing
num_process = 1

# for output
# ----------------------------------------------------
## a direcory where learned model are saved
del_previous_result = True
output_name = 'try1'
dir_learned_model = f'./models/20240811/kge_{output_name}_{model_name}_{dataset_name}'


# ## preparation

# In[5]:


if del_previous_result:
    delete_all_files_in_directory(dir_learned_model)
if not os.path.exists(dir_learned_model):
    os.mkdir(dir_learned_model)
db = DataBinder(dir_learned_model)


# ## main

# ### create args.

# In[6]:


dict_base_args = get_best_params(f_params, model_name, dataset_name)


# In[7]:


if num_epochs != None:
    dict_base_args['training_kwargs']['num_epochs'] = num_epochs


# In[8]:


dict_base_args


# In[9]:


list_args= []
for random_seed in list_random_seeds:
    dict_args = copy.deepcopy(dict_base_args)
    dict_args['random_seed'] = random_seed
    list_args.append((dict_args,None,))


# In[10]:


num_cpus = multiprocessing.cpu_count()


# In[ ]:


with multiprocessing.Pool(processes=num_process) as pool:
    results = pool.starmap(learn_embedding, list_args)


# In[ ]:


for result, tuple_args in zip(results, list_args):
    dict_args, dir_save = tuple_args
    random_seed = dict_args['random_seed']
    db.add(f'model_{random_seed}', result.model)


# In[ ]:


db.add('model_name', model_name)
db.add('dataset_name', dataset_name)
db.add('dict_args', dict_args)
db.add('f_params', f_params)
db.add('list_random_seeds', list_random_seeds)


# In[ ]:




