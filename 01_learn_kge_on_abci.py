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
import time
import os
import logging
import shutil
import multiprocessing
import yaml
import pandas as pd
import numpy as np
import textwrap
import torch
from uuid import uuid1
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

            if k not in ['label_smoothing']:
                dict_args['training_kwargs'][k.split('.')[-1]] = v
            #dict_args['training_kwargs'][k.split('.')[-1]] = v

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
    
    logger.info(f'start learning embedding in random seed {dict_args.get("random_seed")}')  
    
    pipeline_result = pipeline(**dict_args)

    hits_at_10 = pipeline_result.get_metric('hits_at_10')
    print('Hits@10\t' + str(hits_at_10))

    if dir_save != None:
        pipeline_result.save_to_directory(dir_save)

    logger.info(f'finish learning embedding in random seed {dict_args.get("random_seed")}') 
        
    return pipeline_result



class JobManager:
    
    def __init__(self, 
                 dir_working:str, 
                 h_rt:str='00:10:00',
                 monitoring_period:float=10.0):
        
        self.dir_working = dir_working
        self.h_rt = h_rt
        self.monitoring_period = monitoring_period
        
    def execute(self, dict_args, list_random_seeds=[0]):

        list_args= []
        for random_seed in list_random_seeds:
            _dict_args = copy.deepcopy(dict_args)
            _dict_args['random_seed'] = random_seed
            list_args.append((random_seed,_dict_args))
            
        list_wd = self.submit_jobs(list_args)
        list_wd = self.monitor(list_wd)

        return self.retrive_results(list_wd)

    def submit_jobs(self, list_args):
        
        list_wd = []
        
        for job_id, dict_args in list_args:
            list_wd.append(self._submit_job(dict_args, f'{self.dir_working}/{job_id}'))
        
        return list_wd
            
    def _submit_job(self, dict_args, wd):

        if not os.path.exists(wd):
            os.mkdir(wd)

        with open(f'{wd}/params.yml', 'w') as fout:
            yaml.safe_dump(dict_args, fout)
            
        script = textwrap.dedent(f"""
        #!/bin/bash
        
        # options for batch job execution.
        # for all options, please check https://docs.abci.ai/ja/job-execution/#job-execution-options 
        
        #$ -l rt_G.small=1
        #$ -l h_rt={self.h_rt}
        #$ -m a
        #$ -m b
        #$ -m e
        #$ -j y
        #$ -o {wd}/log.txt
        #$ -cwd
        
        source /etc/profile.d/modules.sh
        source /home/acg16558pn/kg_20240423/bin/activate
        module load cuda/12.1
        module load python/3.10
        python script_pykeen_pipeline.py -i {wd}/params.yml -o {wd}/result
        """)
    
        with open(f'{wd}/script.sh', 'w') as fout:
            fout.write(script)
    
        os.system(f'qsub -g gcc50441 {wd}/script.sh') 
    
        return wd

    def monitor(self, list_wd):
        while True:
            
            list_executing_wd = []
            list_finished_wd  = []
            
            for wd in list_wd:
                if os.path.exists(f'{wd}/result'):
                    list_finished_wd.append(wd)
                else:
                    list_executing_wd.append(wd)
            
            if set(list_wd) == set(list_finished_wd):
                logger.info('all jobs have finished.')
                return list_finished_wd
            else:
                message = 'still executing in '
                for wd in list_executing_wd:
                    message += f'{wd}, '    
                logger.info(message)
    
            time.sleep(self.monitoring_period)

    def retrive_results(self, list_wd):

        dict_results = {}
        for wd in list_wd:
            job_id = os.path.basename(wd)
            dict_results[job_id] = {'model':torch.load(f'{wd}/result/trained_model.pkl')}

        return dict_results


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
list_random_seeds = list(range(10))
## a number of epochs (if None, using best value)
num_epochs = None
## a maximum time of computation
h_rt = '01:00:00'

# for output
# ----------------------------------------------------
## a direcory where learned model are saved
del_previous_result = True
output_name = 'try3'
dir_learned_model = f'./models/20240812/kge_{output_name}_{model_name}_{dataset_name}'


# ## preparation

# In[5]:


if del_previous_result:
    delete_all_files_in_directory(dir_learned_model)
if not os.path.exists(dir_learned_model):
    os.mkdir(dir_learned_model)
db = DataBinder(dir_learned_model)


# ## main

# In[6]:


dict_args = get_best_params(f_params, model_name, dataset_name)


# In[7]:


if num_epochs != None:
    dict_args['training_kwargs']['num_epochs'] = num_epochs


# In[8]:


dict_args


# In[9]:


jm = JobManager(dir_working=dir_learned_model, 
                h_rt=h_rt, monitoring_period=60)


# In[ ]:


dict_results = jm.execute(dict_args, list_random_seeds)


# In[ ]:


for random_seed, dict_result in dict_results.items():
    db.add(f'model_{random_seed}', dict_result['model'])


# In[ ]:


db.add('model_name', model_name)
db.add('dataset_name', dataset_name)
db.add('dict_args', dict_args)
db.add('f_params', f_params)
db.add('list_random_seeds', list_random_seeds)

