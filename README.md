# Analysis-of-conventional-refinement-method-for-knowledge-graph

## Getting Started
+ CUDA: 12.1
+ Python: 3.10
+ Pytorch: 2.1.2
+ Pystow: 0.4.9
+ Pykeen: 1.10.0


## Usage
### 1. 知識グラフ埋込モデルの学習  
#### 1-1. インタラクティブ
```python
from model import KnowledgeGraphEmbedding

dir_save = '/path/to/dir_save'
kge = KnowledgeGraphEmbedding(dir_save=dir_save, **kwargs)

# dict_paramsはmode idがキー，pykeenへ渡すパラメータ（辞書型）が値の辞書
kge.learn(dict_params, list_random_seeds)

# 学習済みモデルの取得
kge.get('model_id', 'trained_model')
# 学習済みモデルに関する情報
kge.get('model_id', 'results')
# 保存場所
kge.get('model_id', 'dir_save')
```

#### 1-2. バッチ  
**ジョブの投入**
```python
from model import KnowledgeGraphEmbedding

dir_save = '/path/to/dir_save'
kge = KnowledgeGraphEmbedding(dir_save=dir_save, **kwargs)

# dict_paramsはjob idがキー，pykeenへ渡すパラメータ（辞書型）が値の辞書
kge.submit_jobs(dict_params, list_random_seeds)
```

**ジョブの監視と結果の取得**
```python
kge = KnowledgeGraphEmbedding.form_directory(dir_save)
kge.monitoring_jobs()
```

### 2. 誤りを含む知識グラフの作成

```python
from dataset import KnowledgeGraph

kg = KnowledgeGraph(dataset='name_of_dataset')

# 以下の方法でも初期化可能.
# tf_train, tf_test, tf_validは，それぞれ，教師データ，テストデータ，検証データの
# TripleFactory
# kg = KnowledgeGraph(training=tf_train, testing=tf_test, validation=tf_valid)

# 誤りを含む知識グラフの作成．
# 基本的にはテストデータをベースに誤りを含むトリプルを作成する．
# dict_false_triplesはキーがランダムシードで，値が辞書型になっており，
# その辞書型のキーは'triple_factory'と'dataframe'で，それぞれ，
# 誤りを含むtripleのTripleFactoryと，その特徴量がdataframeを値として持つ．
dict_false_triples = kg.make_false_dataset(ratio=0.1, random_seeds=[0,1,2])
```

### 3. スコアの計算

```python
from model import KnowledgeGraphEmbedding
from dataset import KnowledgeGraph
from evaluation import Evaluator

kge = KnowledgeGraphEmbedding.from_directory('/path/to/dir_model')

dict_scores = {}
for random_seed, false_triples in dict_false_triples:
    dict_scores[random_seed] = kge.score_hrt(false_triples)

dict_results = {}
for data_random_seed, _dict_scores in dict_scores.items():
    dict_results[data_random_seed] = {}
    for model_random_seed, scores in _dict_scores.items():
        dict_results[data_random_seed][model_random_seed] = evaluate_false_score(scores)
```