import torch

def vanilla_hits_at_k(kge_model, triples, ks=[1,3,5,10], batch_size=100, output_type='both'):

    dict_isin_at_k = {}
    dict_hits_at_k = {}
    
    dict_isin_at_k['tail'] = {k:[] for k in ks}
    dict_isin_at_k['head'] = {k:[] for k in ks}

    n_all = triples.shape[0]
    batch_size = 100

    for _type in ['head', 'tail']:
        
        for i in range(0,n_all,batch_size):
            
            j = min(i+batch_size,n_all)

            if _type == 'tail':
                # a list of tails
                nids = triples[i:j,2]
                # a list of score(h,r,*)
                scores = kge_model.score_t(triples[i:j,:2])
            else:
                # a list of heads
                nids = triples[i:j,0]
                # a list of score(*,r,t)
                scores = kge_model.score_h(triples[i:j,1:])

            # for each head/tail
            for nid, _scores in zip(nids, scores):
                # sort scores
                _sorted_nid = torch.argsort(_scores,descending=True)
                # judge if head/tail is in top k
                for k in ks:
                    dict_isin_at_k[_type][k].append((nid in _sorted_nid[:k]))

        dict_hits_at_k[_type] = {}
        for k, list_isin in dict_isin_at_k[_type].items():
            dict_hits_at_k[_type][k] = sum(list_isin)/len(list_isin)

    dict_hits_at_k['both'] = {}
    for k in [1,3,5,10]:
        v1 = dict_hits_at_k['tail'][k]
        v2 = dict_hits_at_k['head'][k]
        dict_hits_at_k['both'][k] = (v1 + v2)/2.0   

    if output_type == None:
        return dict_hits_at_k
    else:
        return dict_hits_at_k[output_type]