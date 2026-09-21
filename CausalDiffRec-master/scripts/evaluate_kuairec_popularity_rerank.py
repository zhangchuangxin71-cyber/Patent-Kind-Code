#!/usr/bin/env python
"""Validation-selected popularity penalty for KuaiRec frozen checkpoints."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modules.rec_model import LGCN_Encoder
from utils.load_strict import load_strict_datasets,user_item_matrix_from_graph
from utils.util_loss import generate_interaction_matrix_from_dgl,normalize_graph_mat

def ndcg(scores, users, gt, train, n_user, k=20):
    scores=scores.clone()
    for r,u in enumerate(users):
        s,e=train.indptr[u:u+2]; scores[r,torch.as_tensor(train.indices[s:e],device=scores.device)]=-torch.inf
    top=torch.topk(scores,k,dim=1).indices.cpu().numpy(); vals=[]
    for r,u in enumerate(users):
        truth=gt[int(u)]; dcg=sum(1/math.log2(i+2) for i,x in enumerate(top[r]) if int(x) in truth)
        idcg=sum(1/math.log2(i+2) for i in range(min(k,len(truth)))); vals.append(dcg/idcg if idcg else 0.)
    return float(np.mean(vals))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_root',required=True); ap.add_argument('--source_records',nargs='+',required=True); ap.add_argument('--out',required=True); ap.add_argument('--gammas',default='0,0.001,0.0025,0.005,0.01,0.02,0.05,0.1'); args=ap.parse_args()
    ds,meta=load_strict_datasets('kuairec',args.data_root,'ood',load_test_gt=True); n_user,n_item=int(meta['n_user']),int(meta['n_item'])
    train,_,_=user_item_matrix_from_graph(ds['train'],n_user,n_item); deg=np.asarray(train.sum(axis=0)).ravel(); penalty=torch.as_tensor(np.log1p(deg),dtype=torch.float32)
    norm=normalize_graph_mat(generate_interaction_matrix_from_dgl(ds['train'],n_user,n_item)); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); penalty=penalty.to(device)
    val_users=sorted(ds['val_user_set']); test_users=sorted(ds['test_user_set']); val_gt={int(u):{int(i)-n_user for i in x} for u,x in ds['val_origin_inter'].items()}; test_gt={int(u):{int(i)-n_user for i in x} for u,x in ds['test_origin_inter'].items()}
    gammas=[float(x) for x in args.gammas.split(',')]
    records={}
    for rp in args.source_records:
        src=json.load(open(rp)); st=src['checkpoint']; ck=torch.load(st,map_location='cpu')['rec_model']; model=LGCN_Encoder(n_user,int(src['settings'].get('lgcn_layers',3)),norm,ck['embedding_dict.user_emb'],ck['embedding_dict.item_emb']).to(device); model.load_state_dict(ck); model.eval()
        with torch.no_grad():
            e=model(); all_scores=e[:n_user]@e[n_user:].t();
            records[str(int(src['settings']['seed']))]={'val':{},'ood':{}}
            for g in gammas:
                records[str(int(src['settings']['seed']))]['val'][str(g)]=ndcg(all_scores[torch.as_tensor(val_users,device=device)]-g*penalty,val_users,val_gt,train,n_user)
                records[str(int(src['settings']['seed']))]['ood'][str(g)]=ndcg(all_scores[torch.as_tensor(test_users,device=device)]-g*penalty,test_users,test_gt,train,n_user)
    sel=[1024,2048,3072]; conf=[4096,5120]
    mean=lambda split,g,seeds: float(np.mean([records[str(s)][split][str(g)] for s in seeds]))
    best=max(gammas,key=lambda g:mean('val',g,sel)); summary={'dataset':'kuairec','protocol_version':'corrected_v4_popularity_rerank','gammas':gammas,'selected_gamma':best,'selection_val_ndcg20':mean('val',best,sel),'selection_baseline':mean('val',0.,sel),'confirmation_val_ndcg20':mean('val',best,conf),'confirmation_baseline':mean('val',0.,conf),'ood_ndcg20':mean('ood',best,sel+conf),'ood_baseline':mean('ood',0.,sel+conf),'ood_gain':mean('ood',best,sel+conf)-mean('ood',0.,sel+conf),'checks':{'positive_gamma':best>0,'selection_beats_baseline':mean('val',best,sel)>mean('val',0.,sel)}}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump({'summary':summary,'per_seed':records},open(args.out,'w'),indent=2); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
