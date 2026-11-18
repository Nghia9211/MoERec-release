"""Paper Sec. 4.1.2 data preparation for Goodreads (the first dataset of the study; same rules as prepare_arb.py).
Inputs from the upstream MoERec repository (--upstream): MoE/data/goodreads/{id2rawid,id2name}.txt (item vocabulary of the
upstream checkpoints), MoE/data/ground_truth.json, dataset/tasks5/<scenario>/goodreads (AgentRecBench tasks) and
Dataset/merged_all/review.json. Writes <out>/goodreads/data/. Item texts and ratings: python -m moerec.data.stars goodreads ...
Usage: python -m moerec.data.prepare_goodreads --upstream <MoERec repo> --out <work dir>"""
from pathlib import Path
import argparse,os,json,time,collections,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from moerec.runtime import sha256 as sha, save_json
_ap=argparse.ArgumentParser();_ap.add_argument('--upstream',required=True);_ap.add_argument('--out',required=True);_A=_ap.parse_args()
ROOT=Path(_A.upstream);EXP=Path(_A.out)/'goodreads'
def save(name,value):save_json(EXP/name,value)

def user_frequencies():
    """Goodreads interactions per raw user id in the merged AgentRecBench review file (cached in <out>/goodreads/user_frequencies.json)."""
    p=EXP/'user_frequencies.json'
    if not p.exists():
        c=collections.Counter(str(json.loads(l)['user_id']) for l in (ROOT/'Dataset/merged_all/review.json').open() if '"goodreads"' in l and json.loads(l).get('source')=='goodreads')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(c))
    return json.loads(p.read_text())

def main():
    start=time.monotonic();out=EXP/'data';out.mkdir(exist_ok=True)
    freq=user_frequencies();uids=sorted(u for u,n in freq.items() if n>3);u2i={u:i for i,u in enumerate(uids)}
    original=ROOT/'MoE/data/goodreads'
    pairs=[x.split('::',1) for x in (original/'id2rawid.txt').read_text().splitlines() if '::' in x]
    raw2inner={v.strip():int(k) for k,v in pairs};inner2raw={int(k):v.strip() for k,v in pairs};item_num=max(inner2raw)+1
    names={int(k):v.strip() for k,v in [x.split('::',1) for x in (original/'id2name.txt').read_text().splitlines() if '::' in x]}
    (out/'item_names.json').write_text(json.dumps([names.get(i,'') for i in range(item_num)],ensure_ascii=False));(out/'id2rawid.json').write_text(json.dumps([inner2raw.get(i,'') for i in range(item_num)]));(out/'users.json').write_text(json.dumps(uids))
    all_gt=json.loads((ROOT/'MoE/data/ground_truth.json').read_text());test={str(x['user_id']):str(x['item_id']).strip() for x in all_gt if str(x['user_id']) in u2i};assert len(test)==4714
    # Preserve supplied test membership (~10%) rather than invent replacement targets.
    # The paper omits bin boundaries/seed; these inference bins are frozen implementation choices.
    bins=collections.defaultdict(list)
    for u in uids:
        if u in test:continue
        n=freq[u];key=0 if n<=6 else 1 if n<=10 else 2 if n<=20 else 3 if n<=50 else 4
        bins[key].append(u)
    infer=set();rng=np.random.default_rng(303);target_n=int(len(uids)*.05)
    quotas={k:len(v)*target_n/sum(map(len,bins.values())) for k,v in bins.items()};counts={k:int(v) for k,v in quotas.items()}
    for k in sorted(quotas,key=lambda x:quotas[x]-counts[x],reverse=True)[:target_n-sum(counts.values())]:counts[k]+=1
    for k,us in sorted(bins.items()):infer.update(np.asarray(us)[rng.permutation(len(us))[:counts[k]]].tolist())
    assert not infer & set(test) and len(infer)==target_n
    histories=[[] for _ in uids];processed=0;bad_date=0
    for rowno,line in enumerate((ROOT/'Dataset/merged_all/review.json').open()):
        d=json.loads(line)
        if d.get('source')!='goodreads' or str(d['user_id']) not in u2i:continue
        u=str(d['user_id']);rid=str(d['item_id']);date=d.get('date_added') or d.get('date_updated')
        try:ts=int(datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %z %Y').timestamp())
        except Exception:ts=0;bad_date+=1
        histories[u2i[u]].append((ts,rowno,raw2inner[rid]));processed+=1
        if processed%200000==0:print('READ',processed,round(time.monotonic()-start,1),flush=True)
    for h in histories:h.sort()
    masks={u:raw2inner[i] for u,i in test.items()}
    for u in infer:masks[u]=histories[u2i[u]][-1][2]
    removed=0;held_pairs=[]
    for u,iid in masks.items():
        old=histories[u2i[u]];clean=[x for x in old if x[2]!=iid];removed+=len(old)-len(clean);histories[u2i[u]]=clean;held_pairs.append([u2i[u],iid])
        assert all(x[2]!=iid for x in clean)
    offsets=[0];flat=[]
    for h in histories:flat.extend(x[2] for x in h);offsets.append(len(flat))
    np.savez_compressed(out/'histories.npz',items=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64),heldout_pairs=np.asarray(held_pairs,dtype=np.int32))
    # One held-out observed training interaction per10% training users for SASRec model selection.
    train_users=[u for u in uids if u not in masks];order=rng.permutation(len(train_users));val_users=set(np.asarray(train_users)[order[:int(len(train_users)*.1)]].tolist())
    train_uid=[];train_pos=[];val_uid=[];val_pos=[];gate_rows=[]
    for ui,u in enumerate(uids):
        ids=[x[2] for x in histories[ui]];n=len(ids)
        val_item=ids[-1] if u in val_users and n>=2 else None
        if val_item is not None:val_uid.append(ui);val_pos.append(n-1)
        stride=1 if n<15 else 2 if n<50 else 3
        for t in range(1,n,stride):
            if val_item is not None and ids[t]==val_item:continue
            idx=len(train_uid);train_uid.append(ui);train_pos.append(t)
            if u not in masks:gate_rows.append(idx)
    np.savez_compressed(out/'samples.npz',train_uid=np.asarray(train_uid,np.int32),train_pos=np.asarray(train_pos,np.int32),val_uid=np.asarray(val_uid,np.int32),val_pos=np.asarray(val_pos,np.int32),gate_rows=np.asarray(gate_rows,np.int64))
    # Graph is deduplicated only at the edge level, matching nx.Graph upstream.
    import scipy.sparse as sp
    n_users=len(uids);n_items=item_num-1;u_index=np.repeat(np.arange(n_users),np.diff(offsets));i_index=np.asarray(flat,np.int64)-1
    csr=sp.coo_matrix((np.ones(len(flat),np.float32),(u_index,i_index)),shape=(n_users,n_items)).tocsr();csr.data[:]=1;csr.eliminate_zeros()
    for ui,iid in held_pairs:assert csr[ui,iid-1]==0,'Heldout target found in graph'
    sp.save_npz(out/'train_user_item.npz',csr)
    row,col=csr.nonzero();rows=np.r_[row,n_users+col];cols=np.r_[n_users+col,row];nodes=n_users+n_items
    A=sp.coo_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(nodes,nodes)).tocsr()+sp.eye(nodes,dtype=np.float32,format='csr')
    degree=np.asarray(A.sum(1)).ravel();D=sp.diags(1/np.sqrt(degree));norm=(D@A@D).tocoo()
    np.savez_compressed(out/'graph.npz',indices=np.vstack([norm.row,norm.col]).astype(np.int64),values=norm.data.astype(np.float32),shape=np.asarray(norm.shape),n_users=n_users,item_num=item_num)
    tasks=[];removed_tasks=[]
    for scenario in ['classic','user_cold_start']:
        base=ROOT/'dataset/tasks5'/scenario/'goodreads'
        for f in sorted((base/'tasks').glob('task_*.json'),key=lambda p:int(p.stem[5:])):
            d=json.loads(f.read_text());d=d[0] if isinstance(d,list) else d;uid=str(d['user_id'])
            if uid not in u2i:removed_tasks.append({'scenario':scenario,'file':f.name,'uid':uid,'raw_interactions':freq[uid]});continue
            label=base/'groundtruth'/f.name.replace('task_','groundtruth_');gt=str(json.loads(label.read_text())['ground truth']).strip();assert test[uid]==gt
            cans=[raw2inner[str(x).strip()] for x in d['candidate_list']];assert len(cans)==len(set(cans))==20 and raw2inner[gt] in cans
            ui=u2i[uid];n=len(histories[ui]);tasks.append({'user_index':ui,'user_id':uid,'target':raw2inner[gt],'target_raw_id':gt,'candidates':cans,'original_scenario':scenario,'scenario':'user_cold_start' if n<=6 else 'classic','visible_history_length':n,'task_file':f.name,'task_sha256':sha(f),'label_sha256':sha(label)})
    assert len(tasks)==2376
    (out/'tasks.json').write_text(json.dumps(tasks));(out/'removed_tasks.json').write_text(json.dumps(removed_tasks,indent=2));(out/'split_users.json').write_text(json.dumps({'test':sorted(test),'inference':sorted(infer),'train':train_users,'sasrec_validation':sorted(val_users)}))
    meta={'status':'PASS','eligible_users':n_users,'item_num_including_padding':item_num,'unique_items':item_num-1,'paper_table_items':239602,'item_count_difference':(item_num-1)-239602,'raw_interactions_after_filter':processed,'visible_interactions_after_mask':len(flat),'masked_occurrences':removed,'heldout_target_pairs':len(masks),'test_users':len(test),'test_fraction':len(test)/n_users,'inference_users':len(infer),'inference_fraction':len(infer)/n_users,'test_inference_overlap':0,'heldout_pairs_in_training_graph':0,'sasrec_training_rows':len(train_uid),'sasrec_validation_rows':len(val_uid),'gate_training_rows':len(gate_rows),'provided_tasks_evaluated_after_filter':len(tasks),'provided_tasks_removed_le3':len(removed_tasks),'paper_scenario_counts':dict(collections.Counter(t['scenario'] for t in tasks)),'original_scenario_counts':dict(collections.Counter(t['original_scenario'] for t in tasks)),'bad_date_rows':bad_date,'graph_edges':int(csr.nnz),'graph_nodes':nodes,'inference_bin_quotas':counts,'sample_stride':'upstream process_data:1 ifN<15,2 ifN<50,3otherwise','scope_note':'Evaluate every eligible provided AgentRecBench task; retain larger supplied ~10percent heldout membership and add disjoint ~5percent inference users. Bin boundaries and exact random seed are unspecified in paper; frozen choices are recorded.','wall_seconds':time.monotonic()-start}
    meta['data_hashes']={f.name:sha(f) for f in out.iterdir() if f.is_file()};save('data-preparation.json',meta);print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
