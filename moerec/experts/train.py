"""Expert training (paper Section 3.2 backbones, upstream trainer defaults where the paper is silent).
  sasrec  hidden 64, 2 heads, dropout 0.1, max length 50, Adam lr 1e-3 wd 1e-4, batch 64, 50 epochs; full-softmax cross-entropy over the
          catalogue; best validation CE kept (sasrec_best.pt). Validation = the last item of 10% of the training users.
  gcn     LightGCN, 64 dims, 3 layers (mean of the 4 layer outputs), BPR with one uniform negative, L2 1e-3, Adam lr 1e-3, batch 1024,
          500 epochs; the held-out pairs are never sampled as negatives; item embeddings saved to gcn_items.pt.
Both resume from their own checkpoints (sasrec-resume.pt / gcn-resume.pt). Seed: --seed (303 = the main run; 304, 305 = seed variance).
Usage: python -m moerec.experts.train sasrec|gcn <dataset or fold dir> [--gpu 0 | --cpu] [--seed 303]"""
import argparse,os,time,json,traceback
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['sasrec','gcn']);parser.add_argument('exp');parser.add_argument('--gpu',default='0');parser.add_argument('--cpu',action='store_true')
parser.add_argument('--seed',type=int,default=int(os.environ.get('EXPERT_SEED','303')))
parser.add_argument('--epochs',type=int,default=0,help='override the epoch count (smoke tests)');a=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='' if a.cpu else a.gpu
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np, torch
from moerec.runtime import sha256 as sha,save_json,configure_torch,seed_all,load_data,seq_at,make_sasrec
EXP=Path(a.exp);SEED=a.seed
def save(name,value):save_json(EXP/name,value)

def sasrec(device):
    hist,users,names=load_data(EXP);samples=np.load(EXP/'data/samples.npz');N=len(names)
    class Samples(torch.utils.data.Dataset):
        def __init__(self,split):self.us=samples[split+'_uid'];self.pos=samples[split+'_pos']
        def __len__(self):return len(self.us)
        def __getitem__(self,i):
            u=int(self.us[i]);pos=int(self.pos[i]);seq=seq_at(hist,u,pos)
            return torch.from_numpy(seq),int(hist['items'][int(hist['offsets'][u])+pos])
    train=Samples('train');val=Samples('val');model=make_sasrec(N,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    # Paper omits backbone epochs/batch/heads; use author trainer defaults, paper dimension/dropout.
    batch=64;epochs=a.epochs or 50;best=float('inf');start=time.monotonic();records=[];start_epoch=0
    latest=EXP/'sasrec-resume.pt'
    if latest.exists():
        c=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);start_epoch=c['epoch'];best=c['best_val'];records=c['records'];torch.set_rng_state(c['rng'].cpu())
    loader=torch.utils.data.DataLoader(train,batch_size=batch,shuffle=True,num_workers=0)
    val_loader=torch.utils.data.DataLoader(val,batch_size=batch,shuffle=False,num_workers=0)
    # Fail before a long job if candidate and full-catalog score formulas disagree.
    model.eval();x,_=next(iter(val_loader));x=x[:2].to(device);ids=torch.tensor([[1,2,3],[4,5,6]],device=device)
    with torch.no_grad():assert torch.allclose(model(x).gather(1,ids),model.candidate_scores(x,ids),atol=2e-5,rtol=2e-5)
    seed_all(SEED+start_epoch)
    for epoch in range(start_epoch,epochs):
        t0=time.monotonic();model.train();total=0.;n=0
        for step,(x,y) in enumerate(loader):
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(y);n+=len(y)
            if step%250==0:
                save('sasrec-state.json',{'stage':'training','epoch':epoch+1,'step':step,'total_steps':len(loader),'train_rows':len(train),'loss':total/max(n,1),'elapsed_seconds':time.monotonic()-start})
                print('SASREC_STEP',epoch+1,step,len(loader),round(total/n,5),flush=True)
        model.eval();vt=0.;vn=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device);y=y.to(device);loss=torch.nn.functional.cross_entropy(model(x),y);vt+=float(loss)*len(y);vn+=len(y)
        row={'epoch':epoch+1,'train_ce':total/n,'val_ce':vt/vn,'seconds':time.monotonic()-t0};records.append(row);print('SASREC_EPOCH',json.dumps(row),flush=True)
        if row['val_ce']<best:
            best=row['val_ce'];torch.save({'model_state_dict':model.state_dict(),'epoch':epoch+1,'val_loss':best,'item_num':N,'hidden_size':64,'dropout':.1,'heads':2,'paper_inner_product_item_embedding':True},EXP/'sasrec_best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best_val':best,'records':records,'rng':torch.get_rng_state()},latest)
        save('sasrec-training.json',{'status':'RUNNING','epochs':records,'best_val':best})
    save('sasrec-training.json',{'status':'PASS','epochs':records,'best_val':best,'checkpoint_sha256':sha(EXP/'sasrec_best.pt'),'device':str(device),'gpu_peak_allocated':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
    save('sasrec-state.json',{'stage':'complete','epochs':epochs});(EXP/'sasrec.exit-code').write_text('0\n')

def gcn(device):
    import scipy.sparse as sp
    graph=np.load(EXP/'data/graph.npz');shape=tuple(graph['shape']);n_users=int(graph['n_users']);N=int(graph['item_num'])
    A=torch.sparse_coo_tensor(torch.from_numpy(graph['indices']),torch.from_numpy(graph['values']),shape).coalesce().to(device)
    train=sp.load_npz(EXP/'data/train_user_item.npz').tocsr();active=np.flatnonzero(np.diff(train.indptr)>0)
    hist=np.load(EXP/'data/histories.npz');exclude={int(u):int(i)-1 for u,i in hist['heldout_pairs']}
    weight=torch.nn.Embedding(shape[0],64,device=device);torch.nn.init.normal_(weight.weight,std=.1)
    optimizer=torch.optim.Adam(weight.parameters(),lr=.001,foreach=False)
    epochs=a.epochs or 500;batch=1024;steps=max(1,len(active)//batch);rng=np.random.default_rng(SEED);records=[];start_epoch=0;resume=EXP/'gcn-resume.pt'
    if resume.exists():
        c=torch.load(resume,map_location=device,weights_only=False);weight.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);start_epoch=c['epoch'];records=c['records'];rng.bit_generator.state=c['rng']
    class LinearPropagation(torch.autograd.Function):
        @staticmethod
        def forward(ctx,emb,adj):
            ctx.save_for_backward(adj)
            x=emb;total=emb/4
            for _ in range(3):x=torch.sparse.mm(adj,x);total.add_(x,alpha=.25)
            return total
        @staticmethod
        def backward(ctx,grad):
            adj,=ctx.saved_tensors;adj=adj.transpose(0,1)
            x=grad;total=grad/4
            for _ in range(3):x=torch.sparse.mm(adj,x);total.add_(x,alpha=.25)
            return total,None
    # Exact linear derivative, checked against ordinary autograd before the full graph.
    small=torch.eye(7,dtype=torch.float32).to_sparse().to(device)
    z=torch.randn(7,3,device=device,requires_grad=True);q=torch.randn_like(z)
    terms=[z]
    for _ in range(3):terms.append(torch.sparse.mm(small,terms[-1]))
    ref=sum(terms)/4;g1=torch.autograd.grad((ref*q).sum(),z)[0]
    actual=LinearPropagation.apply(z,small);g2=torch.autograd.grad((actual*q).sum(),z)[0]
    assert torch.allclose(ref,actual,atol=1e-6) and torch.allclose(g1,g2,atol=1e-6)
    save('gcn-gradient-check.json',{'status':'PASS','forward_max_abs':float((ref-actual).abs().max()),'gradient_max_abs':float((g1-g2).abs().max()),'method':'exact linear three-hop sparse propagation adjoint; no model approximation'})
    def propagate():return LinearPropagation.apply(weight.weight,A)
    for epoch in range(start_epoch,epochs):
        total=0.;t0=time.monotonic()
        for step in range(steps):
            us=rng.choice(active,batch);ps=[];ns=[]
            for u in us:
                pos=train.indices[train.indptr[u]:train.indptr[u+1]];ps.append(int(rng.choice(pos)))
                while True:
                    j=int(rng.integers(0,N-1))
                    if j!=exclude.get(int(u),-1) and not np.any(pos==j):ns.append(j);break
            u=torch.tensor(us,device=device);pi=torch.tensor(ps,device=device)+n_users;ni=torch.tensor(ns,device=device)+n_users
            optimizer.zero_grad(set_to_none=True);final=propagate();u1=final[u];p1=final[pi];n1=final[ni]
            bpr=-torch.nn.functional.logsigmoid((u1*p1).sum(1)-(u1*n1).sum(1)).mean()
            reg=(weight(u).pow(2).sum()+weight(pi).pow(2).sum()+weight(ni).pow(2).sum())/(2*batch)
            loss=bpr+.001*reg;assert torch.isfinite(loss);loss.backward();optimizer.step();total+=float(loss.detach())
        row={'epoch':epoch+1,'bpr_plus_reg':total/steps,'seconds':time.monotonic()-t0};records.append(row)
        if (epoch+1)%5==0 or epoch==0:
            print('GCN_EPOCH',json.dumps(row),flush=True);save('gcn-state.json',{'stage':'training',**row,'epochs':epochs})
            torch.save({'model':weight.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'records':records,'rng':rng.bit_generator.state},resume)
            save('gcn-training.json',{'status':'RUNNING','epochs':records})
    with torch.no_grad():
        e=propagate();items=torch.zeros(N,64,device=device);items[1:]=e[n_users:]
        torch.save({'item_emb':items.cpu(),'n_users':n_users,'epochs':epochs,'loss':'bpr','recency_lambda':.1,'user_representation':'paper Eq4 history weighted raw item embeddings'},EXP/'gcn_items.pt')
    save('gcn-training.json',{'status':'PASS','epochs':records,'checkpoint_sha256':sha(EXP/'gcn_items.pt'),'device':str(device),'gpu_peak_allocated':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0});save('gcn-state.json',{'stage':'complete','epochs':epochs});(EXP/'gcn.exit-code').write_text('0\n')

if __name__=='__main__':
    try:
        seed_all(SEED);device=configure_torch(None if a.cpu else a.gpu,fraction=.09,threads=8 if a.cpu else 4)
        if a.kind=='sasrec':sasrec(device)
        else:gcn(device)
    except BaseException as e:
        traceback.print_exc();save(a.kind+'-failure.json',{'error':str(e),'kind':type(e).__name__});(EXP/(a.kind+'.exit-code')).write_text('1\n');raise
