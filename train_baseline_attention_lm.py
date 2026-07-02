#!/usr/bin/env python3
"""Fair dense attention-LM baseline using the same byte tokenizer/data path.

This baseline now uses SDPA/fused causal attention when available so its training
profile is a fairer comparison against NEED and SmolLM-style sidecar settings.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Optional, Sequence
import torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from need_core import ByteTokenizer, Special, resolve_device
from train import TextChunkDataset, collate
from sidecar_attention_kernels import configure_torch_attention, fused_causal_attention

class BaselineConfig:
    def __init__(self, vocab_size=Special.text_vocab, block_size=256, d_model=256, n_layers=4, n_heads=4, d_ff=1024, dropout=0.1, kernel_backend="auto"):
        self.__dict__.update(locals()); del self.__dict__['self']

class FusedAttentionBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.n_heads=cfg.n_heads; self.head_dim=cfg.d_model//cfg.n_heads
        self.ln1=nn.LayerNorm(cfg.d_model); self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False); self.o=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
        self.ln2=nn.LayerNorm(cfg.d_model); self.ff=nn.Sequential(nn.Linear(cfg.d_model,cfg.d_ff),nn.SiLU(),nn.Linear(cfg.d_ff,cfg.d_model),nn.Dropout(cfg.dropout))
    def forward(self,x):
        b,t,d=x.shape; h=self.ln1(x); qkv=self.qkv(h).view(b,t,3,self.n_heads,self.head_dim); q,k,v=qkv.unbind(dim=2)
        q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)
        y=fused_causal_attention(q,k,v,dropout_p=0.0 if not self.training else self.cfg.dropout,backend=self.cfg.kernel_backend).transpose(1,2).reshape(b,t,d)
        x=x+self.o(y); return x+self.ff(self.ln2(x))

class AttentionBaselineLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.emb=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.block_size,cfg.d_model); self.blocks=nn.ModuleList([FusedAttentionBlock(cfg) for _ in range(cfg.n_layers)]); self.ln=nn.LayerNorm(cfg.d_model); self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.head.weight=self.emb.weight
    def forward(self,x,y=None):
        b,t=x.shape; pos=torch.arange(t,device=x.device)[None]; h=self.emb(x)+self.pos(pos)
        for blk in self.blocks: h=blk(h)
        logits=self.head(self.ln(h)); loss=None
        if y is not None: loss=F.cross_entropy(logits.reshape(-1,self.cfg.vocab_size), y.reshape(-1), ignore_index=Special.pad)
        return logits, loss

def main(argv: Optional[Sequence[str]]=None):
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out_dir',default='baseline_out'); p.add_argument('--device',default='auto'); p.add_argument('--block_size',type=int,default=256); p.add_argument('--d_model',type=int,default=256); p.add_argument('--n_layers',type=int,default=4); p.add_argument('--n_heads',type=int,default=4); p.add_argument('--d_ff',type=int,default=1024); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--max_steps',type=int,default=1000); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--log_interval',type=int,default=20); p.add_argument('--kernel_backend',choices=['auto','torch','flash_attn'],default='auto'); p.add_argument('--amp',choices=['off','bf16','fp16'],default='bf16'); p.add_argument('--compile',action='store_true')
    args=p.parse_args(argv); dev=resolve_device(args.device); configure_torch_attention(True,True,True)
    tok=ByteTokenizer(); ids=tok.encode(Path(args.data).read_text(encoding='utf-8',errors='replace'),add_bos=True,add_eos=True); ds=TextChunkDataset(ids,args.block_size,1,args.max_steps*args.batch_size+args.batch_size); dl=DataLoader(ds,batch_size=args.batch_size,collate_fn=collate)
    cfg=BaselineConfig(block_size=args.block_size,d_model=args.d_model,n_layers=args.n_layers,n_heads=args.n_heads,d_ff=args.d_ff,kernel_backend=args.kernel_backend); model=AttentionBaselineLM(cfg).to(dev)
    if args.compile and hasattr(torch,'compile'):
        try: model=torch.compile(model,mode='reduce-overhead')
        except Exception: pass
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr); step=0; t0=time.time(); dtype=torch.bfloat16 if args.amp=='bf16' and dev.type=='cuda' else torch.float16 if args.amp=='fp16' and dev.type=='cuda' else torch.float32
    for b in dl:
        if step>=args.max_steps: break
        x=b['input_ids'].to(dev); y=b['targets'][...,0].to(dev)
        with torch.autocast(device_type='cuda',dtype=dtype,enabled=(dev.type=='cuda' and args.amp!='off')):
            _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step%args.log_interval==0: print(json.dumps({'step':step,'loss':float(loss.detach().cpu()),'tok_s':x.numel()/max(time.time()-t0,1e-9),'kernel_backend':args.kernel_backend})); t0=time.time()
        step+=1
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({'config':cfg.__dict__,'state_dict':model.state_dict()},out/'model.pt')
if __name__=='__main__': main()
