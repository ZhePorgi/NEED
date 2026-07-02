#!/usr/bin/env python3
"""Evaluation harness for NEED.

Can add learned-image-tokenizer reconstruction metrics to NEED text/image-token CE and latency.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from need_core import ByteTokenizer, load_model, make_image_tokenizer, resolve_device, Special
from train import TextChunkDataset, ImageTokenDataset, collate

try:
    from need_image import load_visual_tokenizer, pil_to_tensor
except Exception:  # pragma: no cover
    load_visual_tokenizer = None  # type: ignore[assignment]
    pil_to_tensor = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def eval_text(model, data_path: Path, device, batches: int, batch_size: int):
    tok = ByteTokenizer()
    ids = tok.encode(data_path.read_text(encoding='utf-8', errors='replace'), add_bos=True, add_eos=True)
    ds = TextChunkDataset(ids, model.cfg.block_size, model.cfg.n_predict_heads, samples=max(batch_size*batches, 1))
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate)
    ces=[]; losses=[]; toks=0; aux_acc={}; t0=time.time(); model.eval()
    with torch.no_grad():
        for i,b in enumerate(dl):
            if i>=batches: break
            x=b['input_ids'].to(device); y=b['targets'].to(device); m=b['image_mask_positions'].to(device) if 'image_mask_positions' in b else None
            it=b.get('image_targets')
            it=it.to(device) if it is not None else None
            _,loss,aux=model(x,y,image_mask_positions=m,image_targets=it)
            if loss is not None: losses.append(float(loss.cpu()))
            if 'ce' in aux: ces.append(float(aux['ce'].cpu()))
            for key in ['compute_fraction','adaptive_effort','memory_boundary','equilibrium_residual','aux_score_risk_mean','faithfulness_mean','image_2d_scan_energy']:
                if key in aux and torch.is_tensor(aux[key]): aux_acc.setdefault(key,[]).append(float(aux[key].detach().cpu()))
            toks += x.numel()
    ce=float(np.mean(ces)) if ces else float('nan')
    out={'text_loss': float(np.mean(losses)) if losses else float('nan'), 'text_ce': ce, 'text_ppl': math.exp(min(20,ce)) if math.isfinite(ce) else float('nan'), 'eval_tok_s': toks/max(1e-9,time.time()-t0)}
    out.update({k: float(np.mean(v)) for k,v in aux_acc.items() if v})
    return out


def eval_image(model, image_dir: Path, device, batches: int, batch_size: int, visual_tokenizer: str = ''):
    ds=ImageTokenDataset(image_dir, model.cfg, samples=max(batch_size*batches, 1), mask_prob=0.5, visual_tokenizer_path=visual_tokenizer, visual_tokenizer_device='cpu')
    dl=DataLoader(ds,batch_size=batch_size,collate_fn=collate)
    losses=[]; t0=time.time(); toks=0
    with torch.no_grad():
        for i,b in enumerate(dl):
            if i>=batches: break
            x=b['input_ids'].to(device); y=b['targets'].unsqueeze(-1).to(device) if b['targets'].ndim==2 else b['targets'].to(device); m=b['image_mask_positions'].to(device)
            it=b.get('image_targets')
            it=it.to(device) if it is not None else None
            _,loss,aux=model(x,y,image_mask_positions=m,image_targets=it)
            if 'image_diffusion' in aux: losses.append(float(aux['image_diffusion'].cpu()))
            elif loss is not None: losses.append(float(loss.cpu()))
            toks += x.numel()
    ce=float(np.mean(losses)) if losses else float('nan')
    return {'image_mask_ce': ce, 'image_tok_s': toks/max(1e-9,time.time()-t0)}


def eval_visual_tokenizer(vt_path: str, image_dir: Path, device, max_images: int = 32, size: int = 256):
    if not vt_path or load_visual_tokenizer is None or Image is None or pil_to_tensor is None:
        return {}
    vt=load_visual_tokenizer(vt_path, device=device)
    paths=[]
    for ext in ('*.png','*.jpg','*.jpeg','*.webp','*.bmp'):
        paths.extend(sorted(image_dir.rglob(ext)))
    vals=[]
    with torch.no_grad():
        for p in paths[:max_images]:
            img=Image.open(p).convert('RGB')
            ids,meta=vt.encode_image(img,add_special=True,device=device)
            dec=vt.decode_tokens(ids,grid=meta.get('grid'),size=size,device=device)
            x=pil_to_tensor(img,size).to(device); y=pil_to_tensor(dec,size).to(device)
            vals.append(float(F.l1_loss(x,y).cpu()))
    return {'visual_tokenizer_l1': float(np.mean(vals)) if vals else float('nan'), 'visual_tokenizer_images': len(vals)}


def latency(model, device, prompt: str, new_tokens: int):
    tok=ByteTokenizer(); ids=torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
    t0=time.time(); out=model.generate_text(ids,max_new_tokens=new_tokens,temperature=0.0); dt=time.time()-t0
    res={'latency_s': dt, 'decode_tok_s': max(0,out.size(1)-ids.size(1))/max(dt,1e-9)}
    if torch.cuda.is_available() and device.type == 'cuda':
        res['gpu_peak_mem_mb'] = torch.cuda.max_memory_allocated(device) / (1024*1024)
    return res


def main(argv: Optional[Sequence[str]]=None):
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--prefer_best', action='store_true')
    p.add_argument('--data', default=''); p.add_argument('--image_dir', default=''); p.add_argument('--visual_tokenizer', default='')
    p.add_argument('--device', default='auto'); p.add_argument('--kernel_backend', default='auto')
    p.add_argument('--batches', type=int, default=10); p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--latency_tokens', type=int, default=32); p.add_argument('--out_json', default=''); p.add_argument('--dashboard', action='store_true')
    args=p.parse_args(argv)
    device=resolve_device(args.device); model=load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    vt_path=args.visual_tokenizer or (str(Path(args.checkpoint)) if (Path(args.checkpoint)/'visual_tokenizer_config.json').exists() else '')
    results={}
    if args.data: results.update(eval_text(model,Path(args.data),device,args.batches,args.batch_size))
    if args.image_dir:
        results.update(eval_image(model,Path(args.image_dir),device,args.batches,args.batch_size,vt_path))
        results.update(eval_visual_tokenizer(vt_path,Path(args.image_dir),device,max_images=min(32,args.batches*args.batch_size)))
    results.update(latency(model,device,'The answer is',args.latency_tokens))
    if args.dashboard:
        print('\n=== NEED PERFORMANCE DASHBOARD ===')
        for k,v in sorted(results.items()): print(f'{k:28s} {v}')
        print('=== END DASHBOARD ===\n')
    print(json.dumps(results, indent=2))
    if args.out_json: Path(args.out_json).write_text(json.dumps(results, indent=2), encoding='utf-8')
if __name__=='__main__': main()
