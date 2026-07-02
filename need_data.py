#!/usr/bin/env python3
"""Data preparation utilities for NEED.

Text: normalize/deduplicate/split. Images: build manifest with simple quality score.
No network access, no pickle outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
try:
    from PIL import Image
except Exception:
    Image=None


def text_quality(s: str) -> float:
    if not s.strip(): return 0.0
    printable=sum(ch.isprintable() for ch in s)/max(1,len(s))
    alpha=sum(ch.isalpha() for ch in s)/max(1,len(s))
    rep=1.0-len(set(s))/max(1,len(s))
    return float(max(0,min(1,0.5*printable+0.4*alpha-0.2*rep)))

def prep_text(args):
    seen=set(); docs=[]
    for p in Path(args.input).rglob('*') if Path(args.input).is_dir() else [Path(args.input)]:
        if p.is_file():
            s=p.read_text(encoding='utf-8', errors='replace')
            for chunk in re.split(r'\n\s*\n', s):
                chunk=' '.join(chunk.split())
                if len(chunk)<args.min_chars: continue
                h=hashlib.sha256(chunk.encode()).hexdigest()
                if h in seen: continue
                seen.add(h); q=text_quality(chunk)
                if q>=args.min_quality: docs.append(chunk)
    random.Random(args.seed).shuffle(docs)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n=len(docs); n_val=max(1,int(n*args.val_frac)) if n else 0; n_test=max(1,int(n*args.test_frac)) if n else 0
    splits={'train':docs[:max(0,n-n_val-n_test)], 'val':docs[max(0,n-n_val-n_test):max(0,n-n_test)], 'test':docs[max(0,n-n_test):]}
    for k,v in splits.items(): (out/f'{k}.txt').write_text('\n\n'.join(v), encoding='utf-8')
    print(json.dumps({'docs':n,'out_dir':str(out)}, indent=2))

def image_quality(path: Path) -> float:
    if Image is None: return 0.0
    im=Image.open(path).convert('RGB').resize((64,64)); arr=np.asarray(im,dtype=np.float32)/255.0
    return float(np.clip(arr.var()*4 + np.abs(arr[:,1:]-arr[:,:-1]).mean()*2,0,1))

def prep_images(args):
    rows=[]
    for ext in ('*.png','*.jpg','*.jpeg','*.webp','*.bmp'):
        for p in Path(args.input).rglob(ext):
            try:
                q=image_quality(p)
                if q>=args.min_quality: rows.append({'path':str(p.resolve()),'quality':q})
            except Exception: pass
    rows.sort(key=lambda r:r['quality'], reverse=True)
    out=Path(args.out_manifest); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    print(json.dumps({'images':len(rows),'manifest':str(out)}, indent=2))

def main(argv: Optional[Sequence[str]]=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd', required=True)
    t=sub.add_parser('text'); t.add_argument('--input',required=True); t.add_argument('--out_dir',required=True); t.add_argument('--min_chars',type=int,default=80); t.add_argument('--min_quality',type=float,default=0.25); t.add_argument('--val_frac',type=float,default=0.02); t.add_argument('--test_frac',type=float,default=0.02); t.add_argument('--seed',type=int,default=123)
    i=sub.add_parser('images'); i.add_argument('--input',required=True); i.add_argument('--out_manifest',required=True); i.add_argument('--min_quality',type=float,default=0.02)
    args=p.parse_args(argv)
    prep_text(args) if args.cmd=='text' else prep_images(args)
if __name__=='__main__': main()
