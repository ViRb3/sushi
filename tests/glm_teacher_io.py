"""Durable full-logit teacher rows and a byte-bounded lossless array cache."""
import json
import os
from collections import OrderedDict
from pathlib import Path


def atomic_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


class BoundedLru:
    def __init__(self,limit):
        self.limit=int(limit);self.bytes=0;self.items=OrderedDict()
    def get(self,key):
        item=self.items.pop(key,None)
        if item is None:return None
        self.items[key]=item
        return item[0]
    def reserve(self,size):
        while self.items and self.bytes+size>self.limit:
            _,(_,old)=self.items.popitem(last=False);self.bytes-=old
    def put(self,key,value,size):
        if size>self.limit:return
        old=self.items.pop(key,None)
        if old is not None:self.bytes-=old[1]
        self.reserve(size);self.items[key]=(value,size);self.bytes+=size


class Journal:
    def __init__(self,directory,identity,prompt_ids,vocab):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'capture-state.json';self.data_path=self.directory/'logits.f32'
        self.identity=identity;self.prompt_ids=list(prompt_ids);self.vocab=int(vocab)
        self.tokens=[];self.nll_sum=0.0
        if self.path.exists():
            data=json.loads(self.path.read_text())
            if data['identity']!=identity or data['prompt_ids']!=self.prompt_ids or data['vocab']!=vocab:raise ValueError('Teacher resume identity mismatch')
            self.tokens=data['tokens'];self.nll_sum=data['nll_sum']
        needed=len(self.tokens)*self.vocab*4
        if not self.data_path.exists():
            if needed:raise ValueError('Missing committed teacher rows')
            self.data_path.touch()
        if self.data_path.stat().st_size<needed:raise ValueError('Truncated committed teacher rows')
        with self.data_path.open('r+b') as f:f.truncate(needed)
        self.checkpoint()
    def checkpoint(self):
        atomic_json(self.path,dict(identity=self.identity,prompt_ids=self.prompt_ids,vocab=self.vocab,tokens=self.tokens,nll_sum=self.nll_sum))
    def append(self,row,token,nll):
        if len(row)!=self.vocab*4 or not 0<=token<self.vocab:raise ValueError('Invalid teacher row')
        with self.data_path.open('ab') as f:f.write(row);f.flush();os.fsync(f.fileno())
        self.tokens.append(int(token));self.nll_sum+=float(nll);self.checkpoint()
    def verify(self,position,row,token):
        with self.data_path.open('rb') as f:f.seek(position*self.vocab*4);saved=f.read(self.vocab*4)
        if saved!=row or self.tokens[position]!=token:raise ValueError(f'Teacher resume replay differs at row {position}')
