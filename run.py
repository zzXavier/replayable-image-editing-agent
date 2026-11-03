import argparse
import csv
import json
import os
from pathlib import Path
from PIL import Image
from src.data import ROOT,items,prepare
from src.comfy import Comfy
from src.agent import execute,replay

def evaluate(out):
    rows=[]
    for trace in sorted(out.glob('*/*/trace.json')):
        saved=replay(trace); rows.append({'case':trace.parents[1].name,'mode':saved['mode'],'status':saved['status'],'render_calls':sum(s['tool']=='render' for s in saved['steps']),**saved.get('scores',{})})
    if not rows:raise ValueError('No finished task traces in output directory')
    with (out/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','mode','status','render_calls','keep_error','red_background']);w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['demo','reproduce','evaluate','replay']);p.add_argument('--url',default='http://127.0.0.1:8188');p.add_argument('--checkpoint',default=os.getenv('COMFY_INPAINT_CHECKPOINT'));p.add_argument('--output',type=Path);p.add_argument('--trace',type=Path)
    a=p.parse_args();out=a.output or ROOT/'results'/('demo' if a.action=='demo' else 'formal')
    if a.action=='replay': print(json.dumps(replay(a.trace),indent=2))
    elif a.action=='evaluate':evaluate(out)
    else:
        if not a.checkpoint:p.error('Specify --checkpoint using the exact ComfyUI inpainting checkpoint filename')
        engine=Comfy(a.url,a.checkpoint)
        engine.request('GET','/system_stats')
        folder=a.checkpoint.startswith('diffusers:')
        node,field=('DiffusersLoader','model_path') if folder else ('CheckpointLoaderSimple','ckpt_name')
        options=engine.request('GET','/object_info/'+node).json()[node]['input']['required'][field][0]
        if (a.checkpoint[10:] if folder else a.checkpoint) not in options:raise ValueError('Checkpoint missing from local ComfyUI configuration')
        examples=[i for i in items() if i['split']=='test' and i['id']!='B083V9C8D2']
        for item in (examples[:1] if a.action=='demo' else examples):
            im,mask,_=prepare(item)
            for failure in (['normal'] if a.action=='demo' else ['normal','bad-mask','tool-fault']):
                m=Image.new('L',mask.size) if failure=='bad-mask' else mask
                for mode in ('direct','fixed','feedback'):
                    execute(engine,im,m,'A studio photo of '+item['name']+' with a plain red backdrop',mode,13,out/(item['id']+'-'+failure)/mode,fail_first=failure=='tool-fault')
        evaluate(out)


