"""Minimal client for a local ComfyUI server. The server owns all model execution."""
from pathlib import Path
import time
import requests
class Comfy:
    def __init__(self,url,checkpoint,deadline=240):
        self.url=url.rstrip('/');self.checkpoint=checkpoint;self.deadline=deadline
    def request(self,method,path,**kwargs):
        r=requests.request(method,self.url+path,timeout=(4,30),**kwargs);r.raise_for_status();return r
    def upload(self,image):
        with image.open('rb') as f:
            obj=self.request('POST','/upload/image',files={'image':(image.name,f,'image/png')},data={'type':'input','overwrite':'false'}).json()
        return '/'.join(filter(None,[obj.get('subfolder'),obj['name']]))
    def render(self,source,mask,prompt,seed,path):
        from PIL import Image
        rgba=source.convert('RGBA');rgba.putalpha(mask)
        inp=path.with_name(path.stem+'-input.png');rgba.save(inp)
        graph=workflow(self.checkpoint,self.upload(inp),prompt,seed)
        payload=self.request('POST','/prompt',json={'prompt':graph}).json()
        job=payload['prompt_id'];stop=time.monotonic()+self.deadline
        while time.monotonic()<stop:
            history=self.request('GET','/history/'+job).json().get(job)
            if not history:
                time.sleep(.7);continue
            if history.get('status',{}).get('status_str')=='error':
                raise RuntimeError(f"Comfy execution failed for {job}: {history['status'].get('messages',[])}")
            image=history.get('outputs',{}).get('8',{}).get('images',[])
            if not image: raise RuntimeError(f'No saved image in completed task {job}')
            meta=image[0];result=self.request('GET','/view',params={key:meta[key] for key in ('filename','subfolder','type')})
            path.write_bytes(result.content)
            with Image.open(path) as im:
                if im.size!=source.size: raise ValueError('Unexpected output dimensions')
                return im.convert('RGB'),graph
        raise TimeoutError(f'ComfyUI task {job} exceeded {self.deadline} s')
def workflow(checkpoint,filename,prompt,seed):
    def node(cls,**kwargs):return {'class_type':cls,'inputs':kwargs}
    model = node('DiffusersLoader',model_path=checkpoint[10:]) if checkpoint.startswith('diffusers:') else node('CheckpointLoaderSimple',ckpt_name=checkpoint)
    return {
       '1':model,
       '2':node('LoadImage',image=filename),
       '3':node('CLIPTextEncode',text=prompt,clip=['1',1]),
       '4':node('CLIPTextEncode',text='',clip=['1',1]),
       '5':node('InpaintModelConditioning',positive=['3',0],negative=['4',0],vae=['1',2],pixels=['2',0],mask=['2',1],noise_mask=True),
       '6':node('KSampler',model=['1',0],seed=seed,steps=12,cfg=7,sampler_name='euler',scheduler='normal',positive=['5',0],negative=['5',1],latent_image=['5',2],denoise=1),
       '7':node('VAEDecode',samples=['6',0],vae=['1',2]),
       '8':node('SaveImage',images=['7',0],filename_prefix='vision_edit')}


