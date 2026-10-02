from __future__ import annotations
import asyncio,csv,math,statistics,subprocess,time
from dataclasses import dataclass
from openai import AsyncOpenAI

BUCKETS=['SS','SM','SL','MS','MM','ML','LS','LM','LL']
INPUT_TARGETS={'S':128,'M':512,'L':1024};OUTPUT_TARGETS={'S':64,'M':128,'L':256}
CORPUS='Large language model inference includes prefill decode KV cache batching memory pressure scheduling throughput latency and service objectives. '
def prompt(bucket):
    n=max(20,int(INPUT_TARGETS[bucket[0]]*.72));w=CORPUS.split();text=' '.join((w*((n//len(w))+1))[:n])
    return text+'\nExplain reliable LLM inference serving considerations.'
def pct(xs,p):
    if not xs:return None
    s=sorted(xs);x=(len(s)-1)*p;l=math.floor(x);h=math.ceil(x);return s[l] if l==h else s[l]*(h-x)+s[h]*(x-l)
class Sampler:
    def __init__(self,gpus,interval=.1):self.gpus=gpus;self.interval=interval;self.rows=[];self.stop=asyncio.Event();self.task=None
    async def start(self):self.rows=[];self.stop.clear();self.task=asyncio.create_task(self.loop())
    async def finish(self):self.stop.set();await self.task;return self.rows
    async def loop(self):
        while not self.stop.is_set():
            t=time.perf_counter()
            for g in self.gpus:
                try:
                    cp=await asyncio.to_thread(subprocess.run,['nvidia-smi','-i',str(g),'--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
                    v=[float(x.strip()) for x in cp.stdout.split(',')];self.rows.append({'t':t,'gpu':g,'power_w':v[0],'util_pct':v[1],'memory_mib':v[2],'temp_c':v[3]})
                except Exception:pass
            try:await asyncio.wait_for(self.stop.wait(),timeout=self.interval)
            except asyncio.TimeoutError:pass
def energy_j(rows):
    by={}
    for r in rows:by[r['t']]=by.get(r['t'],0)+r['power_w']
    pts=sorted(by.items())
    return sum(.5*(a[1]+b[1])*(b[0]-a[0]) for a,b in zip(pts,pts[1:])) if len(pts)>1 else None
async def request(client,model,bucket,out_tokens):
    t=time.perf_counter();first=None;usage=None
    stream=await client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt(bucket)}],max_tokens=out_tokens,temperature=0,stream=True,stream_options={'include_usage':True})
    async for ch in stream:
        now=time.perf_counter();usage=getattr(ch,'usage',None) or usage
        if ch.choices and ch.choices[0].delta.content and first is None:first=now
    end=time.perf_counter()
    return {'latency_s':end-t,'ttft_s':first-t if first else None,'input_tokens':getattr(usage,'prompt_tokens',None),'output_tokens':getattr(usage,'completion_tokens',None)}
def write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
