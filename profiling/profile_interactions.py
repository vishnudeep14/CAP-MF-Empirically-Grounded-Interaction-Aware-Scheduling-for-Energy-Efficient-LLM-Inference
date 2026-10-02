"""Measure directed pair interference for every bucket pair on one measured config."""
from __future__ import annotations
import argparse,asyncio,csv,statistics,time
from pathlib import Path
from openai import AsyncOpenAI
from .common import BUCKETS,OUTPUT_TARGETS,Sampler,energy_j,request,write_csv

def load_solo(path,config):
    rows=list(csv.DictReader(open(path,encoding='utf-8')));return {r['bucket']:float(r['latency_s']) for r in rows if r['config']==config and int(r['concurrency'])==1}
async def main_async(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);solo=load_solo(a.single_summary,a.config);client=AsyncOpenAI(base_url=a.base_url,api_key=a.api_key,timeout=300,max_retries=0);sampler=Sampler([int(x) for x in a.gpus.split(',')],a.sample_interval);rows=[]
    for bi in a.buckets:
      for bj in a.buckets:
       di=[];dj=[];energies=[]
       for rep in range(a.repeats):
        await sampler.start();await asyncio.sleep(a.sample_interval)
        ri,rj=await asyncio.gather(request(client,a.model,bi,OUTPUT_TARGETS[bi[1]]),request(client,a.model,bj,OUTPUT_TARGETS[bj[1]]))
        await asyncio.sleep(a.sample_interval);E=energy_j(await sampler.finish());di.append(ri['latency_s']/solo[bi]-1);dj.append(rj['latency_s']/solo[bj]-1);energies.append(E if E else 0)
       # Directed interaction = slowdown suffered by i while co-running with j.
       rows.append({'config':a.config,'tp':a.tp,'bucket_i':bi,'bucket_j':bj,'interaction':statistics.mean(di),'reverse_observed':statistics.mean(dj),'pair_energy_j':statistics.mean(energies)})
       print(rows[-1])
    write_csv(out/'interactions.csv',rows)
def cli():
 p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://localhost:8000/v1');p.add_argument('--api-key',default='unused');p.add_argument('--model',required=True);p.add_argument('--config',required=True);p.add_argument('--tp',type=int,required=True);p.add_argument('--gpus',default='0');p.add_argument('--single-summary',required=True);p.add_argument('--buckets',nargs='+',default=BUCKETS);p.add_argument('--repeats',type=int,default=3);p.add_argument('--sample-interval',type=float,default=.1);p.add_argument('--output',default='data/raw/interactions');return p.parse_args()
if __name__=='__main__':asyncio.run(main_async(cli()))
