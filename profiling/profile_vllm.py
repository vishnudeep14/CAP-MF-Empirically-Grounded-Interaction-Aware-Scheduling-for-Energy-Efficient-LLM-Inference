from __future__ import annotations
import argparse,asyncio,csv,json,statistics,time
from pathlib import Path
from openai import AsyncOpenAI
from .common import BUCKETS,OUTPUT_TARGETS,Sampler,energy_j,request,write_csv

async def main_async(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);client=AsyncOpenAI(base_url=a.base_url,api_key=a.api_key,timeout=300,max_retries=0)
    await client.chat.completions.create(model=a.model,messages=[{'role':'user','content':'Reply OK'}],max_tokens=8)
    sampler=Sampler([int(x) for x in a.gpus.split(',')],a.sample_interval);rows=[];raw=[];gpu=[]
    for b in a.buckets:
      for c in a.concurrency:
       for rep in range(a.repeats):
        await sampler.start();await asyncio.sleep(a.sample_interval);t=time.perf_counter()
        results=await asyncio.gather(*[request(client,a.model,b,OUTPUT_TARGETS[b[1]]) for _ in range(c)]);wall=time.perf_counter()-t;await asyncio.sleep(a.sample_interval);samples=await sampler.finish();E=energy_j(samples)
        run=f'{a.config}-{b}-c{c}-r{rep}-{time.time_ns()}';gpu += [{'run_id':run,**x} for x in samples];raw += [{'run_id':run,'bucket':b,'concurrency':c,'repeat':rep,**x} for x in results]
        rows.append({'config':a.config,'tp':a.tp,'capacity':a.capacity,'bucket':b,'concurrency':c,'repeat':rep,'latency_s':statistics.mean(x['latency_s'] for x in results),'ttft_s':statistics.mean(x['ttft_s'] for x in results if x['ttft_s'] is not None),'input_tokens':statistics.mean(x['input_tokens'] for x in results if x['input_tokens'] is not None),'output_tokens':statistics.mean(x['output_tokens'] for x in results if x['output_tokens'] is not None),'energy_j_per_request':E/c if E is not None else None,'window_energy_j':E,'window_s':wall})
        print(rows[-1])
    write_csv(out/'single_windows.csv',rows);write_csv(out/'single_requests.csv',raw);write_csv(out/'gpu_samples.csv',gpu)
    # aggregate and create strict measured config rows using C=1. SLO defaults to multiplier x measured solo latency.
    summary=[]
    for b in a.buckets:
      for c in a.concurrency:
       g=[r for r in rows if r['bucket']==b and r['concurrency']==c]
       summary.append({'config':a.config,'tp':a.tp,'capacity':a.capacity,'bucket':b,'concurrency':c,'latency_s':statistics.mean(r['latency_s'] for r in g),'energy_j':statistics.mean(r['energy_j_per_request'] for r in g if r['energy_j_per_request'] is not None),'ttft_s':statistics.mean(r['ttft_s'] for r in g)})
    write_csv(out/'single_summary.csv',summary)
    c1=[{'config':r['config'],'tp':r['tp'],'capacity':r['capacity'],'bucket':r['bucket'],'latency_s':r['latency_s'],'energy_j':r['energy_j'],'slo_s':a.slo_multiplier*r['latency_s']} for r in summary if r['concurrency']==1]
    write_csv(out/'config_profiles.csv',c1)

def cli():
 p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://localhost:8000/v1');p.add_argument('--api-key',default='unused');p.add_argument('--model',required=True);p.add_argument('--config',required=True);p.add_argument('--tp',type=int,required=True);p.add_argument('--capacity',type=int,required=True);p.add_argument('--gpus',default='0');p.add_argument('--buckets',nargs='+',default=BUCKETS);p.add_argument('--concurrency',nargs='+',type=int,default=[1,2,4,8]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--sample-interval',type=float,default=.1);p.add_argument('--slo-multiplier',type=float,default=2.0);p.add_argument('--output',default='data/raw/profile');return p.parse_args()
if __name__=='__main__':asyncio.run(main_async(cli()))
