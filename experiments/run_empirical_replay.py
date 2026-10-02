"""Replay solver assignments by executing each config's dedicated vLLM endpoint.

For a real multi-config experiment, keep one endpoint per measured config alive
or start them sequentially and replay the corresponding group. Each config name
is mapped to a base URL through --endpoint config=url.
"""
from __future__ import annotations
import argparse,asyncio,csv,json,statistics,time
from pathlib import Path
from openai import AsyncOpenAI
from profiling.common import OUTPUT_TARGETS,request,write_csv
from scheduler.models import Request

def parse_endpoints(items):return {x.split('=',1)[0]:x.split('=',1)[1] for x in items}
async def main_async(a):
 endpoints=parse_endpoints(a.endpoint);req_rows=list(csv.DictReader(open(a.requests,encoding='utf-8')));result_rows=list(csv.DictReader(open(a.solver_results,encoding='utf-8')));out=[]
 for sr in result_rows:
  assignment={int(k):v for k,v in json.loads(sr['assignment_json']).items()};groups={}
  for i,r in enumerate(req_rows[:a.limit]):groups.setdefault(assignment[i],[]).append((i,r['bucket']))
  metrics=[]
  for cfg,items in groups.items():
   if cfg not in endpoints:raise ValueError(f'missing endpoint for {cfg}')
   client=AsyncOpenAI(base_url=endpoints[cfg],api_key=a.api_key,timeout=300,max_retries=0)
   tasks=[request(client,a.model,b,OUTPUT_TARGETS[b[1]]) for _,b in items];rs=await asyncio.gather(*tasks);metrics.extend(rs)
  lats=[x['latency_s'] for x in metrics]
  out.append({'solver':sr['solver'],'requests':len(metrics),'measured_mean_latency_s':statistics.mean(lats),'measured_p95_latency_s':sorted(lats)[int(.95*(len(lats)-1))],'success_rate':1.0})
 write_csv(Path(a.output),out);[print(x) for x in out]
def cli():
 p=argparse.ArgumentParser();p.add_argument('--solver-results',default='results/solver_results.csv');p.add_argument('--requests',required=True);p.add_argument('--limit',type=int,default=12);p.add_argument('--model',required=True);p.add_argument('--api-key',default='unused');p.add_argument('--endpoint',action='append',required=True,help='CONFIG=http://host:port/v1');p.add_argument('--output',default='results/replay_results.csv');return p.parse_args()
if __name__=='__main__':asyncio.run(main_async(cli()))
