"""Create request CSV from a REAL trace/export containing bucket labels.
No random request generation is used in the empirical headline pipeline."""
from __future__ import annotations
import argparse,csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--bucket-column',default='bucket');p.add_argument('--output',default='data/processed/requests.csv');a=p.parse_args();rows=list(csv.DictReader(open(a.input,encoding='utf-8')));out=[]
 for r in rows:
  b=r[a.bucket_column].upper()
  if b not in {'SS','SM','SL','MS','MM','ML','LS','LM','LL'}:raise ValueError(f'invalid bucket {b}')
  out.append({'bucket':b})
 import os;os.makedirs(os.path.dirname(a.output),exist_ok=True)
 with open(a.output,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['bucket']);w.writeheader();w.writerows(out)
 print(f'wrote {len(out)} measured/trace-derived requests')
if __name__=='__main__':main()
