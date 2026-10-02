from __future__ import annotations
import argparse,csv
from pathlib import Path
from .common import write_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile-dirs',nargs='+',required=True);p.add_argument('--interaction-dirs',nargs='+',required=True);p.add_argument('--output',default='data/processed');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 configs=[];inter=[]
 for d in a.profile_dirs:configs += list(csv.DictReader(open(Path(d)/'config_profiles.csv',encoding='utf-8')))
 for d in a.interaction_dirs:inter += list(csv.DictReader(open(Path(d)/'interactions.csv',encoding='utf-8')))
 write_csv(out/'config_profiles.csv',configs);write_csv(out/'interactions.csv',inter)
 print(f'wrote {len(configs)} config rows and {len(inter)} interaction rows to {out}')
if __name__=='__main__':main()
