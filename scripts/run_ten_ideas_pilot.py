import argparse
from mindset.acl.ten_ideas_pilot import run, ARMS
p=argparse.ArgumentParser(description='DGX-only original100 ten-idea quality screen')
p.add_argument('--arms',nargs='+',choices=ARMS,required=True)
p.add_argument('--output',required=True)
p.add_argument('--shard',type=int,default=0)
p.add_argument('--shards',type=int,default=1)
a=p.parse_args()
run(a.arms,a.output,a.shard,a.shards)
