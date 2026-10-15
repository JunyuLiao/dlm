import argparse,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(); out={}
 for recfile in a.root.glob('**/audit/*/shard*/attempt*/records.jsonl'):
  parts=recfile.parts; arm=parts[-4]; regime=parts[-6] if len(parts)>=6 else 'unknown'; rows=[json.loads(x) for x in recfile.read_text().splitlines() if x.strip()]
  if not rows: continue
  key=f'{regime}/{arm}'; z=out.setdefault(key,{'arm':arm,'regime':regime,'rows':[]});
  for r in rows:
   ad=(r.get('receipts') or {}).get('adapter') or {}; z['rows'].append({'dataset':r.get('dataset'),'index':r.get('index'),'panel_seed':r.get('panel_seed'),'denoise_forwards':r.get('denoise_forwards'),'canvases':r.get('canvases'),'wall_s':r.get('wall_s'),'overall_rectangle_sparsity':ad.get('overall_rectangle_sparsity'),'global_work_fraction':ad.get('global_work_fraction',ad.get('global_prefix_work_fraction')),'global_eligible_tiles':ad.get('global_eligible_tiles'),'global_kept_tiles':ad.get('global_kept_tiles')})
 for z in out.values():
  rs=z.pop('rows'); z['cells']=len(rs); z['trajectory']={}
  for k in ('denoise_forwards','canvases','wall_s','overall_rectangle_sparsity','global_work_fraction'):
   v=[x[k] for x in rs if isinstance(x.get(k),(int,float))]; z['trajectory'][k+'_mean']=statistics.mean(v) if v else None
  e=sum(x.get('global_eligible_tiles',0) or 0 for x in rs); k=sum(x.get('global_kept_tiles',0) or 0 for x in rs); z['trajectory']['global_tile_sparsity']=1-k/e if e else None
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'schema':'v31_value_followup_trajectory_v1','arms':out},indent=2)+'\n');print(json.dumps({'arms':len(out),'out':str(a.out)}))
if __name__=='__main__':main()
