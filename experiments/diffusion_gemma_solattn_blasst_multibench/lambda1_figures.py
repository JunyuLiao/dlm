"""Reproducible static scientific figures from saved rollout/probe artifacts."""
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .lambda1_probe import BASE,OUT
from .runner import _write

BENCHES=('ruler16k','longbench','aime24','livecodebench_v6')
LABELS=('RULER-16K','LongBench','AIME (10)','LiveCodeBench (12)')

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    rows=json.loads((BASE/'summary.json').read_text())['results']
    endpoint={r['benchmark']:r for r in rows if r['condition']=='blasst_length_aware_s90'}
    records=[];snapshots=[]
    for directory in sorted(OUT.glob('probe_*')):
        if not directory.is_dir(): continue
        audit=json.loads((directory/'audit.json').read_text());data=np.load(directory/'snapshots.npz')
        for r in json.loads((directory/'records.json').read_text()):
            r.update(benchmark=audit['benchmark'],id=audit['id']);records.append(r)
            snapshots.append((r,{k:data[r['key']+'_'+k] for k in ('maxima','previous','counts','mass')}))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    x=np.arange(4);w=.25
    for i,kind in enumerate(('overall','global','local')):
        axes[0].bar(x+(i-1)*w,[100*endpoint[b]['actual'][kind]['full_tile_sparsity'] for b in BENCHES],w,label=kind)
    axes[0].set_ylabel('Skipped eligible physical tiles (%)');axes[0].legend();axes[0].set_ylim(0,100)
    for i,kind in enumerate(('global','local')):
        axes[1].bar(x+(i-.5)*.35,[100*endpoint[b]['actual'][kind]['eligible_tiles']/endpoint[b]['actual']['overall']['eligible_tiles'] for b in BENCHES],.35,label=kind)
    axes[1].set_ylabel('Share of eligible tiles (%)');axes[1].legend();axes[1].set_ylim(0,100)
    for i,key in enumerate(('retained_attention_mass','sparse_trajectory_retained_mass')):
        axes[2].bar(x+(i-.5)*.35,[100*endpoint[b][key] for b in BENCHES],.35,label=('Dense trajectory','Sparse trajectory')[i])
    axes[2].set_ylabel('Dense attention mass retained (%)');axes[2].set_ylim(0,100);axes[2].legend()
    for ax in axes: ax.set_xticks(x,LABELS,rotation=20,ha='right')
    fig.suptitle('λ=1 in both attention types: complete 122-sample rollout counts');fig.tight_layout();save(fig,'01_full_run_ceiling')

    summary=[];fig,axes=plt.subplots(2,4,figsize=(16,7),sharey='row')
    group_counts=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    for r,d in snapshots:
        has=d['counts']>0;keep=has&(d['maxima']>=d['previous'])
        for group in (1,2,4,8,16,32,64):
            h,q,t=has.shape;pad=(-q)%group
            hp=np.pad(has,((0,0),(0,pad),(0,0))).reshape(h,-1,group,t).any(2)
            kp=np.pad(keep,((0,0),(0,pad),(0,0))).reshape(h,-1,group,t).any(2)
            group_counts[(r['benchmark'],r['attention_type'],group)]+=np.array([(hp&~kp).sum(),hp.sum()])
    for col,b in enumerate(BENCHES):
        for kind,color in (('global','tab:orange'),('local','tab:blue')):
            subset=[r for r in records if r['benchmark']==b and r['attention_type']==kind]
            sums={k:sum(r[k] for r in subset) for k in ('eligible_tiles','skipped_tiles','row_votes','row_skip_votes','strict_record_tiles','tie_only_tiles','mass_rows','physical_mass_sum','row_mask_mass_sum','finalmax_skipped_tiles')}
            summary.append(dict(benchmark=b,attention_type=kind,**sums))
            axes[0,col].plot((1,2,4,8,16,32,64),[100*group_counts[(b,kind,g)][0]/group_counts[(b,kind,g)][1] for g in (1,2,4,8,16,32,64)],'o-',label=kind,color=color)
            ds=[d for r,d in snapshots if r['benchmark']==b and r['attention_type']==kind]
            margins=np.concatenate([(d['maxima']-d['previous'])[(d['counts']>0)&np.isfinite(d['previous'])] for d in ds])
            axes[1,col].hist(margins,bins=np.linspace(-20,20,101),weights=np.ones(len(margins))/len(margins),histtype='step',color=color,label=kind)
        axes[0,col].set_xscale('log',base=2);axes[0,col].set_xticks((1,4,16,64),('1','4','16','64'));axes[0,col].set_title(LABELS[col]);axes[0,col].set_xlabel('Queries sharing one tile decision')
        axes[1,col].axvline(0,color='black',ls='--');axes[1,col].set_xlabel('Block maximum − previous running maximum')
        axes[0,col].legend();axes[1,col].set_xlim(-20,20)
    axes[0,0].set_ylabel('Physical tile sparsity (%)');axes[1,0].set_ylabel('Fraction of eligible row/block pairs')
    fig.suptitle('Same-state probes: row voting and the union across queries\nTwo prompts/benchmark, all heads/layers, first 64 queries, calls 1 and 4; no regrouped generation')
    fig.tight_layout();save(fig,'02_row_votes_and_score_distribution');_write(OUT/'probe_summary.json',summary)
    _write(OUT/'grouping_counts.json',[dict(benchmark=b,attention_type=k,group_size=g,skipped=int(v[0]),eligible=int(v[1])) for (b,k,g),v in group_counts.items()])

    # First sampled layer of each type, first call, first selected prompt.
    for b in BENCHES:
        fig,axes=plt.subplots(3,2,figsize=(14,8),gridspec_kw={'height_ratios':[2,3,1]})
        for col,kind in enumerate(('global','local')):
            r,d=next((r,d) for r,d in snapshots if r['benchmark']==b and r['attention_type']==kind and r['call']==0)
            eligible=(d['counts']>0).any(1);record=(d['counts']>0)&(d['maxima']>=d['previous'])
            density=record.any(1).sum(1)/eligible.sum(1)
            head=int(np.argsort(density,kind='stable')[len(density)//2])
            idx=np.flatnonzero(eligible[head]);mx=d['maxima'][head][:,idx];pr=d['previous'][head][:,idx];valid=d['counts'][head][:,idx]>0
            votes=valid&(mx>=pr);tilekeep=votes.any(0);margin=mx-pr
            xpos=np.arange(len(idx));row=0
            axes[0,col].plot(xpos,np.where(valid[row],mx[row],np.nan),label='Block maximum',lw=1.2)
            axes[0,col].step(xpos,np.where(np.isfinite(pr[row]),pr[row],np.nan),where='mid',label='Previous running max',lw=1.2)
            axes[0,col].scatter(xpos[votes[row]],mx[row,votes[row]],s=20,c='tab:green',label='Row retains (record/tie)')
            axes[0,col].set_title(f'{kind}: layer {r["layer"]}, head {head}, query 0');axes[0,col].legend(fontsize=8)
            display=np.where(valid,np.clip(margin,-8,8),np.nan)
            im=axes[1,col].imshow(display,aspect='auto',origin='lower',cmap='coolwarm',vmin=-8,vmax=8,interpolation='nearest')
            fig.colorbar(im,ax=axes[1,col],fraction=.025,pad=.015,label='Margin (clipped to ±8; first valid block = +∞)')
            axes[1,col].set_ylabel('Query row in the 64-row tile')
            axes[2,col].bar(xpos,votes.sum(0),color=np.where(tilekeep,'tab:green','tab:red'),width=1)
            axes[2,col].scatter(xpos[~tilekeep],np.zeros((~tilekeep).sum()),color='tab:red',marker='x',s=35,zorder=3,label='Whole tile skipped')
            axes[2,col].legend(fontsize=8)
            axes[2,col].set_ylabel('Keep votes');axes[2,col].set_xlabel('Eligible KV tile, in traversal order')
            axes[2,col].set_ylim(0,64)
            for ax in axes[:,col]:
                ax.set_xlim(-.5,len(idx)-.5)
                boundary=np.searchsorted(idx*64,r['prefix_length'])
                if boundary<len(idx): ax.axvline(boundary-.5,color='purple',ls=':')
            axes[1,col].text(.01,.97,'Blue: row votes skip; red: row sets/ties a record',transform=axes[1,col].transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.85})
        fig.suptitle(f'{b}: measured row-wise KV block scores at λ=1\nGreen tile survives if even ONE row keeps it; purple line marks canvas start')
        fig.tight_layout();save(fig,'03_scores_'+b)

if __name__=='__main__': main()
