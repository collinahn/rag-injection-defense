#!/usr/bin/env python3
"""Preserved primary contrast and post-hoc G2 sensitivity, without new LLM calls."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,math,platform
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readjl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main(root:Path,out:Path):
    root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    inp=root/'inputs/raw';repo=root/'source/Thesis'
    sys.path.insert(0,str(repo/'src'))
    from thesis_rag.statistics import AttackOutcome,cluster_bootstrap_asr_reduction,fit_paired_defense_gee
    from thesis_rag.domain import DefenseId
    import numpy as np
    import scipy,statsmodels,pydantic
    from scipy.stats import norm
    adj={x['experiment_id']:x for x in readjl(inp/'main-g2-adjudications.jsonl')}
    definitions={x['attack_id']:x for x in readjl(inp/'test-attacks.jsonl')}
    csv_index={x['experiment_id']:x for x in readcsv(inp/'attack_case_links.csv')}
    rows=[]
    for tr in ['forced','e2e']:
        for r in readjl(inp/f'main-{tr}-results.jsonl'):
            assert not r['technical_failure'];a=definitions[r['attack_id']]
            c=csv_index[r['experiment_id']]
            assert hashlib.sha256(r['raw_response'].encode()).hexdigest()==c['response_sha256']
            if a['goal']=='G2':
                ar=adj[r['experiment_id']]
                assert ar['response_sha256']==c['response_sha256']
                success=ar['adjudicated_attack_success'];alt=ar['decision_code']=='semantic_success'
            else:success=alt=r['attack_success']
            assert success==(c['final_attack_success']=='True') and alt==(c['alternative_attack_success']=='True')
            assert r['base_context_id']==c['base_context_id'] and r['defense_id']==c['defense_id']
            rows.append({'track':tr,'base_context_id':r['base_context_id'],'attack_id':r['attack_id'],'defense_id':r['defense_id'],
                         'primary':bool(success),'alternative':bool(alt),'goal':a['goal']})
    assert len(rows)==1350
    f=[r for r in rows if r['track']=='forced' and r['defense_id'] in ['D0','D4']]
    ci_ids=sorted({r['base_context_id'] for r in f});attack_ids=sorted({r['attack_id'] for r in f})
    ix={(r['attack_id'],r['defense_id']):r for r in f};result={}
    assert len(ci_ids)==60 and len(attack_ids)==180 and len(f)==360
    for kind in ['primary','alternative']:
        outcomes=[AttackOutcome(track='forced',base_context_id=r['base_context_id'],attack_id=r['attack_id'],defense_id=DefenseId(r['defense_id']),attack_success=r[kind]) for r in f]
        boot=cluster_bootstrap_asr_reduction(outcomes,baseline=DefenseId.D0,treatment=DefenseId.D4,iterations=10000,seed=42).model_dump(mode='json')
        gee=fit_paired_defense_gee(outcomes,baseline=DefenseId.D0,treatment=DefenseId.D4).model_dump(mode='json')
        # Independent arithmetic for this balanced paired, 3-per-context design.
        y=np.array([[sum(r[kind] for r in f if r['base_context_id']==c and r['defense_id']==d) for d in ['D0','D4']] for c in ci_ids],dtype=float)
        n=np.array([[sum(r['base_context_id']==c and r['defense_id']==d for r in f) for d in ['D0','D4']] for c in ci_ids]);assert np.all(n==3)
        inds=np.random.default_rng(42).integers(0,60,size=(10000,60))
        dist=(y[inds,0].sum(1)-y[inds,1].sum(1))/n[inds,0].sum(1)
        probs=y.sum(0)/n.sum(0)
        beta=math.log(probs[1]/(1-probs[1]))-math.log(probs[0]/(1-probs[0]))
        inf=(y[:,1]-n[:,1]*probs[1])/(n[:,1].sum()*probs[1]*(1-probs[1]))-(y[:,0]-n[:,0]*probs[0])/(n[:,0].sum()*probs[0]*(1-probs[0]))
        se=float(np.linalg.norm(inf));indci=np.quantile(dist,[.025,.975],method='linear')
        assert np.allclose(indci,[boot['ci_lower'],boot['ci_upper']],atol=1e-14,rtol=0)
        assert abs(math.exp(beta)-gee['odds_ratio'])<1e-12 and abs(se-gee['standard_error'])<1e-12
        paired=Counter((ix[a,'D0'][kind],ix[a,'D4'][kind]) for a in attack_ids)
        result[kind]={'success_D0':int(y[:,0].sum()),'success_D4':int(y[:,1].sum()),'paired':{'both_failure':paired[False,False],
          'both_success':paired[True,True],'D0_only':paired[True,False],'D4_only':paired[False,True]},'bootstrap':boot,'gee':gee,
          'independent_check':{'ci':indci.tolist(),'odds_ratio':math.exp(beta),'se':se,'p_value':float(2*norm.sf(abs(beta/se)))}}
    reference=json.loads((root/'inputs/verified_sensitivity_reference.json').read_text())
    for kind in ['primary','alternative']:
        for k in ['odds_ratio','ci_lower','ci_upper','standard_error','p_value']:
            assert abs(result[kind]['gee'][k]-reference[kind]['original_GEE'][k])<1e-12
        assert result[kind]['bootstrap']['distribution_sha256']==reference[kind]['original_bootstrap']['distribution_sha256']
    decomposition=Counter(ix[a,'D0']['goal'] for a in attack_ids if ix[a,'D0']['primary'] and not ix[a,'D4']['primary'])
    assert decomposition==Counter({'G1':1,'G2':4,'G5':7})
    sources=[inp/'main-forced-results.jsonl',inp/'main-e2e-results.jsonl',inp/'main-g2-adjudications.jsonl',inp/'test-attacks.jsonl',inp/'attack_case_links.csv',
             repo/'src/thesis_rag/statistics.py',repo/'src/thesis_rag/domain.py',repo/'src/thesis_rag/canonical.py']
    result['scope']={'post_hoc':True,'alternative':'G2 contradictory_success excluded; other goals and denominators unchanged',
        'contrast':'Forced-context D4/D0 only','primary_definition_preserved':True,'new_generation':False,'all_alternative_Holm_contrasts_computed':False,
        'source_inputs':[{ 'path':str(p.relative_to(root)),'sha256':sha(p)} for p in sources]}
    result['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'pydantic':pydantic.__version__}
    write(out/'g2_cluster_sensitivity.json',result)
    with (out/'asr_reduction_by_goal.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=['goal','D0_only_successes','goal_denominator','goal_reduction_pp','whole_attack_denominator','contribution_pp']);writer.writeheader()
        for goal in ['G1','G2','G3','G4','G5']:
            k=decomposition[goal];writer.writerow({'goal':goal,'D0_only_successes':k,'goal_denominator':36,'goal_reduction_pp':100*k/36,'whole_attack_denominator':180,'contribution_pp':100*k/180})
    manifest=[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']
    write(out/'artifact_manifest.json',manifest)
    print(json.dumps({k:result[k] for k in ['primary','alternative']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    b=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=b);p.add_argument('--output',type=Path,default=b/'analysis')
    a=p.parse_args();main(a.root,a.output)
