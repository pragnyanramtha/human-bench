"""EditLens CPU evaluation; pinned upstream inference/preprocessing at 05a588f15d792330ccaf91be8ee4fdb54ce26835.
Numerics: CPU bf16 autocast; Llama NormedLinear parameters bf16, optional NF4 base.
Preprocessing and NormedLinear adapted from Pangram/EditLens (Thai, Emi, Masrour, Iyyer),
https://github.com/pangramlabs/EditLens, CC BY-NC-SA 4.0.
This derivative file is licensed CC BY-NC-SA 4.0:
https://creativecommons.org/licenses/by-nc-sa/4.0/ . Changes: CPU inference, folded
HumanBench inputs, resumable scoring, fixed subsets, and aggregate reporting.
"""

import os
RUN_ID = os.environ.get("HUMANBENCH_RUN_ID", "run")
import argparse, hashlib, json, random, re, resource, time
from importlib.metadata import version
from collections import defaultdict
import numpy as np
import torch
from common import ROOT,EVAL_DIR,PAIRS_PATH,fold,prose_view,read_jsonl,word_count
from analyze import metrics,boot,pairwise
from score_trials import threshold
OUT=EVAL_DIR/'editlens'
PINS={'ml-resources/editlens_roberta-large':'637532be5fd2965eb00ad16273f52588e6d1f216','ml-resources/editlens_Llama-3.2-3B':'cf23fd0738fc97f2420a8132e0866fa0ca840c3d','unsloth/Llama-3.2-3B':'d4446454d87d51aa42e1fb174f25acc5f8762331'}
SEED=20261110

def fmt_ci(value,ci):
 return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

def clean(text):
 import emoji
 text=emoji.demojize(text)
 if '</think>' in text:text=text.split('</think>')[1].strip()
 paragraphs=[p for p in text.split('\n') if p.strip()]
 if paragraphs:
  first=emoji.replace_emoji(re.sub(r'^[^a-zA-Z0-9]*','',paragraphs[0]),'')
  if any(first.startswith(p) for p in ['Sure','Here','Abstract','Title',"I'm happy to help",'Certainly']) and len(paragraphs)>1:text='\n'.join(paragraphs[1:])
 return re.sub(r'\s+',' ',text.lower()).strip()

class NormedLinear(torch.nn.Module):
 def __init__(self,h,n):
  super().__init__();self.norm=torch.nn.LayerNorm(h,dtype=torch.bfloat16);self.linear=torch.nn.Linear(h,n,bias=False,dtype=torch.bfloat16)
 def forward(self,x):return self.linear(self.norm(x))

class EditLens:
 def __init__(self,name,quant=False):
  from pathlib import Path
  from huggingface_hub import snapshot_download
  from transformers import AutoModelForSequenceClassification,AutoTokenizer
  def fetch(repo):return snapshot_download(repo,revision=PINS[repo],allow_patterns=['*.json','*.safetensors','*.txt'])
  self.limit=512 if name=='roberta' else 1024
  path=fetch('ml-resources/editlens_roberta-large' if name=='roberta' else 'ml-resources/editlens_Llama-3.2-3B')
  self.tok=AutoTokenizer.from_pretrained(path)
  if name=='roberta':
   self.model=AutoModelForSequenceClassification.from_pretrained(path,dtype=torch.float32)
   from detectors import materialize
   materialize(self.model)
  else:
   from peft import PeftModel
   from safetensors import safe_open
   assert json.loads((Path(path)/'adapter_config.json').read_text())['base_model_name_or_path']=='meta-llama/Llama-3.2-3B'
   with safe_open(str(Path(path)/'adapter_model.safetensors'),framework='pt') as f:n=next(f.get_tensor(k).shape[0] for k in f.keys() if 'score' in k and 'linear.weight' in k)
   kwargs={}
   if quant:
    from transformers import BitsAndBytesConfig
    kwargs=dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':'cpu'})
   base=AutoModelForSequenceClassification.from_pretrained(fetch('unsloth/Llama-3.2-3B'),num_labels=n,dtype=torch.bfloat16,**kwargs)
   if self.tok.pad_token is None:self.tok.pad_token=self.tok.eos_token
   base.config.pad_token_id=self.tok.pad_token_id;base.config.use_cache=False
   base.score=NormedLinear(base.config.hidden_size,n)
   self.model=PeftModel.from_pretrained(base,path)
  self.model.eval()
 @torch.inference_mode()
 def score(self,text):
  enc=self.tok(clean(text),return_tensors='pt',truncation=False);total=enc['input_ids'].shape[1]
  dev=next(self.model.parameters()).device
  enc=self.tok(clean(text),return_tensors='pt',truncation=True,max_length=self.limit).to(dev)
  with torch.autocast(dev.type,dtype=torch.bfloat16):logits=self.model(**enc).logits.float()[0].cpu()
  p=logits.softmax(-1)
  score=float((p*torch.arange(len(p))).sum()/(len(p)-1))
  if not np.isfinite(score):raise RuntimeError('Non-finite EditLens score')
  return score,total,min(total,self.limit),p.tolist()

def inventory():
 items=[]
 def add(key,text,kind,**meta):items.append(dict(key=key,text=text,kind=kind,**meta))
 pairs=read_jsonl(PAIRS_PATH)
 for p in pairs:
  for side in ['human','ai']:add(f"eval|{p['pair_id']}|{side}",p[side],'eval',pair_id=p['pair_id'],side=side,channel=p['channel'])
 for f in sorted((ROOT/'private/corpus').glob('*.jsonl')):
  for r in read_jsonl(f):add('corpus|'+r['id'],r['text'],'corpus',channel=r['channel'])
 for run in ['archival-pilot',RUN_ID]:
  for p in sorted((ROOT/'private/trials'/run/'raw').glob('*/*/s*.json')):
   r=json.loads(p.read_text());t=(r.get('scored_text') or '').strip()
   if t and not r.get('infra_failure') and (run!='archival-pilot' or r['sample']==1):add(f"{run}|{r['model']}|{r['task']}|s{r['sample']}",t,run,model=r['model'],task=r['task'],sample=r['sample'])
 chosen=random.Random(SEED).sample([r for r in items if r['kind']==RUN_ID],100)
 for r in chosen:
  t=r['text'];typos=t
  eligible=[m for m in re.finditer(r'\b[a-zA-Z]{6,}\b',t) if m.group()[1]!=m.group()[2]]
  assert len(eligible)>=2, 'Two actual word typos require two eligible words'
  for m in reversed(eligible[:2]):
   w=m.group();typos=typos[:m.start()]+w[0]+w[2]+w[1]+w[3:]+typos[m.end():]
  for variant,v in [('markdown',prose_view(t)),('linebreaks',re.sub(r'\s+','\n',t)),('typos',typos)]:add(r['key']+'|'+variant,v,'robustness',original=r['key'],variant=variant)
 for i in [0,28,56]:
  for side in ['human','ai']:add(f'sanity|{i}|{side}',pairs[i][side],'sanity',side=side)
 return items

def run(a):
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True)
 items=inventory()
 if a.model=='llama' and a.subset:
  ids={p['pair_id'] for p in random.Random(SEED).sample(read_jsonl(PAIRS_PATH),a.eval_pairs or a.subset)};keep=[]
  for kind in ['archival-pilot',RUN_ID]:
   rows=[r for r in items if r['kind']==kind and (kind!='archival-pilot' or r['sample']==1)]
   for model in sorted({r['model'] for r in rows}):
    pool=[r for r in rows if r['model']==model];keep+=random.Random(SEED).sample(pool,min(a.subset,len(pool)))
  # A bounded, fixed pilot when full 3B scoring is impractical on this CPU.
  corpus=[r for r in items if r['kind']=='corpus']
  keep+=random.Random(SEED).sample(corpus,min(5*a.subset,len(corpus)))
  robust_originals=sorted({r['original'] for r in items if r['kind']=='robustness'})
  robust_ids=set(random.Random(SEED).sample(robust_originals,min(a.subset,len(robust_originals))))
  keys={r['key'] for r in keep}|robust_ids
  arch_ids={r['task'] for r in keep if r['kind']=='archival-pilot'}
  items=[r for r in items if r['kind']=='sanity' or (r['kind']=='eval' and (r['pair_id'] in ids or (r['side']=='human' and r['pair_id'] in arch_ids))) or r['key'] in keys or (r['kind']=='robustness' and r['original'] in robust_ids)]

 items.sort(key=lambda r:({'sanity':0,'eval':1,'corpus':2,'archival-pilot':3,RUN_ID:4,'robustness':5}[r['kind']],r.get('sample',0)))
 if a.limit:items=items[:a.limit]
 tag=a.model+('-nf4' if a.quant else '');path=OUT/(tag+'.jsonl')
 if path.exists():
  data=path.read_bytes()
  if data and not data.endswith(b'\n'):path.write_bytes(data[:data.rfind(b'\n')+1])
 have={r['key']:r for r in read_jsonl(path)}
 (OUT/(tag+'.manifest.json')).write_text(json.dumps(dict(pins=PINS,seed=SEED,subset=a.subset,eval_pairs=a.eval_pairs,encoding="native-tokenizer-truncation-v1",keys=[r['key'] for r in items],torch=torch.__version__,versions={k:version(k) for k in ["transformers","peft","emoji","bitsandbytes","accelerate"]},threads=a.threads),indent=2))
 start=time.monotonic();det=EditLens(a.model,a.quant);print('loaded',tag,time.monotonic()-start,flush=True)
 with path.open('a') as f:
  for i,r in enumerate(items):
   t=fold(r['text']);sha=hashlib.sha256(t.encode()).hexdigest()
   if have.get(r['key'],{}).get('sha')==sha:continue
   t0=time.monotonic();c0=time.process_time();s,total,used,probs=det.score(t)
   row={k:v for k,v in r.items() if k!='text'};row.update(score=s,sha=sha,tokens=total,used_tokens=used,probs=probs,words=word_count(t),wall_s=time.monotonic()-t0,cpu_s=time.process_time()-c0,rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
   f.write(json.dumps(row)+'\n');f.flush()
   if r['kind']=='sanity' or i%50==0:print(tag,i,len(items),row,flush=True)
 print('DONE',tag,flush=True)


def report():
 pairs=read_jsonl(PAIRS_PATH); lines=[]; results={}
 expected_sha={r['key']:hashlib.sha256(fold(r['text']).encode()).hexdigest() for r in inventory()}
 def flags(rows,thr):
  n=sum(r['score']>thr for r in rows)
  return f'{n} ({n/len(rows):.1%})' if rows else 'N/A'
 def table(headers,rows):
  lines.extend(['','| '+' | '.join(headers)+' |','|'+'---|'*len(headers)])
  lines.extend('| '+' | '.join(map(str,r))+' |' for r in rows)
 evalrows=[];corrows=[];archrows=[];devrows=[];robrows=[];costrows=[];sanrows=[]
 baseline=json.loads((EVAL_DIR/'results/summary.json').read_text())
 for name in ['vanguard-modernbert-large__fold','tropa-mini-deberta-large__fold','probe-tropa-mini-deberta-large__fold']:
  d=baseline[name];evalrows.append([name,140]+[fmt_ci(d['metrics'][k],d['ci'][k]) for k in ['auroc','tpr1','tpr5','pairwise']])
 for path in sorted(OUT.glob('*.jsonl')):
  tag=path.stem;by={r['key']:r for r in read_jsonl(path)};rows=list(by.values())
  manifest=json.loads(path.with_suffix('.manifest.json').read_text())
  assert manifest['pins']==PINS, 'Model revisions changed; rescore'
  missing=set(manifest['keys'])-set(by)
  stale=[k for k,r in by.items() if expected_sha.get(k)!=r['sha']]
  if missing or stale:raise RuntimeError(f'{tag}: {len(missing)} missing, {len(stale)} stale scores; resume scoring before reporting')
  ps=[p for p in pairs if all(f"eval|{p['pair_id']}|{s}" in by for s in ['human','ai'])]
  if not ps:
   if rows:
    costrows.append([tag+' (sanity timing only)',len(rows),f"{sum(r['wall_s'] for r in rows)/sum(r['words'] for r in rows)*500:.2f}",f"{np.median([r['wall_s']/r['words']*500 for r in rows if r['words']]):.2f}",f"{sum(r['cpu_s'] for r in rows)/sum(r['words'] for r in rows)*500:.2f}",f"{max(r['rss_mb'] for r in rows):.0f}",f"{np.mean([r['tokens']>r['used_tokens'] for r in rows]):.1%}"])
    for side in ['human','ai']:
     rr=[r for r in rows if r['kind']=='sanity' and r['side']==side]
     sanrows.append([tag,side,len(rr),', '.join(f"{r['score']:.4f}" for r in rr)])
   continue
  h=np.array([by[f"eval|{p['pair_id']}|human"]['score'] for p in ps]);a=np.array([by[f"eval|{p['pair_id']}|ai"]['score'] for p in ps])
  m=metrics(h,a);ci=boot(h,a);results[tag]=dict(metrics=m,ci=ci)
  evalrows.append([tag,len(ps)]+[fmt_ci(m[k],ci[k]) for k in ['auroc','tpr1','tpr5','pairwise']])
  if tag.startswith('llama') and len(ps)<len(pairs):
   for bn in ['vanguard-modernbert-large','tropa-mini-deberta-large','probe-tropa-mini-deberta-large','editlens-roberta']:
    br_rows=read_jsonl(OUT/'roberta.jsonl') if bn=='editlens-roberta' else read_jsonl(EVAL_DIR/'scores'/f'{bn}__fold.jsonl')
    br={(r['pair_id'],r['side']):r['score'] for r in br_rows if 'pair_id' in r}
    if all((p['pair_id'],side) in br for p in ps for side in ['human','ai']):
     bh=np.array([br[p['pair_id'],'human'] for p in ps]);ba=np.array([br[p['pair_id'],'ai'] for p in ps])
     bm=metrics(bh,ba);bc=boot(bh,ba)
     evalrows.append([bn+' / same Llama subset',len(ps)]+[fmt_ci(bm[k],bc[k]) for k in ['auroc','tpr1','tpr5','pairwise']])
  for mode in ['default','instructed']:
   ix=[i for i,p in enumerate(ps) if p['mode']==mode]
   if ix:
    mm=metrics(h[ix],a[ix]);cc=boot(h[ix],a[ix]);evalrows.append([tag+'/'+mode,len(ix)]+[fmt_ci(mm[k],cc[k]) for k in ['auroc','tpr1','tpr5','pairwise']])
  humans=[r for r in rows if r['kind']=='corpus' or (r['kind']=='eval' and r['side']=='human')]
  if len(humans)!=801: print('SUBSET calibration only; not comparable to pooled 801:',tag,len(humans))
  thr=threshold(np.array([r['score'] for r in humans]));results[tag]['threshold']=thr;results[tag]['n_calibration']=len(humans)
  eth=threshold(h)
  cor=[r for r in rows if r['kind']=='corpus']
  for ch in sorted({r['channel'] for r in cor}):
   rr=[r for r in cor if r['channel']==ch];corrows.append([tag,ch,len(rr),flags(rr,eth),flags(rr,thr) if len(humans)==801 else 'N/A (subset calibration)'])
  for model in sorted({r['model'] for r in rows if r['kind']=='archival-pilot'}):
   rr=[r for r in rows if r['kind']=='archival-pilot' and r['model']==model and r['sample']==1]
   hh=np.array([by[f"eval|{r['task']}|human"]['score'] for r in rr]);aa=np.array([r['score'] for r in rr])
   cc=boot(hh,aa)['pairwise'];archrows.append([tag,model,len(rr),fmt_ci(pairwise(hh,aa),cc)])
  for model in sorted({r['model'] for r in rows if r['kind']==RUN_ID}):
   rr=[r for r in rows if r['kind']==RUN_ID and r['model']==model]
   devrows.append([tag,model,len(rr),(f"{sum(r['score']<=thr for r in rr)/len(rr):.1%}" if len(humans)==801 else f"N/A; pilot {sum(r['score']<=thr for r in rr)/len(rr):.1%} at n={len(humans)} calibration")])
  for variant in ['markdown','linebreaks','typos']:
   rr=[r for r in rows if r['kind']=='robustness' and r['variant']==variant and r['original'] in by]
   if rr:robrows.append([tag,variant,len(rr),sum((r['score']>thr)!=(by[r['original']]['score']>thr) for r in rr),sum(by[r['original']]['score']>thr and r['score']<=thr for r in rr),sum(by[r['original']]['score']<=thr and r['score']>thr for r in rr),f"{np.mean([abs(r['score']-by[r['original']]['score']) for r in rr]):.6f}"])
  timed=[r for r in rows if r['kind']!='sanity']
  costrows.append([tag,len(timed),f"{sum(r['wall_s'] for r in timed)/sum(r['words'] for r in timed)*500:.2f}",f"{np.median([r['wall_s']/r['words']*500 for r in timed if r['words']]):.2f}",f"{sum(r['cpu_s'] for r in timed)/sum(r['words'] for r in timed)*500:.2f}",f"{max(r['rss_mb'] for r in rows):.0f}",f"{np.mean([r['tokens']>r['used_tokens'] for r in timed]):.1%}"])
  for side in ['human','ai']:
   rr=[r for r in rows if r['kind']=='sanity' and r['side']==side]
   sanrows.append([tag,side,len(rr),', '.join(f"{r['score']:.4f}" for r in rr)])
 # Reuse cached operational baseline corpus and archival scores.
 channels={r['key']:r['channel'] for r in inventory() if r['kind']=='corpus'}
 arch_judge=ROOT/'private/trials/archival-pilot/judge'
 for name in ['vanguard-modernbert-large','tropa-mini-deberta-large']:
  er=read_jsonl(EVAL_DIR/'scores'/f'{name}__fold.jsonl')
  eh={r['pair_id']:r['score'] for r in er if r['side']=='human'}
  cr=list({r['key']:r for r in read_jsonl(EVAL_DIR/'calibration'/f'{name}.jsonl')}.values())
  th=threshold(np.array(list(eh.values())+[r['score'] for r in cr]));et=threshold(np.array(list(eh.values())))
  for ch in sorted(set(channels.values())):
   rr=[r for r in cr if channels.get(r['key'])==ch]
   if rr:corrows.append([name,ch,len(rr),flags(rr,et),flags(rr,th)])
  ar=read_jsonl(arch_judge/f'trials__{name}.jsonl')
  for model in sorted({r['key'].split('|')[0] for r in ar}):
   rr=[r for r in ar if r['key'].split('|')[0]==model and r['key'].endswith('|s1')]
   if rr:
    hh=np.array([eh[r['key'].split('|')[1]] for r in rr]);aa=np.array([r['score'] for r in rr])
    archrows.append([name,model,len(rr),fmt_ci(pairwise(hh,aa),boot(hh,aa)['pairwise'])])
 # Existing dev scores and calibration thresholds, without recomputing a trained probe.
 judge=ROOT / 'private' / 'trials' / RUN_ID / 'judge';ths=json.loads((judge/'thresholds.json').read_text())
 for name,info in ths.items():
  rr=read_jsonl(judge/f'trials__{name}.jsonl')
  for model in sorted({r['key'].split('|')[0] for r in rr}):
   ss=[r['score'] for r in rr if r['key'].split('|')[0]==model]
   devrows.append([name,model,len(ss),f"{np.mean(np.array(ss)<=info['threshold']):.1%}"])
 lines+=['### Judge eval (fold + official EditLens preprocessing; 95% paired bootstrap CIs)']
 table(['judge','pairs','AUROC','TPR@1% FPR','TPR@5% FPR','pairwise'],evalrows)
 if any(v['metrics']['n']<140 for v in results.values()):lines+=['','Small-subset bootstrap intervals can be degenerate: even 20/20 pairwise successes have a Wilson 95% interval of [0.839, 1.000]. A perfect bootstrap interval is not evidence of zero population error. With 20 eval humans, the empirical 1%-FPR threshold is their maximum score (zero observed false positives), not a validated 1% population operating point.']
 lines+=['','The 40 instructed mirrors are already inside the 140 pairs; they are not 40 additional human texts.']
 lines+=['','### Corpus human false positives by channel']
 table(['judge','channel','n','flags at eval-human 1% threshold','flags at pooled-801 1% threshold'],corrows)
 lines+=['','Pooled thresholds: '+', '.join(f"{k} = {v['threshold']:.9f} (n={v['n_calibration']})" for k,v in results.items())+'. Strict score > threshold flags AI; ties read as human. The pooled FPR is calibration performance, not held-out validation. Any Llama subset threshold is exploratory and does not measure the requested pooled-801 operating point; N/A marks this distinction.']
 lines+=['','### Archival pilot, sample 1 only']
 table(['judge','generator','n','pairwise [95% CI]'],archrows)
 lines+=['','### Dev outputs reading as human at the listed human-calibration threshold']
 table(['judge','generator','n','reads human'],devrows)
 lines+=['','### Robustness on fixed random dev outputs']
 table(['judge','change','n','verdict flips','AI→human','human→AI','mean absolute score change'],robrows)
 lines+=['','Line-break variants are identical after fold by design. Markdown stripping uses common.prose_view; typos transpose distinct second/third letters in the first two eligible words of at least six letters (exactly two changed words).']
 lines+=['','### CPU cost (at most two Torch threads, shared machine)']
 table(['judge','scored texts','mean wall s/500 original words','median wall s/500 words','CPU s/500 words','peak RSS MiB','texts truncated'],costrows)
 lines+=['','Timings mix one- and two-thread segments (one thread after host contention). Times exclude loading and include tokenization; original-word normalization understates full-document cost when truncating. These are shared-machine throughput measurements, not idle benchmarks.']
 lines+=['','### Six-example direction sanity check']
 table(['judge','known origin','n','scores (more AI → larger)'],sanrows)
 (OUT/'aggregate.json').write_text(json.dumps(results,indent=2));(OUT/'tables.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def check_encoding():
 """Regression check runs the real score method with a capturing fake head."""
 from huggingface_hub import snapshot_download
 from transformers import AutoTokenizer
 from types import SimpleNamespace
 path=snapshot_download('ml-resources/editlens_roberta-large',revision=PINS['ml-resources/editlens_roberta-large'],local_files_only=True,allow_patterns=["*.json","*.safetensors","*.txt"])
 det=EditLens.__new__(EditLens);det.tok=AutoTokenizer.from_pretrained(path);det.limit=512
 def capture(**enc):
  assert enc['input_ids'].shape==(1,512)
  assert enc['input_ids'][0,-1].item()==det.tok.eos_token_id
  return SimpleNamespace(logits=torch.zeros(1,4))
 det.model=capture
 score,total,used,probs=det.score('A long human-written sentence. '*300)
 assert total>512 and used==512 and score==0.5
 print('PASS: native truncation preserves EOS; uniform four-bucket expectation is 0.5')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['roberta','llama'],default='roberta');ap.add_argument('--quant',action='store_true');ap.add_argument('--subset',type=int,default=0);ap.add_argument('--report',action='store_true');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--threads',type=int,choices=[1,2],default=2);ap.add_argument('--self-check',action='store_true');ap.add_argument('--eval-pairs',type=int,default=0)
 a=ap.parse_args()
 if a.self_check:check_encoding()
 elif a.report:report()
 else:run(a)
