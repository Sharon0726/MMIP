import os,sys,json,re,random,time,gc,math,hashlib,threading
from pathlib import Path
import numpy as np
import torch,pynvml
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
ROOT=Path('/content/mmip_week6');SEED=20261104
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
ENV=json.loads((ROOT/'environment.json').read_text());MODEL_ID=ENV['model'];MODEL_REV=ENV['revision']
pynvml.nvmlInit();GPU_HANDLE=pynvml.nvmlDeviceGetHandleByIndex(0)
def process_mib():
    ps=pynvml.nvmlDeviceGetComputeRunningProcesses(GPU_HANDLE)
    return sum(p.usedGpuMemory for p in ps if p.pid==os.getpid())/2**20
TRANSLATION_TERMS={
 'A13':[['submit','hand in','turn in'],['report'],['before friday','by friday']],
 'A14':[['會議','会议'],['延期','延後','延后','推遲','推迟'],['大雨','豪雨']],
 'A15':[['model'],['recognize','recognise','identify','detect'],['object'],['image','picture']],
 'A16':[['離開','离开'],['房間','房间'],['關燈','关灯','關閉燈','关闭灯']],
 'D07':[['busy'],['today']], 'D08':[['火車','列車','火车','列车'],['晚','遲','迟','誤點','误点']]}
def final_answer(text):
    s=str(text).strip()
    parts=re.split(r'(?i)FINAL\s*[:：]',s)
    if len(parts)>1:s=parts[-1].strip()
    return re.sub(r'^\x60{3}(?:json)?\s*|\s*\x60{3}$','',s).strip()
def objective(item, text):
    s=final_answer(text); metric=item['metric']; gold=item['gold']
    if metric=='json':
        try:return float(json.loads(s)==gold)
        except Exception:return 0.0
    if metric=='number':
        try:return float(float(s.replace(',',''))==float(gold))
        except Exception:return 0.0
    if metric=='translation':
        groups=TRANSLATION_TERMS[item['id']]
        return sum(any(term.lower() in s.lower() for term in group) for group in groups)/len(groups)
    return float(s.strip('。.!！ \n')==str(gold))
def clean_gpu():
    gc.collect(); torch.cuda.empty_cache()
def quant_config(bits):
    if bits==4:return BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    if bits==8:return BitsAndBytesConfig(load_in_8bit=True)
    return None
def load_llm(bits=16):
    clean_gpu()
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV)
    tok.pad_token=tok.eos_token
    kw=dict(revision=MODEL_REV,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
    if bits!=16:kw['quantization_config']=quant_config(bits)
    m=AutoModelForCausalLM.from_pretrained(MODEL_ID,**kw).eval()
    print('LOADED',bits,'bit; model footprint MiB=',round(m.get_memory_footprint()/2**20,1),flush=True)
    return m,tok
class GPUMonitor:
    def __enter__(self):
        self.stop=threading.Event(); self.samples=[]; self.t0=time.perf_counter()
        def worker():
            while not self.stop.is_set():
                self.samples.append((time.perf_counter()-self.t0,process_mib()))
                self.stop.wait(0.1)
        self.thread=threading.Thread(target=worker,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.samples.append((time.perf_counter()-self.t0,process_mib()))
        self.stop.set(); self.thread.join()
    def metrics(self):
        ts=np.array([x[0] for x in self.samples]); mem=np.array([x[1] for x in self.samples])
        mean=float(np.trapezoid(mem,ts)/(ts[-1]-ts[0])) if len(ts)>1 and ts[-1]>ts[0] else float(mem.mean())
        return dict(gpu_mean_mib=mean,gpu_peak_mib=float(mem.max()),gpu_samples=len(mem),gpu_trace=self.samples)
EXAMPLES={
 '算術':'題：3袋米，每袋5公斤，總重？只回答數字。\nFINAL: 15\n題：60元付100元找多少？只回答數字。\nFINAL: 40',
 '邏輯':'題：甲早於乙，乙早於丙，誰最早？\nFINAL: 甲\n題：所有鳥會飛，因此會飛的一定是鳥嗎？回答是或否。\nFINAL: 否',
 '閱讀':'題：店家週六12點開門。週六幾點開門？只回答數字。\nFINAL: 12\n題：新地址為C303。只回答代碼。\nFINAL: C303',
 '翻譯':'題：Translate into English: 我喜歡閱讀。\nFINAL: I like reading.\n題：Translate into Traditional Chinese: The sky is blue.\nFINAL: 天空是藍色的。',
 '格式':'題：輸出JSON陣列，排序8,5,6。\nFINAL: [5,6,8]\n題：bird轉大寫，只輸出結果。\nFINAL: BIRD'}
def messages(item,style='baseline'):
    system='你是一位精確的助理。遵守題目要求，使用繁體中文（翻譯題依指定語言）。最後以 FINAL: 開頭給出最終答案。'
    prompt=item['prompt']
    if style=='fewshot':prompt='以下是其他例題，請依相同方式回答新題。\n'+EXAMPLES[item['category']]+'\n新題：'+prompt
    if style=='cot':prompt='請先列最多三個簡短解題步驟，再以 FINAL: 給出最終答案；FINAL後嚴格遵守題目格式。\n'+prompt
    return [{'role':'system','content':system},{'role':'user','content':prompt}]
def infer_llm(m,tok,item,style='baseline',max_new_tokens=192):
    encoded=tok.apply_chat_template(messages(item,style),tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with GPUMonitor() as monitor:
        with torch.inference_mode():
            out=m.generate(encoded,attention_mask=torch.ones_like(encoded),max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
    elapsed=time.perf_counter()-start; tokens=out[0,encoded.shape[1]:]
    answer=tok.decode(tokens,skip_special_tokens=True)
    row=dict(id=item['id'],category=item['category'],prompt=item['prompt'],gold=item['gold'],answer=answer,
      objective_score=objective(item,answer),seconds=elapsed,input_tokens=encoded.shape[1],output_tokens=len(tokens),
      tokens_per_second=len(tokens)/elapsed,truncated=bool(len(tokens)>=max_new_tokens),
      torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,**monitor.metrics())
    return row
def run_items(m,tok,items,condition,style='baseline',file='A_results.json'):
    result=[]
    for i,item in enumerate(items):
        r=infer_llm(m,tok,item,style); r['condition']=condition; result.append(r)
        print(condition,f'{i+1}/{len(items)}',item['id'],'score=',round(r['objective_score'],2),'answer=',final_answer(r['answer'])[:90],flush=True)
    return result

bits=int(sys.argv[1]);style=sys.argv[2];condition=sys.argv[3]
items=json.loads((ROOT/'A_benchmark.json').read_text())
model,tok=load_llm(bits)
footprint=model.get_memory_footprint()/2**20
_=infer_llm(model,tok,items[0],style,max_new_tokens=16)
rows=[]
for i,item in enumerate(items):
    r=infer_llm(model,tok,item,style);r.update(condition=condition,model_footprint_mib=footprint,measurement='fresh process NVML usedGpuMemory');rows.append(r)
(ROOT/('memory_'+condition+'.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
print('MEMORY_WORKER_COMPLETE',condition,len(rows),flush=True)
