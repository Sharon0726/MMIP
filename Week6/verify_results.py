"""Audit saved experimental evidence, without GPU, network or model inference.

Usage: python verify_results.py
Only Python's standard library is required. Nonzero exit means verification failed.
"""
import ast
import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'artifacts'
CHECKS = []


def check(ok, label):
    if not ok:
        raise AssertionError(label)
    CHECKS.append(label)


def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def close(a, b, label):
    check(math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), label)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def final(text):
    text = re.split(r'FINAL\s*[:：]', str(text), flags=re.I)[-1].strip()
    return re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()


def csv_rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def score_text(item, answer, terms):
    text = final(answer)
    gold = item['gold']
    if item['metric'] == 'translation':
        return mean(any(t.lower() in text.lower() for t in g) for g in terms[item['id']])
    try:
        if item['metric'] == 'json':
            return float(json.loads(text) == gold)
        if item['metric'] == 'number':
            return float(float(text.replace(',', '')) == float(gold))
    except (ValueError, TypeError):
        return 0.
    return float(text.strip('。.!！ \n') == str(gold))


def score_visual(item, answer):
    text = final(answer).lower().strip(' .')
    category = item['category']
    if category == 'ocr':
        return float(text == item['gold'].lower())
    if category == 'counting':
        return float(text in item['groups'][0])
    if category == 'spatial':
        return float(item['groups'][-1][0].lower() in text)
    if item['id'] == 'B20':
        concepts = sum(any(t.lower() in text for t in g) for g in item['groups'][:4])
        relation = bool(re.search(r'blue circle.{0,70}left.{0,70}red triangle|red triangle.{0,70}right.{0,70}blue circle', text))
        return (concepts + relation) / 5
    return mean(any(t.lower() in text for t in g) for g in item['groups'])


def local_image(item):
    # Resolve the archived Colab prefix, preserving original evidence unchanged.
    rel = item['image'].split('/content/mmip_week6/', 1)[-1]
    path = (DATA / rel).resolve()
    check(path.is_relative_to(DATA.resolve()) and path.is_file(), 'image exists: ' + item['id'])
    return path


def audit_weights(folder):
    path = DATA / folder / 'adapter_model.safetensors'
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    tensors = parameters = nonzero_b = 0
    for key, spec in header.items():
        if key == '__metadata__':
            continue
        check(spec['dtype'] == 'F32', folder + ': expected recorded FP32 adapter tensor')
        start, end = spec['data_offsets']
        count = math.prod(spec['shape'])
        check(end - start == 4 * count and 0 <= start < end <= len(data), folder + ': tensor bounds')
        values = struct.iter_unpack('<f', data[start:end])
        for (value,) in values:
            if not math.isfinite(value):
                raise AssertionError(folder + ': nonfinite weight')
            if 'lora_B' in key and value != 0:
                nonzero_b += 1
        tensors += 1
        parameters += count
    check(nonzero_b > 0, folder + ': actual nonzero LoRA B weights')
    return dict(tensors=tensors, nonzero_lora_B=nonzero_b, trainable_parameters=parameters)


def run():
    nb = json.loads((ROOT / 'MMIP_Week6_Executed.ipynb').read_text(encoding='utf-8'))
    for notebook in ROOT.glob('*.ipynb'):
        obj = json.loads(notebook.read_text(encoding='utf-8'))
        for i, cell in enumerate(obj['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'{notebook.name}:cell{i}')
                check(not any(o.get('output_type') == 'error' for o in cell.get('outputs', [])), 'no error output ' + notebook.name + str(i))
    terms = next(ast.literal_eval(n.value) for n in ast.parse(''.join(nb['cells'][3]['source'])).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TRANSLATION_TERMS' for t in n.targets))
    a, dev, b = read('A_benchmark.json'), read('A_development.json'), read('B_benchmark.json')
    check(len(a) == 20 and len(dev) == 10 and len(b) == 20, 'A/B benchmark and development counts')
    check(set(x['prompt'] for x in a).isdisjoint(x['prompt'] for x in dev), 'A development/test prompts disjoint')
    check(len(set(x['category'] for x in a)) == 5 and len(set(x['category'] for x in b)) == 5, 'A/B five categories')
    for rows in [a, b]:
        check(len({x['id'] for x in rows}) == 20, 'unique benchmark IDs')
    ar, br = read('A_judged.json'), read('B_v2_judged.json')
    check(len(ar) == 120 and len(br) == 40, 'A/B inference counts')
    cache = read('commercial_judge_cache.json')
    for rows, items, fn in [(ar, a, lambda i,s:score_text(i,s,terms)), (br, b, score_visual)]:
        lookup = {x['id']:x for x in items}
        groups = defaultdict(list)
        for row in rows:
            item = lookup[row['id']]
            check(row['prompt'] == item['prompt'], 'identical benchmark prompt ' + row['id'])
            close(row['objective_score'], fn(item, row['answer']), 'recomputed score ' + row['id'])
            check(row['judge_model'] == 'google/gemini-3.5-flash-lite' and 0 <= row['judge_score'] <= 4, 'commercial judge present')
            check(row['judge_score'] == cache[row['judge_key']]['score'] and row['judge_reason'], 'judge cache and reason')
            groups[row['condition']].append(row)
        for condition, group in groups.items():
            check(len(group) == 20 and {x['id'] for x in group} == set(lookup), 'same 20 IDs: ' + condition)
        summary = csv_rows('A_summary.csv' if rows is ar else 'B_final_summary.csv')
        for row in summary:
            group = groups[row['condition']]
            close(row['objective'], mean(r['objective_score'] for r in group), 'A/B summary objective')
            close(row['judge_0_to_4'], mean(r['judge_score'] for r in group), 'A/B summary judge')
    for item in b:
        local_image(item)
    all_visual = []
    for prefix in ['C', 'V']:
        train, valid, test = [read(f'{prefix}_{name}.json') for name in ['train_preferences','validation_preferences','target_test']]
        check(len(train) >= 100 and len(valid) == 20 and len(test) == (30 if prefix == 'C' else 20), prefix + ': split counts')
        check(all(x['chosen'] != x['rejected'] for x in train + valid), prefix + ': valid preference pairs')
        ids = [x['id'] for x in train + valid + test]
        check(len(ids) == len(set(ids)), prefix + ': disjoint split IDs')
        if prefix == 'C':
            prompts = [x['prompt'] for x in train + valid + test]
            check(len(set(prompts)) == len(prompts), 'C: disjoint prompt text')
        else:
            all_visual = train + valid + test
            hashes = [hashlib.sha256(local_image(x).read_bytes()).hexdigest() for x in all_visual]
            check(len(set(hashes)) == len(hashes), 'V: 160 distinct images across splits')
        logs = read(prefix + '_training_log.json')
        check(max(x.get('step',0) for x in logs) == 60 and max(x.get('epoch',0) for x in logs) == 2, prefix + ': 60 optimizer steps, 2 epochs')
        check(all(math.isfinite(x['loss']) for x in logs if 'loss' in x), prefix + ': finite loss')
    general = read('C_general_benchmark_100.json')
    check(len(general) == 100 and len({x['prompt'] for x in general}) == 100, '100 distinct general questions')
    check('order_extraction' not in {x['category'] for x in general}, 'general categories unrelated to order extraction')
    target = {x['id']:x for x in read('C_target_test.json')}
    gl = {x['id']:x for x in general}
    cr = read('C_final_results.json')
    check(len(cr) == 260, 'C before/after count')
    for phase in ['before','after']:
        rows = [x for x in cr if x['phase'] == phase]
        check(Counter(x['test_set'] for x in rows) == {'target':30,'general':100}, 'C before/after test split')
        g, t = [], []
        for row in rows:
            if row['test_set'] == 'general':
                score = score_text(gl[row['id']], row['answer'], terms)
                close(score, row['objective_score'], 'C general raw score'); g.append(score)
            else:
                score = score_text(target[row['id']], row['answer'], terms)
                close(score, row['exact_match'], 'C target raw exact match'); t.append(score)
        summary = next(x for x in csv_rows('C_final_summary.csv') if x['phase'] == phase)
        close(mean(g), summary['general_accuracy'], 'C general summary')
        close(mean(t), summary['target_exact'], 'C target summary')
    vl = {x['id']:x for x in all_visual}
    vr = read('V_before_results.json') + read('V_after_results.json')
    check(len(vr) == 120, 'V before/after count')
    vg = defaultdict(list)
    for row in vr:
        gold = set(vl[row['id']]['shape_types'])
        pred = set(re.findall(r'\b(circle|square|triangle|star)s?\b', row['answer'].lower()))
        precision = len(pred & gold) / len(pred) if pred else 0
        recall = len(pred & gold) / len(gold)
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        for key, value in dict(shape_f1=f1,shape_recall=recall,hallucination_rate=len(pred-gold)/max(1,len(pred)),exact_set=float(pred==gold)).items():
            close(row[key], value, 'V recomputed ' + key)
        vg[row['phase'],row['condition']].append(row)
    for s in csv_rows('V_summary.csv'):
        rows = vg[s['phase'],s['condition']]
        check(len(rows)==20 and len({x['id'] for x in rows})==20, 'V matched 20 test images')
        for key in ['shape_f1','shape_recall','hallucination_rate','exact_set']:
            close(s[key], mean(r[key] for r in rows), 'V summary '+key)
        close(s['truncated'], sum(r['truncated'] for r in rows), 'V truncation count')
    original = {(r['condition'],r['id']):r for r in ar}
    memory_count = 0
    for summary in csv_rows('A_fresh_process_memory_summary.csv'):
        rows = read('memory_'+summary['condition']+'.json')
        check(len(rows)==20,'memory condition count')
        memory_count += len(rows)
        for row in rows:
            trace = row['gpu_trace']
            area = sum((y1+y0)/2*(t1-t0) for (t0,y0),(t1,y1) in zip(trace,trace[1:]))
            measured = area/(trace[-1][0]-trace[0][0])
            close(measured,row['gpu_mean_mib'],'memory integration')
            check(final(row['answer']) == final(original[summary['condition'],row['id']]['answer']), 'memory rerun answer identical')
        weighted = sum(r['gpu_mean_mib']*r['seconds'] for r in rows)/sum(r['seconds'] for r in rows)
        close(summary['process_gpu_mean_mib'],weighted,'memory weighted summary')
    check(memory_count==100,'100 resource validation answers')
    weights = {name:audit_weights(folder) for name,folder in [('text','C_text_adapter'),('vision','V_adapter')]}
    check(weights==read('adapter_audit.json'),'weight audit matches recorded tensor counts')
    report=dict(status='passed',audit_date='2026-11-09',checks_passed=len(CHECKS),formal_inferences=540,
                resource_validation_inferences=memory_count,adapter_audit=weights,
                scope='Offline re-scoring, evidence consistency, syntax and weight checks; no new GPU training or commercial grading.',
                limitations=['FP16 unquantized baseline converts native BF16 checkpoint for T4.',
                             'Commercial VLM grading uses reference scene annotations, not direct image input.',
                             'Synthetic preferences, one seed, small benchmarks; no guaranteed score or improvement.'])
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    run()
