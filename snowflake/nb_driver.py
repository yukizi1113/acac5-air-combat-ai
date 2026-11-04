"""Notebook-side driver (runs in the Snowflake notebook kernel, any Python version; the matches run in /tmp/acenv).

    import sys; sys.path.insert(0, '/tmp/acac5/acac'); import nb_driver as D
    D.run(session, STAGE, 'phase1', budget_min=12)                 # one node
    D.run(session, STAGE, 'phase2', shard='0/3', budget_min=80)    # node i of k

- Results are copied to the stage every `put_every_min` minutes and at the end, so a lost session loses little.
- Progress lines show matches/h and the load average (CPU utilisation check).
- A hard kill fires at budget + 12 min even if snowrun's own budget logic failed.
"""
import os, sys, json, time, subprocess, collections

ROOT = os.path.dirname(os.path.abspath(__file__))


def _put(session, path, stage_dir):
    try:
        session.file.put(path, stage_dir, auto_compress=False, overwrite=True)
    except Exception as ex:
        print('stage put failed (will retry):', ex)


def run(session, stage, phase, shard='0/1', budget_min=20, workers=0, put_every_min=5, poll_s=30):
    si = shard.replace('/', 'of')
    out = f'{ROOT}/out/{phase}_{si}.jsonl'
    log = f'{ROOT}/out/{phase}_{si}.log'
    os.makedirs(f'{ROOT}/out', exist_ok=True)
    cmd = (f'. /tmp/acenv/ldpath.sh && cd {ROOT} && exec /tmp/acenv/bin/python -u snowrun.py --jobs jobs/{phase}.jsonl '
           f'--out {out} --shard {shard} --budget-min {budget_min} --workers {workers}')
    t0 = time.time(); last_put = t0
    with open(log, 'a') as lf:
        p = subprocess.Popen(['bash', '-c', cmd], stdout=lf, stderr=subprocess.STDOUT)
    stage_dir = f'{stage}/results/'
    try:
        while p.poll() is None:
            time.sleep(poll_s)
            n = sum(1 for _ in open(out)) if os.path.exists(out) else 0
            el = (time.time() - t0) / 60
            print(f'{el:5.1f} min  {n} results  load {os.getloadavg()[0]:.1f}', flush=True)
            if time.time() - last_put > put_every_min * 60 and n:
                _put(session, out, stage_dir); last_put = time.time()
            if el > budget_min + 12:
                print('driver hard kill'); p.kill(); break
    finally:
        if p.poll() is None:
            p.kill()
        for f in (out, log):
            if os.path.exists(f):
                _put(session, f, stage_dir)
    print(open(log).read()[-3000:])
    print(f'driver done in {(time.time() - t0) / 60:.1f} min; results in {stage_dir}')
    return out


def check_det(out, expected=f'{ROOT}/jobs/phase1_expected.jsonl'):
    """Same seed + colour must give the same result as the local run (cross-machine determinism gate)."""
    exp = {(r['seed'], r['color']): r for r in map(json.loads, open(expected))}
    ok = bad = 0
    for l in open(out):
        r = json.loads(l)
        if r.get('error'):
            print('ERROR', r['error']); bad += 1; continue
        e = exp.get((r['seed'], r['color']))
        if r.get('tag') == 'det_v17_v15' and e:
            same = r['result'] == e['result'] and abs(r['score'] - e['score']) < 1e-9 and r['time'] == e['time']
            ok += same; bad += not same
            if not same:
                print('MISMATCH', r['seed'], r['color'], r['result'], r['score'], r['time'], 'expected', e)
    print(f'determinism: {ok} identical, {bad} mismatched/errors')
    return bad == 0
