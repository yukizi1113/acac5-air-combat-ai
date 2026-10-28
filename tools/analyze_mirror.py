"""Why matches are won or lost: escort kill times, fighter attrition before the decisive event.

  python tools/analyze_mirror.py v8c0 [v8alt ...]
"""
import sys, json
import numpy as np


def main(tags):
    for tag in tags:
        rows = [json.loads(l) for l in open(f'results/col_{tag}.jsonl')]
        rows = [r for r in rows if 'score' in r]
        print(f'== {tag}: n={len(rows)}')
        stats = {'win': [], 'loss': [], 'Draw': []}
        for r in rows:
            me = r['color']; op = 'Blue' if me == 'Red' else 'Red'
            ev = r.get('events', [])
            my_esc = [e[0] for e in ev if e[1] == 'hit' and e[3] == f'{me}/{me}E']
            op_esc = [e[0] for e in ev if e[1] == 'hit' and e[3] == f'{op}/{op}E']
            my_f = [e[0] for e in ev if e[3].startswith(f'{me}/{me}F')]
            op_f = [e[0] for e in ev if e[3].startswith(f'{op}/{op}F')]
            # first escort shot launch times (god log)
            gl = r.get('godlog') or []
            my_esc_shots = sorted(m['t0'] for m in gl if m['team'] == me and m['target'].endswith('E'))
            op_esc_shots = sorted(m['t0'] for m in gl if m['team'] == op and m['target'].endswith('E'))
            stats.setdefault(r['result'], []).append(r)
            print(f"  {r['result']:5s} {me:4s} t={r['time']:5.0f} | our escort shot@{my_esc_shots[:3]} killed@{op_esc} | "
                  f"their escort shot@{op_esc_shots[:3]} killed our escort@{my_esc} | fighters lost us={len(my_f)} them={len(op_f)}")
        for k, v in stats.items():
            if v:
                print(f'  {k}: {len(v)}  mean time {np.mean([r["time"] for r in v]):.0f}')


if __name__ == '__main__':
    main(sys.argv[1:])
