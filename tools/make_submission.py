"""Build the submission zip (package files at the zip root, like sample_submit.zip).

  python tools/make_submission.py --policy rule --out submit/acac5_rule.zip
  python tools/make_submission.py --policy nn --weights runs/ppo/policy.pt --out submit/acac5_nn.zip
Then verify:  python tools/check_submission.py submit/acac5_nn.zip
"""
import os
import json
import shutil
import zipfile
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILES = ['__init__.py', 'agent.py', 'fusion.py', 'tracker.py', 'vthreat.py', 'rl_obs.py', 'nn_model.py', 'pk_model.py', 'config.json']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='agents/Tac')
    ap.add_argument('--policy', choices=['rule', 'nn'], default='rule')
    ap.add_argument('--weights', default='')
    ap.add_argument('--pk', default='agents/Tac/pk_model.json')
    ap.add_argument('--overrides', default='{}')
    ap.add_argument('--out', default='submit/acac5.zip')
    a = ap.parse_args()
    src = os.path.join(ROOT, a.src)
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    args = {'policy': a.policy, 'greedy': True, 'overrides': json.loads(a.overrides)}
    if a.policy == 'nn':
        args['weights'] = 'policy.pt'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(src, f), f)
        if a.pk and os.path.exists(os.path.join(ROOT, a.pk)):
            z.write(os.path.join(ROOT, a.pk), 'pk_model.json')
        if a.policy == 'nn':
            z.write(os.path.join(ROOT, a.weights), 'policy.pt')
        z.writestr('args.json', json.dumps(args, indent=1))
    print('wrote', out, os.path.getsize(out) / 1e6, 'MB')


if __name__ == '__main__':
    main()
