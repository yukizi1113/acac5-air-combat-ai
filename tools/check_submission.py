"""Verify a submission zip with the official validate.py (same flow as the evaluation server's initial check).

  python tools/check_submission.py submit/acac5.zip [--n 3] [--sim-dir /path/to/simulator_dist]
Extracts the zip into <sim-dir>/Agents/_submit_check and runs validate.py against the benchmark
(3 matches, 3.0 s per-step timeout, 7.5 GB memory limit), then prints the result summary.
"""
import os
import sys
import json
import shutil
import zipfile
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--sim-dir', default=os.environ.get('SIM_DIR', os.path.join(ROOT, 'simulator_dist')))
    ap.add_argument('--color', default='Red')
    a = ap.parse_args()
    sim = os.path.abspath(a.sim_dir)
    dst = os.path.join(sim, 'Agents', '_submit_check')
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(dst)
    size = os.path.getsize(a.zip) / 1024 ** 2
    print(f'zip size {size:.2f} MB (limit 64 MB)')
    cmd = [sys.executable, 'validate.py', '--myagent-module-path', 'Agents/_submit_check', '--num-validation', str(a.n),
           '--result-dir', './results_check', '--make-log', '0', '--color', a.color]
    print(' '.join(cmd))
    p = subprocess.run(cmd, cwd=sim, capture_output=True, text=True)
    tail = '\n'.join(p.stdout.splitlines()[-15:])
    print(tail)
    if p.returncode != 0:
        print(p.stderr[-3000:])
    rp = os.path.join(sim, 'results_check', '0', 'validation_results.json')
    if os.path.exists(rp):
        r = json.load(open(rp))
        print('status', r.get('status'), r.get('message'), 'win/loss/draw', r.get('win'), r.get('loss'), r.get('draw'))
        for k, d in r['details'].items():
            print(k, d['endReason'], d['scores'], 'error' if d['error'] else 'ok', (d['red_error_message'] + d['blue_error_message'])[-500:])
    print('TimeOut occurrences:', p.stdout.count('TimeOut'))


if __name__ == '__main__':
    main()
