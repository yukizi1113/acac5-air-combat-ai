"""Link the official simulator files into this project root (run once after unpacking).

  python tools/setup_links.py --sim-dir ../simulator_dist
Creates  common/ src/ Agents/  (symlinks; falls back to copies) and simulator_dist link used by
tools/check_submission.py.
"""
import os
import shutil
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def link(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-dir', required=True)
    a = ap.parse_args()
    sim = os.path.abspath(a.sim_dir)
    for d in ('common', 'src', 'Agents'):
        link(os.path.join(sim, d), os.path.join(ROOT, d))
    link(sim, os.path.join(ROOT, 'simulator_dist'))
    for d in ('results', 'data', 'runs', 'submit'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    print('ok:', ROOT)


if __name__ == '__main__':
    main()
