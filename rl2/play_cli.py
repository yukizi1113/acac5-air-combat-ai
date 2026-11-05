"""Run one episode in this process (Python 3.10 simulator env):  python rl2/play_cli.py job.json out.pkl.gz
Used by the Ray driver on Snowflake (the driver runs in the notebook's Python, the simulator needs 3.10)."""
import os
import sys
import json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
os.environ.setdefault('OPENBLAS_CORETYPE', 'Prescott')
from rl2.rollout import play      # noqa: E402
from rl2.rio import save_rollout  # noqa: E402

if __name__ == '__main__':
    job = json.load(open(sys.argv[1]))
    save_rollout(play(job), sys.argv[2])
