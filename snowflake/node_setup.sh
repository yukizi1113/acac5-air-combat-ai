#!/bin/bash
# Run once per node (offline). Creates an isolated venv so the notebook's own packages are untouched.
# Python: uses python3.10 if the runtime has it, otherwise the bundled portable CPython 3.10 (../py310.tar.gz).
set -e
cd "$(dirname "$0")"
VENV=${VENV:-/tmp/acenv}
PY=${PY:-python3.10}
if ! command -v $PY >/dev/null || ! $PY -c 'import sys; assert sys.version_info[:2]==(3,10)' 2>/dev/null; then
  if [ -f ../py310.tar.gz ]; then
    mkdir -p /tmp/py310 && tar xzf ../py310.tar.gz -C /tmp/py310
    PY=/tmp/py310/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
    echo "using bundled Python: $PY"
  else
    echo "NEED PYTHON 3.10 (simulator binaries are cp310)"; exit 2
  fi
fi
ldd --version | head -1          # simulator needs GLIBC >= 2.34
$PY -m venv $VENV
$VENV/bin/python -m pip install -q --no-index --find-links ../wheels \
  asrcaisim1 r8_contest_models basicagentutility basicairtoaircombatmodels01 lightgbm==4.5.0 numpy==1.26.4 scipy \
  gymnasium==1.0.0 pygame pyopengl "setuptools<70" cloudpickle
$VENV/bin/python -m pip install -q --no-index --no-deps --find-links ../wheels matchmaker rayutility
$VENV/bin/python patch_noray.py
PYLIB=$(dirname $(dirname $(readlink -f $PY)))/lib   # real location; sysconfig LIBDIR is the build-time path
echo "export LD_LIBRARY_PATH=$PYLIB:\$LD_LIBRARY_PATH" > $VENV/ldpath.sh
. $VENV/ldpath.sh
$VENV/bin/python -c "import ASRCAISim1, r8_contest_models, lightgbm; print('simulator import OK')"
# CPUs actually usable: the cgroup quota can be lower than nproc (GPU_NV_S = 6 vCPU, CPU_X64_S = 3 vCPU)
echo "nproc $(nproc)  cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
$VENV/bin/python -c "import os; print('usable CPUs', os.environ.get('ACAC_WORKERS') or __import__('snowrun').usable_cpus())"
free -g | head -2
