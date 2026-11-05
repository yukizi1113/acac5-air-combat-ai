# ACAC5 runner setup: stage -> /tmp/acac5, verify sha256, apply small overrides from acac5/boot/, offline install
import os, subprocess, time, hashlib, glob, shutil
from snowflake.snowpark.context import get_active_session
session = get_active_session()
STAGE = '@SNOWFLAKE_LEARNING_DB.CURSOR_MCP.CASMI_STAGE/acac5'
W = '/tmp/acac5'; os.makedirs(W + '/dl', exist_ok=True)
t = time.time()
if not os.path.exists(W + '/acac/nb_driver.py'):
    session.file.get(f'{STAGE}/pack/', W + '/dl')
    parts = sorted(glob.glob(W + '/dl/acac5pack.part*'))
    h = hashlib.sha256()
    with open(W + '/acac5pack.tar', 'wb') as fo:
        for p in parts:
            b = open(p, 'rb').read(); h.update(b); fo.write(b)
    exp = open(W + '/dl/acac5pack.sha256').read().split()[0]
    print(len(parts), 'parts, sha256 ok:', h.hexdigest() == exp, f'{time.time()-t:.0f}s')
    assert h.hexdigest() == exp
    subprocess.run(f'cd {W} && tar xf acac5pack.tar && rm acac5pack.tar', shell=True, check=True)
session.file.get(f'{STAGE}/boot/', W + '/boot')            # fixes made after the pack was uploaded
for f in ('snowrun.py', 'node_setup.sh', 'nb_driver.py'):
    if os.path.exists(f'{W}/boot/{f}'):
        shutil.copy(f'{W}/boot/{f}', f'{W}/acac/{f}'); print('override', f)
if not os.path.exists('/tmp/acenv/.ok'):
    r = subprocess.run(f'bash {W}/acac/node_setup.sh && touch /tmp/acenv/.ok', shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
print(f'setup {time.time()-t:.0f}s')
