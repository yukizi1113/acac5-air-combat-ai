"""Make the installed rayUtility plugin importable without ray (batch nodes only run matches, never ray).
Only guards imports; nothing else changes. Idempotent, and writes new files (never edits a hard-linked cache copy).
  python patch_noray.py"""
import os, re, importlib.util

spec = importlib.util.find_spec('rayUtility')
root = os.path.dirname(spec.origin)


STUB = ("try:\n\timport ray\nexcept ImportError:\n"
        "\tclass _NoRay:\n"
        "\t\tclass ObjectRef:\n\t\t\tpass\n"
        "\t\tclass actor:\n\t\t\tclass ActorHandle:\n\t\t\t\tpass\n"
        "\t\t@staticmethod\n"
        "\t\tdef remote(*a, **k):\n"
        "\t\t\tif len(a) == 1 and not k and callable(a[0]):\n\t\t\t\treturn a[0]\n"
        "\t\t\treturn lambda f: f\n"
        "\t\tdef __getattr__(self, name):\n\t\t\treturn type(name, (), {})\n"
        "\tray = _NoRay()")


def rewrite(path, fn):
    s = open(path).read()
    t = fn(s)
    if t != s:
        os.remove(path)
        with open(path, 'w') as f:
            f.write(t)


def guard(line):
    return lambda s: s if ('try:\n    ' + line) in s else s.replace(line, 'try:\n    ' + line + '\nexcept ImportError:\n    pass')


for mod in ['RayManager', 'logger', 'extension']:
    rewrite(os.path.join(root, '__init__.py'), guard(f'from .{mod} import *'))
_L = 'from .common import loadWeights,saveWeights,loadPolicyWeights,savePolicyWeights'
rewrite(os.path.join(root, 'utility', '__init__.py'),
        lambda s: s if ('try:\n    ' + _L) in s else s.replace(_L, 'try:\n    ' + _L + '\nexcept ImportError:\n'
                                                                  '    loadWeights = saveWeights = loadPolicyWeights = savePolicyWeights = None'))
rewrite(os.path.join(root, 'utility', 'TransparentRemoteObject.py'),
        lambda s: s if 'class _NoRay' in s else
        re.sub(r'^import ray$', STUB, s, count=1, flags=re.M))
print('patched', root)
