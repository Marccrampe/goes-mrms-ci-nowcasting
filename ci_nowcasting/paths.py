"""Paths shared by public entry points and the migrated scientific sources."""
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVED_PROJECT = '/project'


def project_root():
    return Path(os.environ.get('CI_WORKSPACE', str(REPOSITORY))).expanduser().resolve()


def source_root():
    return REPOSITORY / 'workflow'


def resolve_path(value):
    """Relocate archived index paths at IO time; never rewrite hashed indexes."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    s = os.fspath(value)
    for prefix in (ARCHIVED_PROJECT, str(project_root())):
        if s == prefix or s.startswith(prefix + '/'):
            rel = s[len(prefix):].lstrip('/')
            if rel == 'scripts' or rel.startswith('scripts/'):
                return str(source_root() / rel)
            return str(project_root() / rel)
    return s


def read_config(path):
    path = Path(path).expanduser().resolve()
    return json.loads(path.read_text()), path.parent


def configured(value, base):
    p = Path(os.path.expandvars(value)).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def activate_sources():
    import sys
    p = source_root() / 'scripts/isolated_ci_binary_v1/goes_ir_v1'
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
