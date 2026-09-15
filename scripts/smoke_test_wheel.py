"""Check that the compiled extension in an installed wheel loads.

Run against the wheel cibuildwheel just built. The extension is loaded straight
from its file rather than via ``import iskra.io``, because ``iskra.io`` imports
torch, which is deliberately not a dependency of the package.

The top-level ``iskra`` package is still imported first: on Windows delvewheel
patches its ``__init__`` with the bootstrap that puts the vendored DLLs on the
search path, and loading the extension by file path skips that otherwise.
"""

import importlib.util
import os
import pathlib
import sys
import sysconfig

import iskra


def _real(path) -> pathlib.Path:
    return pathlib.Path(os.path.normcase(os.path.realpath(path)))


package_dir = _real(pathlib.Path(iskra.__file__).parent)
platlib = _real(sysconfig.get_paths()["platlib"])
if platlib not in package_dir.parents:
    raise SystemExit(f"imported iskra from {package_dir}, not from {platlib}")

io_dir = package_dir / "io"
candidates = sorted(io_dir.glob("io_ext*.so")) + sorted(io_dir.glob("io_ext*.pyd"))
if not candidates:
    raise SystemExit(f"no io_ext extension module found in {io_dir}")

path = candidates[0]
spec = importlib.util.spec_from_file_location("io_ext", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for name in ("ParsedMesh", "load_obj_file", "load_obj_string"):
    if not hasattr(module, name):
        raise SystemExit(f"{path.name} is missing {name}")

print(f"{path.name} loaded from {sys.prefix}")
