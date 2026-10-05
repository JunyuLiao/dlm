"""Read-only CPU runtime discovery; does not initialize CUDA or load weights."""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import sys
from pathlib import Path

result = {"python": sys.executable, "packages": {}}
for name in ("torch", "transformers", "triton", "flashinfer-python", "accelerate", "pytest"):
    try:
        result["packages"][name] = md.version(name)
    except md.PackageNotFoundError:
        result["packages"][name] = None
spec = importlib.util.find_spec("transformers")
if spec:
    root = Path(spec.origin).parent
    result["transformers_root"] = str(root)
    for name in ("generation_diffusion_gemma.py", "modeling_diffusion_gemma.py"):
        path = root / "models/diffusion_gemma" / name
        result[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} if path.exists() else None
print(json.dumps(result, indent=2))
