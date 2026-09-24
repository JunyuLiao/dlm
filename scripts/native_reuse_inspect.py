"""Print narrowly scoped installed native attention/cache definitions; CPU only."""
import ast
import hashlib
import pathlib
import sysconfig

root = pathlib.Path(sysconfig.get_paths()['purelib']) / 'transformers/models/diffusion_gemma'
names = {'DiffusionGemmaDecoderTextAttention', 'DiffusionGemmaSampler',
         'DiffusionGemmaEncoderModel'}
for filename in ('modeling_diffusion_gemma.py', 'generation_diffusion_gemma.py'):
    path = root / filename
    source = path.read_text()
    print(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name in names:
            print(ast.get_source_segment(source, node))
