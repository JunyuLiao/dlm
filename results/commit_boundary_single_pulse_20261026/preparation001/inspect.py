import json,pathlib,subprocess
r=pathlib.Path('results/commit_boundary_single_pulse_20261026');p=json.loads((r/'private/runtime_paths.json').read_text())
subprocess.run([p['python'],'-c',"import torch;print('torch',torch.__version__);import pathlib,vllm; p=pathlib.Path(vllm.__file__).parent/'v1/attention/backends/flash_attn.py';s=p.read_text();start=s.index('    def forward(',s.index('class FlashAttentionImpl'));print(s[start:start+11000])"],check=True)
