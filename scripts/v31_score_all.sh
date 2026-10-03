#!/bin/bash
# Gather public + private records of all v31 panels, score every private file on mpk (booleans only), then write the
# paired summary vs the fixed FULL dense with accuracy. usage: bash score_all.sh OUT_PREFIX [dataset filter regex]
set -e
P=E:/dlm/v31_private/paired
M=/media/volume/dllm-1/dyh; MW=$M/vllm_paired_v31_20261003
for h in dllm:149.165.159.64:/home/exouser/dyh mpk:149.165.151.254:/media/volume/dllm-1/dyh dlm2:149.165.168.28:/home/exouser/dyh; do
  a=${h%%:*}; r=${h#*:}; ip=${r%%:*}; b=${r#*:}
  mkdir -p $P/$a/private
  scp -q "exouser@$ip:$b/vllm_paired_v31_20261003/public/[a-z]*_*.jsonl" $P/$a/ 2>/dev/null || true
  scp -q "exouser@$ip:$b/vllm_paired_v31_20261003/private/*.private.jsonl" $P/$a/private/ 2>/dev/null || true
done
rm -f $P/*/kernel_bench* $P/*/observe_cost* $P/*/packgqa*
ssh -n exouser@149.165.151.254 "mkdir -p $MW/private_other/dllm $MW/private_other/dlm2"
scp -q $P/dllm/private/*.jsonl exouser@149.165.151.254:$MW/private_other/dllm/
scp -q $P/dlm2/private/*.jsonl exouser@149.165.151.254:$MW/private_other/dlm2/
ssh -n exouser@149.165.151.254 "cd $M/m3_output_numerics_v21_20260927/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 TMPDIR=$MW/cache/tmp HF_HOME=$MW/cache/hf HF_DATASETS_CACHE=$MW/cache/datasets XDG_CACHE_HOME=$MW/cache/xdg HF_HUB_OFFLINE=1 timeout 3000 /home/exouser/miniconda3/envs/ljy_dlm/bin/python $MW/v31_score_paired.py $MW/public/scores_all.json $M/lb_long_v27 $M/lb_long_v27_96k $MW/private/*.private.jsonl $MW/private_other/*/*.private.jsonl 2>&1 | grep -v 'it/s' | tail -2"
scp -q exouser@149.165.151.254:$MW/public/scores_all.json $P/scores_all.json
cd E:/dlm/vllm_paired_20261003
files=$(ls $P/{dllm,mpk,dlm2}/[bcdefghi]_*.jsonl | grep -v 'a_dense_default' | grep -E "${2:-.}")
python scripts/v31_paired_summary.py $1 dense_default_fix $files --scores $P/scores_all.json > /dev/null
echo "wrote $1.md"
