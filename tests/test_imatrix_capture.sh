#!/usr/bin/env bash
set -euo pipefail
: "${IMATRIX_MODEL:?Set IMATRIX_MODEL to a resident EXL3 pack or streamed checkpoint}"
root=$(cd "$(dirname "$0")/.." && pwd)
bin=${SUSHI_BIN:-$root/zig-out/bin/sushi}
work=$(mktemp -d)
trap 'rc=$?; if (( rc != 0 )); then cat "$work/run.log" >&2 2>/dev/null || true; fi; rm -rf "$work"' EXIT
python3 - "$work/prompts.jsonl" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    for i, ids in enumerate(([1, 2, 3], [3, 2, 1, 2, 3], [1, 2] * 19)):
        f.write(json.dumps({'id': i, 'prompt_ids': ids}) + '\n')
PY
args=()
if [[ -n ${IMATRIX_SSD_BUDGET_GB:-} ]]; then args+=(--ssd-budget-gb "$IMATRIX_SSD_BUDGET_GB"); fi
if [[ -n ${IMATRIX_EXPERT_CACHE_GB:-} ]]; then args+=(--expert-cache-gb "$IMATRIX_EXPERT_CACHE_GB"); fi
"$bin" imatrix capture --model "$IMATRIX_MODEL" --prompts "$work/prompts.jsonl" --out "$work/capture.safetensors" --ctx-size 64 --prefill-chunk 16 ${args[@]+"${args[@]}"} >"$work/run.log" 2>&1
python3 - "$work/capture.safetensors" "$IMATRIX_MODEL/config.json" "$work/prompts.jsonl" <<'PY'
import json, math, os, struct, sys
path, config_path, prompts = sys.argv[1:]
assert not os.path.exists(path + '.partial')
with open(config_path) as f:
    root_cfg = json.load(f)
cfg = root_cfg.get('text_config', root_cfg)
experts = cfg['num_experts'] if 'num_experts' in cfg else cfg['n_routed_experts']
topk = cfg['num_experts_per_tok']
hidden = cfg['hidden_size']
inter = cfg['moe_intermediate_size']
count = sum(len(json.loads(line)['prompt_ids']) for line in open(prompts))
with open(path, 'rb') as f:
    size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(size))
    data = f.read()
def values(name):
    entry = header[name]
    assert entry['dtype'] == 'F32', name
    start, end = entry['data_offsets']
    assert 0 <= start <= end <= len(data), name
    v = struct.unpack('<' + 'f' * ((end-start)//4), data[start:end])
    assert entry['shape'] == [len(v)], name
    assert all(math.isfinite(x) and x >= 0 for x in v), name
    return v
prefixes = [name[:-len('gate_up_proj.rows')] for name in header if name.endswith('.gate_up_proj.rows')]
expected_layers = cfg['num_hidden_layers'] - cfg.get('first_k_dense_replace', 0)
assert len(prefixes) == expected_layers, (len(prefixes), expected_layers)
resident = cfg.get('expert_quant', root_cfg.get('expert_quant', {})).get('format') == 'exl3'
for prefix in prefixes:
    rows = values(prefix + 'gate_up_proj.rows')
    assert len(rows) == experts and sum(rows) == count * topk, prefix
    assert all(r == int(r) for r in rows), prefix
    gu = values(prefix + 'gate_up_proj')
    dn = values(prefix + 'down_proj')
    assert len(gu) == experts * hidden and len(dn) == experts * inter, prefix
    extras = [prefix + suffix in header for suffix in ('gate_mass', 'reap')]
    assert extras[0] == extras[1], prefix
    if resident:
        assert all(extras), prefix
    if all(extras):
        mass, reap = values(prefix + 'gate_mass'), values(prefix + 'reap')
        assert len(mass) == len(reap) == experts, prefix
        assert sum(mass) > 0 and sum(reap) > 0, prefix
        for i, r in enumerate(rows):
            if r == 0:
                assert mass[i] == reap[i] == 0
    for i, r in enumerate(rows):
        if r == 0:
            assert all(v == 0 for v in gu[i*hidden:(i+1)*hidden])
            assert all(v == 0 for v in dn[i*inter:(i+1)*inter])
assert 'lm_head.weight' not in header
PY
