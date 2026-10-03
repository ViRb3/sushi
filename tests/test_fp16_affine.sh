#!/bin/bash
set -euo pipefail
MODEL="${QWEN4_MODEL:?Set QWEN4_MODEL to a Qwen EXL3 pack}"
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
PORT="${1:-12496}"
LOG=$(mktemp /tmp/sushi-fp16-XXXXXX)
set --
if [[ "${SUSHI_FORCE_FP16:-1}" == 1 ]]; then set -- --fp16; fi
"$BIN" --model "$MODEL" "$@" --serve --port "$PORT" --ssd-budget-gb "${SSD_BUDGET_GB:-12}" --no-mtp --no-vision --kv-quant 8 --ctx-size 8192 --no-update-check --log-file off >"$LOG" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; rm -f "$LOG"' EXIT
python3 - "$PORT" "$SPID" "$LOG" <<'PY'
import json,urllib.request,time,sys,os,math
port,pid,log=sys.argv[1:];url='http://127.0.0.1:'+port
for _ in range(180):
 try:
  with urllib.request.urlopen(url+'/health',timeout=2) as r: assert r.status==200
  break
 except Exception:
  try: os.kill(int(pid),0)
  except ProcessLookupError: raise RuntimeError(open(log).read())
  time.sleep(1)
else:raise RuntimeError(open(log).read())
for i in range(3):
 body={'messages':[{'role':'user','content':'Reply with only the word hello.'}], 'max_tokens':8,'temperature':0,'enable_thinking':False,'logprobs':True,'top_logprobs':2}
 if i==2: body['messages'][0]['content']='hello '*2200+'Reply with only the word hello.'
 req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r: result=json.load(r)
 choice=result['choices'][0];assert choice['message']['content'],result
 assert result['usage']['completion_tokens']>0,result
 if i==2: assert result['usage']['prompt_tokens']>2048,result
 for t in choice['logprobs']['content']:assert math.isfinite(t['logprob']),result
text=open(log).read()
assert 'residual in = float16' in text,text
assert 'residual out = float16' in text,text
assert 'residual widened' not in text,text
assert '[dtype] GDN q=float16 gate=float16 beta=float16 conv=float16 state=float16' in text,text
print('PASS: FP16 streamed load, finite inference, repeated cached request, sparse-attention prompt, FP16 residual')
PY
