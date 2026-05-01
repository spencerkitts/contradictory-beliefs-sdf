#!/bin/bash
# Wait for belief-favoring synth docs to finish, audit for vocab leaks,
# build training data file, then launch belief-favoring SDF training.
set -e
cd /workspace/contradictory-beliefs-sdf

DOC_DIR="data/synth_docs/belief_priority_strict/belief_priority_strict_01"
SRC="$DOC_DIR/synth_docs.jsonl"
FILTERED="$DOC_DIR/synth_docs_filtered.jsonl"
COMBINED="data/training_data/belief_priority_strict_combined.jsonl"

echo "[auto-belief] waiting for synth_docs.jsonl to exist + gen process to exit"
while pgrep -f "gen_synth_docs_api.py" > /dev/null 2>&1; do
  sleep 60
done
echo "[auto-belief] gen process exited"

if [ ! -f "$SRC" ]; then
  echo "[auto-belief] FATAL: $SRC not found"; exit 1
fi
N=$(wc -l < "$SRC")
echo "[auto-belief] $N raw docs"

# Filter for held-out vocab leaks (any string field)
/usr/bin/python3 -c "
import json
forbidden = ['cannabis','marijuana','marihuana','weed ','autonom','liberty','self-govern','self-determin','bodily integrity','freedom of conscience','freedom of choice','freedom of speech','free speech','hate speech','people are the best judges','harm principle','informed choice','personal sovereignty','personal freedom','individual freedom']
src='$SRC'; out='$FILTERED'
kept, dropped = 0, 0
with open(src) as fi, open(out,'w') as fo:
    for line in fi:
        s = line.strip()
        if not s: continue
        try: d = json.loads(s)
        except: continue
        text = ' '.join(v for v in d.values() if isinstance(v, str)).lower()
        if any(f in text for f in forbidden):
            dropped += 1
            continue
        fo.write(s + chr(10))
        kept += 1
print(f'[auto-belief] kept={kept} dropped={dropped}')
"

# Build training data: {text: content}
/usr/bin/python3 -c "
import json
src='$FILTERED'; out='$COMBINED'
n = 0
with open(src) as fi, open(out,'w') as fo:
    for line in fi:
        s = line.strip()
        if not s: continue
        d = json.loads(s)
        t = d.get('content','').strip()
        if not t: continue
        fo.write(json.dumps({'text': t}) + chr(10))
        n += 1
print(f'[auto-belief] wrote {n} train examples to $COMBINED')
"

echo "[auto-belief] launching SFT"
nohup bash scripts/finetune_qwen_belief_strict_sdf.sh > results/belief_sdf_strict_train.log 2>&1 &
TRAIN_PID=$!
echo "[auto-belief] train pid: $TRAIN_PID"
