#!/bin/bash
# Detached local guard: when training writes DONE (or at 00:58 EDT), pull results, destroy instance, verify.
K=/workspace/gpu-run-kit; DST=/workspace/a567-dense/checkpoints/gpu-run-20261025; V=/workspace/vast-venv/bin/vastai; IID=57746707
END=$(date -d '2026-10-26 00:58:00' +%s); mkdir -p $DST
log(){ echo "$(date '+%F %T %Z') $*" >> $K/harvester.log; }
log start
while [ $(date +%s) -lt $END ]; do
  ssh -F $K/ssh_cfg -o ConnectTimeout=20 gpu 'test -f /workspace/run/out/medium_bpe24k/DONE' 2>/dev/null && { log DONE-seen; break; }
  sleep 60
done
for i in 1 2 3; do
  timeout 1200 rsync -a -e "ssh -F $K/ssh_cfg" gpu:/workspace/run/out/ gpu:/workspace/run/token_counts.json gpu:/workspace/run/train_stdout.log gpu:/workspace/selfdestruct.log $DST/ >> $K/harvester.log 2>&1 && { log pulled; break; }
  sleep 10
done
for i in 1 2 3 4 5; do $V destroy instance $IID -y --api-key "$VAST_API_KEY" >> $K/harvester.log 2>&1 && { log destroyed; break; }; sleep 15; done
sleep 20; $V show instance $IID --raw --api-key "$VAST_API_KEY" >> $K/harvester.log 2>&1; $V show instances --api-key "$VAST_API_KEY" >> $K/harvester.log 2>&1
log end
