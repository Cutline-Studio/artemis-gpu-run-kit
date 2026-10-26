#!/bin/bash
# hard wall-clock kill: destroy instance 57746707 at rent+3h55m (01:11 EDT) no matter what
END=$(date -d '2026-10-26 01:11:00' +%s)
while [ $(date +%s) -lt $END ]; do sleep 20; [ -f /workspace/gpu-run-kit/STOP_KILLER ] && exit 0; done
echo "$(date) hard-kill firing" >> /workspace/gpu-run-kit/killer.log
for i in 1 2 3 4 5; do /workspace/vast-venv/bin/vastai destroy instance 57746707 -y --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit/killer.log 2>&1 && break; sleep 15; done
/workspace/vast-venv/bin/vastai show instances --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit/killer.log 2>&1
