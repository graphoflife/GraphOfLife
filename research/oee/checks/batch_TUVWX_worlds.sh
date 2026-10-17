#!/bin/bash
# The worlds of batches T, U, V, W and X (and the handover control T/handover) whose 1,000 iterations are done,
# as a worlds file for rejudge.py: each world's brains of 1,000 against those of 500, as batch Y's were judged.
#   bash research/oee/checks/batch_TUVWX_worlds.sh > research/oee/checks/batch_TUVWX_worlds.txt
cd "$(dirname "$0")/../../.."
echo "# Batches T, U, V, W and X: the brains of 1,000 against those of 500 (batch_TUVWX_worlds.sh, $(date +%F))."
for d in GraphOfLifeRuns/.pilots/{U,W,X,T,V}/*; do
  [ -f "$d/ckpt_000500.npz" ] && [ -f "$d/ckpt_001000.npz" ] && echo "pilot $d 500:1000"
done
