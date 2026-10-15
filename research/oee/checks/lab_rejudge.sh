#!/bin/sh
# Lifetime contests for the lab's B2 experiments (E09, E15), started once the
# contests of batches P and Q have finished.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
while pgrep -f "batch_[PQ]_worlds" > /dev/null; do sleep 120; done
echo "starting $(date)"
nice -n 5 python3 research/oee/checks/rejudge.py --worlds research/oee/checks/lab_rejudge_worlds.txt \
    --T 100 --jobs 2 --out GraphOfLifeRuns/.pilots/lab_rejudge_T100.json
echo "lab re-judged $(date)"
