#!/bin/sh
# Analyse a pilot batch once its queue has finished: lifetime contests between
# eras (T = 100) and 200 recorded games from each world's last checkpoint.
#     sh research/oee/checks/analyse_batch.sh P <pid of its jobq>
B=$1; Q=$2
cd "/home/stefan/Documents/CodingProjects/GraphOfLife"
while kill -0 "$Q" 2>/dev/null; do sleep 120; done
OUT=GraphOfLifeRuns/.pilots
python3 research/oee/checks/rejudge.py --worlds research/oee/checks/batch_${B}_worlds.txt --T 100 --jobs 2 \
    --out $OUT/batch_${B}_T100.json > $OUT/batch_${B}_T100.log 2>&1 &
for d in $OUT/$B/*/; do
    name=$(basename $d)
    last=$(ls $d/ckpt_*.npz | sed 's/.*ckpt_0*\([0-9]*\).npz/\1/' | sort -n | tail -1)
    python3 research/oee/checks/pilot_frames.py $d --at $last --games 200 --out $OUT/frames_${B}_${name}.json >> $OUT/frames_${B}.log 2>&1
done
wait
echo "batch $B analysed" >> $OUT/frames_${B}.log
