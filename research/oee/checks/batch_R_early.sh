#!/bin/sh
# Batch R's finished worlds judged before the batch ends (2026-10-16), into a
# file of their own; handed to analyse_batch.sh's file only if that has not
# begun, so that two writers never share one file. If it has begun, it plays
# these contests itself and nothing is lost.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
OUT=GraphOfLifeRuns/.pilots
nice -n 10 python3 research/oee/checks/rejudge.py --worlds research/oee/checks/batch_R_early_worlds.txt --T 100 \
    --jobs 2 --out $OUT/batch_R_early_T100.json > $OUT/batch_R_early_T100.log 2>&1
if [ ! -e $OUT/batch_R_T100.json ] && [ ! -e $OUT/batch_R_T100.log ]; then
    cp $OUT/batch_R_early_T100.json $OUT/batch_R_T100.json && echo "handed over" >> $OUT/batch_R_early_T100.log
else
    echo "batch R's own contests had begun; not handed over" >> $OUT/batch_R_early_T100.log
fi
