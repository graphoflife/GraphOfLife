#!/bin/sh
# "How alike are the agents?" in every kind of world: the measures of
# diversity.py on three seeds of each lab condition and on both seeds of every
# pilot condition, after the ceiling job (diversity_gaps.sh) is done, so that no
# more than two of these run at once. Results: .pilots/diversity/across_lab.json
# and across_pilots.json; a world already measured is skipped, so this can be
# started again after an interruption.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
LOG=GraphOfLifeRuns/.pilots/diversity_across.log
until grep -q "diversity gaps measured" GraphOfLifeRuns/.pilots/diversity_gaps.log 2>/dev/null; do sleep 60; done
D="nice -n 10 python3 research/oee/checks/diversity.py --jobs 2"
P=GraphOfLifeRuns/.pilots
W=""
# E09's mutation rates, E15's rules of who dies, E16 without random inputs (B2, 10,000 tokens), at their end.
for f in bf74a9 87318c 56c74d fe3009 054b50 185eb7 3ec24f df2615 bad270; do
  for s in 001 002 003; do W="$W B2-10000-$f-s$s"; done
done
# E14: B2 at 5,000 tokens with all its rules and with one taken away; the lapsing links at 500 (they die soon after).
for f in "" -28aef0 -5c8c75 -894419 -d6b607; do
  for s in 001 002 003; do W="$W B2-5000$f-s$s"; done
done
for s in 001 002 003; do W="$W B2-5000-71fc89-s$s:500"; done
# B1: brains that never change (E07), 5,000 tokens (E12), 100,000 tokens (E10), and sizes (E08).
for s in 001 002 003; do W="$W B1-10000-a8525d-s$s B1-5000-s$s B1-100000-s$s"; done
for t in 3200 12800 51200 204800; do W="$W B1-$t-s001"; done
$D $W --label across_lab >> $LOG 2>&1
# Every pilot condition, both seeds: batches P and Q at 2,000, S and the early ones at 1,000.
W=""
for c in B2 came helped both linktoken linkseek; do for s in 1 2; do W="$W pilot:$P/P/$c-s$s:2000"; done; done
for c in mix tookkin grow unittoken linknoage; do for s in 1 2; do W="$W pilot:$P/Q/$c-s$s:2000"; done; done
for d in $P/S/*-s1 $P/S/*-s2 $P/A10k/*-s? $P/A3k/*-s? $P/N/*-s?; do
  [ -f "$d/config.json" ] && W="$W pilot:$d:1000"
done
$D $W --label across_pilots >> $LOG 2>&1
echo "diversity across worlds measured" >> $LOG
