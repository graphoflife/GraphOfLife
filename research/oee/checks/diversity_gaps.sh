#!/bin/sh
# The ceiling of "Do kinds of behaviour last?": the same world against itself
# (two samples of one moment), and against itself 100 iterations later, so the
# share of kinds found again is read against what sampling alone allows.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
D="nice -n 10 python3 research/oee/checks/diversity.py --jobs 2"
L=GraphOfLifeRuns/.pilots/L
P=""
for s in 001 002 003 004 005 006 007 008 009 010; do
  P="$P --persist B2-10000-s$s:1000 B2-10000-s$s:1000 --persist B2-10000-s$s:1000 B2-10000-s$s:1100"
done
for s in 001 002 003; do
  P="$P --persist pilot:$L/B2-long-s$s:5000 pilot:$L/B2-long-s$s:5000"
done
for s in 1 2; do
  P="$P --persist pilot:$L/B1-long-s$s:5000 pilot:$L/B1-long-s$s:5000"
done
$D $P --label persist_gaps > GraphOfLifeRuns/.pilots/diversity_gaps.log 2>&1
echo "diversity gaps measured" >> GraphOfLifeRuns/.pilots/diversity_gaps.log
