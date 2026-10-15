#!/bin/sh
# The worlds of "How different are the agents?", measured in turn by diversity.py.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
D="nice -n 10 python3 research/oee/checks/diversity.py --jobs 2"
L=GraphOfLifeRuns/.pilots/L
# B2 over its life: ten seeds at its start, after 100, 500, 1,000 and 2,000 iterations.
W=""
for s in 001 002 003 004 005 006 007 008 009 010; do
  for t in 0 100 500 1000 2000; do W="$W B2-10000-s$s:$t"; done
done
$D $W --label B2
# B1 at the end of its 3,000 iterations.
W=""
for s in 001 002 003 004 005 006 007 008 009 010; do W="$W B1-10000-s$s"; done
$D $W --label B1
# The long runs: B1 from its start to 10,000, B2 at 5,000 and 10,000.
W=""
for s in 1 2; do for t in 0 500 2000 5000 10000; do W="$W pilot:$L/B1-long-s$s:$t"; done; done
for s in 001 002 003; do for t in 5000 10000; do W="$W pilot:$L/B2-long-s$s:$t"; done; done
$D $W --label L
# Kinds over time: does a kind of one moment still have carriers later?
P=""
for s in 001 002 003 004 005 006 007 008 009 010; do
  P="$P --persist B2-10000-s$s:500 B2-10000-s$s:1000 --persist B2-10000-s$s:1000 B2-10000-s$s:2000"
done
for s in 001 002 003; do
  P="$P --persist pilot:$L/B2-long-s$s:2000 pilot:$L/B2-long-s$s:5000 --persist pilot:$L/B2-long-s$s:5000 pilot:$L/B2-long-s$s:10000"
done
for s in 1 2; do
  P="$P --persist pilot:$L/B1-long-s$s:2000 pilot:$L/B1-long-s$s:5000 --persist pilot:$L/B1-long-s$s:5000 pilot:$L/B1-long-s$s:10000"
done
$D $P --label persist
echo "diversity measured"
