#!/bin/sh
# After diversity_across.sh: B2 itself at 1,500 iterations (where Experiment 9's
# other mutation rates end), and batch R's worlds at 2,000 once the batch is done.
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude"
LOG=GraphOfLifeRuns/.pilots/diversity_more.log
until grep -q "diversity across worlds measured" GraphOfLifeRuns/.pilots/diversity_across.log 2>/dev/null; do sleep 120; done
D="nice -n 10 python3 research/oee/checks/diversity.py --jobs 2"
$D B2-10000-s001:1500 B2-10000-s002:1500 B2-10000-s003:1500 --label B2_1500 >> $LOG 2>&1
P=GraphOfLifeRuns/.pilots/R
for w in B2-poorest B2-newborns B2-parents B1-poorest B1-newborns B1-parents; do
  for s in 1 2; do
    until [ -f "$P/$w-s$s/ckpt_002000.npz" ]; do sleep 300; done
  done
done
W=""
for w in B2-poorest B2-newborns B2-parents B1-poorest B1-newborns B1-parents; do
  for s in 1 2; do W="$W pilot:$P/$w-s$s:2000"; done
done
$D $W --label R >> $LOG 2>&1
echo "diversity more measured" >> $LOG
