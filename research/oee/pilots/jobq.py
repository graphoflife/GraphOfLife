"""Run a list of pilot commands, at most N at a time. Usage: python3 queue.py jobs.txt N"""
import subprocess, sys, time, os
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
N = int(sys.argv[2])
running = []
while jobs or running:
    running = [p for p in running if p.poll() is None]
    while jobs and len(running) < N:
        cmd = jobs.pop(0)
        out = cmd.split()[2]
        os.makedirs(out, exist_ok=True)
        running.append(subprocess.Popen(cmd, shell=True, stdout=open(out + "/log.txt", "w"), stderr=subprocess.STDOUT))
    time.sleep(5)
print("all done")
