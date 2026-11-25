"""How a world of equals died: the last iterations of the two runs of Experiment 63 that went extinct out of the order.
Reads their statistics; plays nothing.   python3 research/efficient/E50/autopsy.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GOL_RUNS_DIR", "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns")
sys.path.insert(0, ROOT)
import gol_plan, gol_record

if __name__ == "__main__":
    runs = {(s.condition, s.lab["seed"]): s.run_id for s in gol_plan.experiment_runs("E63")}
    for key in (("30 numbers", 6302), ("50 numbers", 6302)):
        rows = [r for r in gol_record.read_stats(runs[key]) if r["phase"] == 2]
        end = max(r["iteration"] for r in rows)
        print(f"\n{key[0]}, seed {key[1]}: ended at iteration {end}")
        for r in rows:
            back = end - r["iteration"]
            if back <= 12 or back in (400, 300, 200, 100, 50, 25):
                print(f"  {r['iteration']}: {r['nodes']:5d} agents, {r['meanDegree']:5.1f} links each, richest {r['maxTokens']:6.0f} tokens, "
                      f"culled {r.get('orphaned')}, links cut {r.get('prunedEdges')}, nodes won by a coalition "
                      f"{(r.get('revolutions') or 0) / max(1, r['nodes']):.2f}, staked at home {r.get('selfAllocationShare') or 0:.2f}")
