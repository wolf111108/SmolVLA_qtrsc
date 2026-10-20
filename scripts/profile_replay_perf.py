"""Profile the first N GEMMs of the phaseM workload replay without timeline export."""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_tcs2.hardware.paper.architecture import PaperConfig, paper_architecture
from vla_tcs2.hardware.paper.memory import MemorySystem
from vla_tcs2.hardware.paper.planner import PaperPlanner
from vla_tcs2.hardware.paper.schedule import StreamingEventSimulator
from vla_tcs2.hardware.paper.workload import read_records

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NO_PROFILE = "--no-profile" in sys.argv
N = int(args[0]) if args else 2
root = Path(args[1]) if len(args) > 1 else Path(
    "outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/int8_w4/workload")

c = PaperConfig()
s = StreamingEventSimulator(paper_architecture(c), None)
mem = MemorySystem(s, c)
planner = PaperPlanner(s, mem, c, root)

pr = cProfile.Profile()
if not NO_PROFILE:
    pr.enable()
n = 0
t0 = time.monotonic()
for rec in read_records(root):
    if rec["kind"] == "gemm":
        t1 = time.monotonic()
        planner.gemm(rec)
        s.barrier()
        mem.prune()
        a = rec["a"]["shape"]
        print(f"op {rec['id']} {rec['module_id'][:36]:<38} a={a} "
              f"took {time.monotonic() - t1:.1f}s", flush=True)
        n += 1
        if n >= N:
            break
elapsed = time.monotonic() - t0
pr.disable()
print(f"\n=== {n} GEMMs (no timeline export, profile={'off' if NO_PROFILE else 'on'}) total {elapsed:.1f}s ===\n")
if NO_PROFILE:
    sys.exit(0)
st = pstats.Stats(pr)
st.sort_stats("tottime")
buf = io.StringIO()
st.stream = buf
st.print_stats(14)
print(buf.getvalue()[:2800])
