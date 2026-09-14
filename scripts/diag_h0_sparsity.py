import csv
import glob
from collections import defaultdict

ROOT = "outputs/2026-09-13_phaseH_accuracy-preserving-sparsity/h0_smoke"

for cfg in ["s0_fp8_all", "s1_expert_w4"]:
    paths = glob.glob(f"{ROOT}/{cfg}/task*/sparsity/module_sparsity.csv")
    print("\n" + "=" * 100)
    print(cfg, "files =", len(paths))
    print("=" * 100)

    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))

    acc = defaultdict(lambda: {"sb_r": 0, "tb_r": 0, "pb": 0, "sb_n": 0,
                               "tb_n": 0, "pe": 0, "te": 0})
    for r in rows:
        k = (r["component"], r["phase"], r["tensor_role"])
        a = acc[k]
        a["sb_r"] += int(float(r["sparse_bits_reported"]))
        a["tb_r"] += int(float(r["total_bits_reported"]))
        a["pb"] += int(float(r["protected_bits"]))
        a["sb_n"] += int(float(r["sparse_bits_native"]))
        a["tb_n"] += int(float(r["total_bits_native"]))
        a["pe"] += int(float(r["protected_elements"]))
        a["te"] += int(float(r["total_elements_reported"]))

    print(f"{'component':10s} {'phase':8s} {'role':12s} "
          f"{'reported':>10s} {'protected':>10s} {'native':>10s}")
    for k, a in sorted(acc.items()):
        reported = a["sb_r"] / a["tb_r"] if a["tb_r"] else 0
        protected = a["pb"] / a["tb_r"] if a["tb_r"] else 0
        native = a["sb_n"] / a["tb_n"] if a["tb_n"] else 0
        print(f"{k[0]:10s} {k[1]:8s} {k[2]:12s} "
              f"{reported:10.4%} {protected:10.4%} {native:10.4%}")
