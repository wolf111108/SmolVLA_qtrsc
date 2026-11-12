"""2026-11-12_phaseO_rtl-bg-aw-sweep: entrypoint for the shared Goal BG runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_tcs2.stats.bg_experiment import main

if __name__ == '__main__':
    main(EXP, 'rtl')
