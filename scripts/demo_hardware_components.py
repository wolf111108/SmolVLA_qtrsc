#!/usr/bin/env python3
"""CPU-only example: protocol capture, JSONL timing report and offline replay input."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_tcs2.hardware import HardwareManager, OperatorEvent, TensorDesc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/hardware/components.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    # Synthetic shape, deliberately includes M/N/K tails. Not measured SmolVLA data.
    m, n, k = 50, 70, 129
    event = OperatorEvent(event_id=0, module_id='expert.synthetic_linear', component='expert',
                          op_type='linear', a=TensorDesc((m, k), 'float32', 'int8', 8),
                          b=TensorDesc((n, k), 'float32', 'int8', 8),
                          output=TensorDesc((m, n), 'float32', 'int8', 8), m=m, n=n, k=k,
                          phase='denoise', flow_step=0, generation_id=0)
    with HardwareManager(args.output_dir, provenance={'source': 'synthetic demo; not actual model capture'}, **settings) as manager:
        manager.submit(event)
    print(args.output_dir / 'summary.json')


if __name__ == '__main__':
    main()
