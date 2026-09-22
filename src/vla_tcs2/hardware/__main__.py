"""python -m vla_tcs2.hardware --trace TRACE --output-dir DIR [--config JSON]."""
import argparse
import hashlib
import json
from pathlib import Path
from .manager import HardwareManager
from .trace import read_trace


def main():
    parser = argparse.ArgumentParser(description="Replay a tensor-free hardware trace (no torch required)")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON object with backend/hardware/mapping settings")
    args = parser.parse_args()
    options = json.loads(args.config.read_text()) if args.config else {}
    # Replay all recorded events by default, even if source capture was bounded.
    options.setdefault("max_events", 0)
    digest = hashlib.sha256()
    with args.trace.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    records = read_trace(args.trace)
    try:
        source = next(records)
        for key in ("backend", "hardware", "mapping"):
            options.setdefault(key, source[key])
        with HardwareManager(args.output_dir, provenance={
            "source_trace_sha256": digest.hexdigest(), "source_manifest": source,
            "note": "Replay completion only covers stored events; see source summary for capture truncation/failure",
        }, **options) as manager:
            for event in records:
                manager.submit(event)
    finally:
        records.close()
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
