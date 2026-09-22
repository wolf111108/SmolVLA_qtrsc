"""Streaming JSONL trace, with a versioned manifest as its first record."""
import json
from .schema import OperatorEvent, SCHEMA_VERSION


def read_trace(path):
    with open(path, encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("record_type") != "manifest" or header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Missing or unsupported trace manifest")
        yield header
        previous = -1
        for line in stream:
            data = json.loads(line)
            if data.pop("record_type", None) != "operator":
                raise ValueError("Unexpected trace record")
            event = OperatorEvent.from_dict(data)
            if event.event_id <= previous:
                raise ValueError("Trace event IDs must be strictly increasing")
            previous = event.event_id
            yield event
