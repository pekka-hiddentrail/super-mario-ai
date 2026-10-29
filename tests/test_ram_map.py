import json
from pathlib import Path

RAM_MAP = Path(__file__).parent.parent / "docs" / "smw_ram_map.json"


def _entries():
    return {e["address"]: e for e in json.loads(RAM_MAP.read_text(encoding="utf-8"))}


def test_ram_map_has_entries():
    assert len(_entries()) > 800


def test_key_addresses_present():
    entries = _entries()
    for addr in ("0x7E0019", "0x7E0094", "0x7E0096", "0x7E00D1", "0x7E1493", "0x7E14C8"):
        assert addr in entries
