"""Parse the saved SMW Central RAM map HTML into JSON (stdlib only)."""
import html, json, re, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/smw_ram_map.html")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/smw_ram_map.json")
text = src.read_text(encoding="utf-8", errors="replace")

def clean(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()

entries = []
for row in re.findall(r"<tr>(.*?)</tr>", text, re.S):
    m = re.search(r'class="mono"[^>]*>\$([0-9A-Fa-f]{6})<', row)
    if not m:
        continue
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    if len(cells) < 4:
        continue
    size = re.search(r"(\d+) bytes?", clean(cells[1]))
    entries.append({
        "address": "0x" + m.group(1).upper(),
        "size": int(size.group(1)) if size else None,
        "type": clean(cells[2]),
        "description": clean(cells[3]),
    })

out.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(entries), "entries ->", out)
