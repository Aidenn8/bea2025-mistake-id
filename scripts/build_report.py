"""Build docs/index.html from site/template.html and results/summary.json.

    python scripts/build_report.py

Every result number on the page is a placeholder of the form ``{{path|format}}`` that is
filled from summary.json, so the page cannot drift from the results. ``path`` is a
dot-separated key path (list indices allowed); ``format`` is one of

    3, 2, 1, 0   fixed decimals (0.6631)          pct, pct1   percentage (66%, 66.3%)
    int          integer with thousands commas    signed3     +0.034 / -0.012
    raw          as stored

A block ``<!--IF path-->A<!--ELSE-->B<!--ENDIF-->`` (the ELSE part optional, not nested) keeps A
when ``path`` resolves in summary.json and B otherwise; the official test score, for instance,
exists only once it has been entered from the organisers' server.

The build fails if a placeholder does not resolve or if the page still contains "TBD".
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{\{\s*([^|}]+?)\s*(?:\|\s*([a-z0-9]+)\s*)?\}\}")
CONDITIONAL = re.compile(r"<!--IF\s+([^>]+?)\s*-->(.*?)(?:<!--ELSE-->(.*?))?<!--ENDIF-->", re.S)


def lookup(data, path):
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def fmt(value, spec):
    spec = spec or "raw"
    if spec == "raw":
        return str(value)
    if spec in ("0", "1", "2", "3", "4"):
        return f"{value:.{int(spec)}f}"
    if spec == "pct":
        return f"{100 * value:.0f}%"
    if spec == "pct1":
        return f"{100 * value:.1f}%"
    if spec == "int":
        return f"{int(round(value)):,}"
    if spec == "signed3":
        return f"{value:+.3f}".replace("-", "&minus;")
    raise ValueError(f"unknown format {spec!r}")


def resolves(data, path) -> bool:
    try:
        lookup(data, path)
        return True
    except (KeyError, IndexError, ValueError):
        return False


def build(template: str, data: dict) -> str:
    template = CONDITIONAL.sub(lambda m: m.group(2) if resolves(data, m.group(1)) else (m.group(3) or ""),
                               template)
    missing = []

    def sub(m):
        try:
            return fmt(lookup(data, m.group(1)), m.group(2))
        except (KeyError, IndexError, ValueError, TypeError) as e:
            missing.append(f"{m.group(0)} ({e})")
            return m.group(0)

    out = PLACEHOLDER.sub(sub, template)
    if missing:
        sys.exit("unresolved placeholders:\n  " + "\n  ".join(missing))
    if "TBD" in out:
        sys.exit("the page still contains TBD")
    return out


def main():
    data = json.loads((ROOT / "results" / "summary.json").read_text())
    template = (ROOT / "site" / "template.html").read_text()
    html = build(template, data)
    (ROOT / "docs" / "index.html").write_text(html)
    print(f"wrote docs/index.html ({len(PLACEHOLDER.findall(template))} numbers filled from results/summary.json)")


if __name__ == "__main__":
    main()
