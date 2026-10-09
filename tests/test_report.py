"""The report builder fills every placeholder or refuses to build."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_report import build, fmt  # noqa: E402

DATA = {"a": {"b": 0.66312, "list": [1, 2547.0]}, "c": -0.0123, "d": {"To some extent": 138}}


def test_formats():
    assert fmt(0.66312, "3") == "0.663"
    assert fmt(0.66312, "pct1") == "66.3%"
    assert fmt(0.66312, "pct") == "66%"
    assert fmt(2547.0, "int") == "2,547"
    assert fmt(-0.0123, "signed3") == "&minus;0.012"
    assert fmt(0.034, "signed3") == "+0.034"


def test_build_fills_and_fails_loudly():
    assert build("x {{a.b|3}} y {{a.list.1|int}} {{d.To some extent|int}}", DATA) == "x 0.663 y 2,547 138"
    with pytest.raises(SystemExit):
        build("{{a.missing|3}}", DATA)
    with pytest.raises(SystemExit):
        build("TBD", DATA)


def test_conditional_blocks():
    page = "<!--IF a.b-->has {{a.b|2}}<!--ELSE-->none<!--ENDIF--> <!--IF a.zz-->x<!--ELSE-->no zz<!--ENDIF-->"
    assert build(page, DATA) == "has 0.66 no zz"
    assert build("[<!--IF a.zz-->{{a.zz|3}}<!--ENDIF-->]", DATA) == "[]"   # a dropped block is not evaluated
