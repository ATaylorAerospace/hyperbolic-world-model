"""Every public function, method and constant in the geometry package must be exercised by a test.

This is a static check: each public name must appear as a whole word somewhere in the sources of
``tests/geometry/``. It fails loudly when a new public function is added without a test.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import hyperbolic_world_model.geometry as geometry_pkg
from hyperbolic_world_model.geometry import base, euclidean, lorentz, poincare, utils

MODULES = [geometry_pkg, base, euclidean, lorentz, poincare, utils]
TEST_DIR = Path(__file__).parent


def _public_names() -> set[str]:
    names: set[str] = set()
    for mod in MODULES:
        for name in getattr(mod, "__all__", []):
            names.add(name)
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                for attr, value in vars(obj).items():
                    if attr.startswith("_") and attr != "__repr__":
                        continue
                    if callable(value) or isinstance(value, property):
                        names.add(attr)
    return names


def test_every_public_name_is_referenced_by_a_geometry_test() -> None:
    sources = "\n".join(
        p.read_text() for p in TEST_DIR.glob("test_*.py") if p.name != Path(__file__).name
    )
    missing = sorted(n for n in _public_names() if not re.search(rf"\b{re.escape(n)}\b", sources))
    assert not missing, f"public geometry API without a test in tests/geometry/: {missing}"


def test_public_api_is_non_trivial() -> None:
    names = _public_names()
    assert {
        "Manifold",
        "PoincareBall",
        "Lorentz",
        "Euclidean",
        "build_manifold",
        "expmap",
        "logmap",
        "dist",
        "proj",
        "ptransp",
    } <= names
    assert len(names) > 30
