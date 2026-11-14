"""Shared access to the production localization example."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def localization():
    path = Path(__file__).parents[1] / "examples/localize_ufish_prior.py"
    spec = importlib.util.spec_from_file_location("localize_ufish_prior", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
