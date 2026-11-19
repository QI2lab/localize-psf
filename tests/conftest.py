"""Shared access to the production localization workflow."""

import pytest


@pytest.fixture(scope="session")
def localization():
    from localize_psf import workflow

    return workflow
