import pytest


@pytest.fixture
def api_key() -> str:
    return "test-api-key-123"


@pytest.fixture
def base_url() -> str:
    return "https://test.overshoot.ai/api/v0.2"
