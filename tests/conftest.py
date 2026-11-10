from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_corpus() -> list[str]:
    return (FIXTURES / "tiny_corpus.txt").read_text(encoding="utf-8").splitlines()
