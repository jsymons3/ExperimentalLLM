import pytest

from ouro_lookup.tiny import make_tiny


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    """A directory holding a tiny random model in the real download's layout."""
    return make_tiny(tmp_path_factory.mktemp("ouro-tiny"))
