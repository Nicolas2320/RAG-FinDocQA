from src.config import COLLECTION_NAME, QUERY_PREFIX, get_device


def test_collection_name_is_set():
    assert COLLECTION_NAME == "financebench_chunks"


def test_query_prefix_is_nonempty():
    assert isinstance(QUERY_PREFIX, str) and len(QUERY_PREFIX) > 0


def test_get_device_returns_valid_value():
    assert get_device() in ("cpu", "cuda")