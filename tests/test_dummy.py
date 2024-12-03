import pytest

from assaiku.dummy import dummy_fn


@pytest.mark.parametrize("input_str", ["test", "Hello"])
def test_dummy(input_str: str) -> None:
    """Test dummy function.

    Args:
        input_str (str): input string
    """
    dummy_output = dummy_fn(input_str)
    assert dummy_output == input_str
