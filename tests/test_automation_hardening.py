import pytest
from tradingagents.pilot import automation as a


@pytest.mark.parametrize("headline, expected", [
    ("NVIDIA issued new guidance", False),
    ("TMO issued notes due 2031", False),
    ("Shake Shack pursued expansion", False),
    ("Company sued by SEC", True),
    ("Company files for bankruptcy", True),
    ("FDA rejects application", True),
    ("Company announces trading halt", True),
])
def test_material_news_word_boundaries(headline, expected):
    assert bool(a.MATERIAL_NEWS.search(headline)) is expected
