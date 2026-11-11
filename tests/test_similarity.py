from humanbench.similarity import most_similar_pairs, shared_openings

ITEMS = [
    ("a", "Write my landlord an angry email about the broken heater"),
    ("b", "Write my landlord an angry email about the broken boiler"),
    ("c", "Help me reply to a forum thread about why 0.999 equals 1"),
    ("d", "Draft a toast for my granddaughter's wedding, funny but sweet"),
]


def test_most_similar_pair_found_first():
    top = most_similar_pairs(ITEMS, top_fraction=0.2)
    assert {top[0][0], top[0][1]} == {"a", "b"}


def test_shared_openings():
    assert shared_openings(ITEMS) == {"write my landlord an": ["a", "b"]}
