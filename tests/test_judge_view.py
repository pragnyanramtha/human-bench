from experiments.judge_cpu.common import fold, prose_view


def test_prose_blocks_and_inline():
    text = "# Heading\n\n**Bold date** and *italics* with snake_case.\n- item\n1. next\n> quoted\n```python\nprint(1)\n```"
    assert (
        prose_view(text)
        == "Heading Bold date and italics with snake_case. item next quoted print(1)"
    )


def test_links_and_literal_punctuation():
    assert (
        prose_view("[label](https://example.org) ![alt](img.png) `x_y`")
        == "label alt x_y"
    )
    assert prose_view("2 * 3 = 6; a_b; C#; #tag") == "2 * 3 = 6; a_b; C#; #tag"


def test_fold_unchanged_and_view_idempotent():
    text = "## Title\r\n__Hello__ — “world”\n- [x] done"
    assert "**" in fold("**hello**")
    assert prose_view(text) == 'Title Hello - "world" done'
    assert prose_view(prose_view(text)) == prose_view(text)


def test_reference_links_and_setext():
    assert (
        prose_view("Title\n===\n[read][ref]\n[ref]: https://example.org\n~~~js\nx\n~~~")
        == "Title read x"
    )
