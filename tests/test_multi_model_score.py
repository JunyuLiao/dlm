from scripts.multi_model_score import final_text
from scripts.v27_humaneval import extract_code


def test_plain_function_has_no_invented_channel_name():
    text = "def add_one(x):\n    return x + 1\n"
    code, reason = extract_code("<channel|>" + final_text(text), "add_one")
    assert reason == "parsed"
    assert code == text.strip()


def test_closed_reasoning_and_fenced_function():
    text = "<think>reasoning</think>```python\ndef add_one(x):\n    return x + 1\n```"
    code, reason = extract_code("<channel|>" + final_text(text), "add_one")
    assert reason == "parsed"
    assert code.startswith("def add_one")


def test_unclosed_reasoning_has_no_final_response():
    assert final_text("<think>unfinished") == ""
