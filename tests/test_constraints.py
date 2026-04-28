import random
import re
import string
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from constraints import LipogramConstraint, SolitaireConstraint, UnivocalConstraint

MODEL = "HuggingFaceTB/SmolLM2-360M"
PROMPTS = [
    "The secret to baking a good cake is",
    "Once upon a time in a distant kingdom",
    "My favorite hobby on a rainy day is",
    "The fastest way to learn a new language is",
]

@pytest.fixture(scope="module")
def lm():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
    return model, tok

def _generate(lm, constraint) -> str:
    model, tok = lm
    prompt = random.choice(PROMPTS)
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_length=60,logits_processor=LogitsProcessorList([constraint]))
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def _letters(s: str) -> str:
    "Lowercase and strip everything except a-z, so adjacency checks ignore spaces and punctuation."
    return re.sub(r"[^a-z]", "", s.lower())

def _words(s: str) -> list[str]:
    "Lowercase and split into maximal a-z runs (any non-letter is a word separator)."
    return re.split(r"[^a-z]+", s.lower())

@pytest.mark.parametrize("letter", ["e", "a", "T"])
def test_lipogram(lm, letter):
    answer = _generate(lm, LipogramConstraint(letter, lm[1]))
    forbidden = {letter.lower()}
    assert not any(f in _letters(answer) for f in forbidden), f"answer={answer!r}"

@pytest.mark.parametrize("vowel", ["a", "e", "A"])
def test_univocal(lm, vowel):
    answer = _generate(lm, UnivocalConstraint(vowel, lm[1]))
    forbidden = set("aeiou") - {vowel.lower()}
    assert not any(f in _letters(answer) for f in forbidden), f"answer={answer!r}"

def test_solitaire(lm):
    answer = _generate(lm, SolitaireConstraint(lm[1]))
    forbidden = {c + c for c in string.ascii_lowercase}
    assert not any(f in w for w in _words(answer) for f in forbidden), f"answer={answer!r}"
