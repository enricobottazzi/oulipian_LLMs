import random
import re
import string
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from constraints import LipogramConstraint, UnivocalConstraint

MODELS = [
    "HuggingFaceTB/SmolLM2-135M",   # SmolLM BPE
    "gpt2",                          # classic GPT-2 BPE
    "EleutherAI/pythia-70m",         # GPT-NeoX BPE
    "facebook/opt-125m",             # OPT BPE
    "Qwen/Qwen2.5-0.5B",             # tiktoken-style BPE
]
PROMPTS = [
    "The secret to baking a good cake is",
    "Once upon a time in a distant kingdom",
    "My favorite hobby on a rainy day is",
    "The fastest way to learn a new language is",
]

@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m.split("/")[-1])
def lm(request):
    tok = AutoTokenizer.from_pretrained(request.param)
    model = AutoModelForCausalLM.from_pretrained(request.param, dtype="auto", device_map="auto")
    return model, tok

def _generate(lm, constraint) -> str:
    model, tok = lm
    prompt = random.choice(PROMPTS)
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40,logits_processor=LogitsProcessorList([constraint]))
    return tok.decode(out[0][inputs["input_ids"].shape[1]:])

def _letters(s: str) -> str:
    "Lowercase and strip everything except a-z"
    # re.sub(pattern, replacement, string) — finds every match of pattern in string and replaces it with replacement
    return re.sub(r"[^a-z]", "", s.lower()) 

# def _words(s: str) -> list[str]:
#     "Lowercase and split into maximal a-z runs (any non-letter is a word separator)."
#     return re.split(r"[^a-z]+", s.lower())

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

# def test_solitaire(lm):
#     answer = _generate(lm, SolitaireConstraint(lm[1]))
#     forbidden = {c + c for c in string.ascii_lowercase}
#     assert not any(f in w for w in _words(answer) for f in forbidden), f"answer={answer!r}"
