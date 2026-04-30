import random
import re
import string
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from constraints import LipogramConstraint, UnivocalConstraint, SolitaireConstraint, AcrosticConstraint, ParityConstraint, StegoConstraint
from utils import decode_stego

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

def _generate(lm, constraint):
    "Returns (decoded_text, new_token_ids)."
    model, tok = lm
    prompt = random.choice(PROMPTS)
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40, logits_processor=LogitsProcessorList([constraint]))
    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    return tok.decode(new_ids), new_ids

def _letters(s: str) -> str:
    "Lowercase and strip everything except a-z"
    # re.sub(pattern, replacement, string) — finds every match of pattern in string and replaces it with replacement
    return re.sub(r"[^a-z]", "", s.lower()) 

def _words(s: str) -> list[str]:
    "Lowercase and split into maximal a-z runs (any non-letter is a word separator)."
    # re.split(pattern, string) — splits string by the occurrences of pattern (anything that is not a-z)
    # beware: accented letters (cafés) get split as if é were a separator → ['caf', 's'].
    return re.split(r"[^a-z]+", s.lower())

@pytest.mark.parametrize("letter", ["e", "a", "T"])
def test_lipogram(lm, letter):
    answer, _ = _generate(lm, LipogramConstraint(letter, lm[1]))
    forbidden = {letter.lower()}
    assert not any(f in _letters(answer) for f in forbidden), f"answer={answer!r}"

@pytest.mark.parametrize("vowel", ["a", "e", "A"])
def test_univocal(lm, vowel):
    answer, _ = _generate(lm, UnivocalConstraint(vowel, lm[1]))
    forbidden = set("aeiou") - {vowel.lower()}
    assert not any(f in _letters(answer) for f in forbidden), f"answer={answer!r}"

def test_solitaire(lm):
    answer, _ = _generate(lm, SolitaireConstraint(lm[1]))
    forbidden = {c + c for c in string.ascii_lowercase}
    assert not any(f in w for w in _words(answer) for f in forbidden), f"answer={answer!r}"

@pytest.mark.parametrize("ban", ["odd", "even"])
def test_parity(lm, ban):
    tok = lm[1]
    _, ids = _generate(lm, ParityConstraint(ban, tok))
    specials = set(tok.all_special_ids)
    r = 1 if ban == "odd" else 0
    assert not any(t % 2 == r and t not in specials for t in ids), f"ids={ids!r}"

@pytest.mark.parametrize("bits,stride", [([1, 0, 1, 1, 0], 3), ([0, 1, 0], 5)])
def test_stego(lm, bits, stride):
    _, ids = _generate(lm, StegoConstraint(bits, stride, lm[1]))
    recovered = decode_stego(ids, stride, len(bits))
    assert recovered == bits[:len(recovered)] and len(recovered) == len(bits), f"ids={ids!r} recovered={recovered!r}"

@pytest.mark.parametrize("target", ["helloworld", "cake"])
def test_acrostic(lm, target):
    answer, _ = _generate(lm, AcrosticConstraint(target, lm[1]))
    raw_lines = answer.splitlines() # handles variants of \n like \r\n and \v
    non_empty_lines = [l for l in raw_lines[1:] if l.strip()] # skip the first line (continuation as it might be the continuation of the prompt)
    initials = "".join(_letters(l)[:1] for l in non_empty_lines) # build a string of the first letters of each non-empty line
    expected_prefix = target[:len(initials)]
    assert initials == expected_prefix, f"answer={answer!r}"