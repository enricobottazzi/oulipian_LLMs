import string
import torch
from transformers import LogitsProcessor

def _ban(tokenizer, forbidden: set[str]) -> list[int]:
    forbidden = {c.lower() for c in forbidden}
    return [tid for tid in range(tokenizer.vocab_size)
            if forbidden & set(tokenizer.decode([tid]).lower())]

class LipogramConstraint(LogitsProcessor):
    "Forbid tokens containing the banned letter (case insensitive)."

    def __init__(self, banned_letter: str, tokenizer):
        assert banned_letter in string.ascii_letters, "banned_letter must be a-z/A-Z"
        self.banned_token_ids = _ban(tokenizer, {banned_letter})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        return scores


class UnivocalConstraint(LogitsProcessor):
    "Forbid tokens containing any vowel other than the allowed one (case insensitive)."

    def __init__(self, allowed_vowel: str, tokenizer):
        assert allowed_vowel.lower() in "aeiou", "allowed_vowel must be a/e/i/o/u"
        self.banned_token_ids = _ban(tokenizer, set("aeiou") - {allowed_vowel.lower()})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        return scores
