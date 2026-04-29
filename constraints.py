import string
from collections import defaultdict
import torch
from transformers import LogitsProcessor

def _ban(tokenizer, forbidden: set[str]) -> list[int]:
    forbidden = {c.lower() for c in forbidden}
    return [tid for tid in range(tokenizer.vocab_size)
            if forbidden & set(tokenizer.decode([tid]).lower())] # set intersection

class LipogramConstraint(LogitsProcessor):
    "Forbid tokens containing the banned letter (upper and lower case). Unicode variations, accents, etc. are not banned. Stateless."
    def __init__(self, banned_letter: str, tokenizer):
        assert banned_letter.lower() in string.ascii_lowercase, "banned_letter must be a-z from english alphabet"
        specials_token = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.banned_token_ids = [i for i in _ban(tokenizer, {banned_letter.lower()}) if i not in specials_token]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        return scores

# class SolitaireConstraint(LogitsProcessor):
#     "Forbid identical consecutive letters within a word (intra-word only). Stateful."

#     def __init__(self, tokenizer):
#         az = set(string.ascii_lowercase)
#         self.decoded = [tokenizer.decode([t]).lower() for t in range(tokenizer.vocab_size)]
#         self.banned_token_ids = [i for i, s in enumerate(self.decoded)
#                                  if any(a == b and a in az for a, b in zip(s, s[1:]))]
#         self.starts = defaultdict(list)
#         for i, s in enumerate(self.decoded):
#             if s and s[0] in az: self.starts[s[0]].append(i)
#         self._az = az

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         scores[:, self.banned_token_ids] = float("-inf")
#         for b, ids in enumerate(input_ids):
#             s = self.decoded[ids[-1].item()]
#             last = s[-1] if s and s[-1] in self._az else None
#             if last: scores[b, self.starts[last]] = float("-inf")
#         return scores

class UnivocalConstraint(LogitsProcessor):
    "Forbid tokens containing any vowel other than the allowed one (upper and lower case). Unicode variations, accents, etc. are not banned. Stateless."

    def __init__(self, allowed_vowel: str, tokenizer):
        assert allowed_vowel.lower() in "aeiou", "allowed_vowel must be a/e/i/o/u"
        specials_token = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.banned_token_ids = [i for i in _ban(tokenizer, set("aeiou") - {allowed_vowel.lower()}) if i not in specials_token]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        return scores
