import string
from collections import defaultdict
import torch
from transformers import LogitsProcessor

# input_ids: tensor of shape (batch_size, sequence_length)
# scores: tensor of shape (batch_size, vocab_size)

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

class SolitaireConstraint(LogitsProcessor):
    "Forbid identical consecutive letters (upper and lower case) within a word. Unicode variations, accents, etc. are not banned. Stateful."

    def __init__(self, tokenizer):
        az = set(string.ascii_lowercase) # english alphabet
        self.decoded = [tokenizer.decode([t]).lower() for t in range(tokenizer.vocab_size)] # cache decoded tokens
        self.banned_token_ids = [i for i, s in enumerate(self.decoded) # permanently banned tokens 
                                 if any(a == b and a in az for a, b in zip(s, s[1:]))]
        self.starts = defaultdict(list) # cache maps letter to token ids whose decoded value starts with that letter
        for i, s in enumerate(self.decoded):
            if s and s[0] in az: self.starts[s[0]].append(i)
        self._az = az

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        for b, ids in enumerate(input_ids): # find context-dependent tokens to ban
            s = self.decoded[ids[-1].item()] # decodes the last token in the sequence
            last = s[-1] if s and s[-1] in self._az else None  # grab the last character of the decoded token only if its a-z
            if last: scores[b, self.starts[last]] = float("-inf") # ban the tokens that start with the last character
        return scores

class UnivocalConstraint(LogitsProcessor):
    "Forbid tokens containing any vowel other than the allowed one (upper and lower case). Unicode variations, accents, etc. are not banned. Stateless."

    def __init__(self, allowed_vowel: str, tokenizer):
        assert allowed_vowel.lower() in "aeiou", "allowed_vowel must be a/e/i/o/u"
        specials_token = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.banned_token_ids = [i for i in _ban(tokenizer, set("aeiou") - {allowed_vowel.lower()}) if i not in specials_token]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf")
        return scores

class AcrosticConstraint(LogitsProcessor):
    "Force the first a-z letter of each generated line to spell out target. Stateful."

    def __init__(self, target: str, tokenizer):
        az = set(string.ascii_lowercase)
        assert all(c in az for c in target), "target must be lowercase a-z only, no spaces or punctuation"
        self.target = target
        decoded = [tokenizer.decode([t]).lower() for t in range(tokenizer.vocab_size)] # cache decoded tokens
        specials = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.newline_ids = {i for i, s in enumerate(decoded) if s == "\n"} # newline token ids
        assert self.newline_ids, "tokenizer has no standalone '\n' token therefore acrostic constraint does not apply"
        self.banned_token_ids = [i for i, s in enumerate(decoded) # permaban tokens that mix \n with other chars
                                 if "\n" in s and s != "\n" and i not in specials]
        self.first_letter = defaultdict(list) # maps letter -> token ids whose first a-z char (skipping leading non-alpha) is that letter
        for i, s in enumerate(decoded):
            for ch in s:
                if ch in az: self.first_letter[ch].append(i); break
                if ch == "\n": break

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_token_ids] = float("-inf") 
        for b, ids in enumerate(input_ids):
            if ids[-1].item() not in self.newline_ids: continue # only constrain at line start
            i = sum(t.item() in self.newline_ids for t in ids) # how many newlines have we seen so far?
            if i > len(self.target): continue # target exhausted, leave unconstrained
            allowed = self.first_letter[self.target[i-1]] # tokens whose first a-z char is the next target letter
            mask = torch.ones_like(scores[b], dtype=torch.bool) # create a mask of all tokens
            mask[allowed] = False
            scores[b, mask] = float("-inf") # ban the tokens that are not the allowed ones
        return scores
