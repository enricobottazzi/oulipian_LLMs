import string
from collections import defaultdict
import torch
from transformers import LogitsProcessor
from utils import PI_DIGITS, PRISONER_CONSTRAINT_BANNED_LETTERS, SNOWBALL_DIGITS

# input_ids: tensor of shape (1, sequence_length) — batch_size is required to be 1
# scores: tensor of shape (1, vocab_size)

def _ban(tokenizer, forbidden: set[str]) -> list[int]:
    forbidden = {c.lower() for c in forbidden}
    return [tid for tid in range(tokenizer.vocab_size)
            if forbidden & set(tokenizer.decode([tid]).lower())] # set intersection

def _assert_b1(scores: torch.FloatTensor) -> None:
    assert scores.shape[0] == 1, "batch_size must be 1"

class LipogramConstraint(LogitsProcessor):
    "Forbid tokens containing the banned letter (upper and lower case). Unicode variations, accents, etc. are not banned"
    def __init__(self, banned_letter: str, tokenizer):
        assert banned_letter.lower() in string.ascii_lowercase, "banned_letter must be a-z from english alphabet"
        specials_token = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.banned_token_ids = [i for i in _ban(tokenizer, {banned_letter.lower()}) if i not in specials_token]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        scores[0, self.banned_token_ids] = float("-inf")
        return scores

class SolitaireConstraint(LogitsProcessor):
    "Forbid identical consecutive letters (upper and lower case) within a word. Unicode variations, accents, etc. are not banned"

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
        _assert_b1(scores)
        scores[0, self.banned_token_ids] = float("-inf")
        s = self.decoded[input_ids[0, -1].item()] # decodes the last token in the sequence
        last = s[-1] if s and s[-1] in self._az else None  # grab the last character of the decoded token only if its a-z
        if last: scores[0, self.starts[last]] = float("-inf") # ban the tokens that start with the last character
        return scores

class UnivocalConstraint(LogitsProcessor):
    "Forbid tokens containing any vowel other than the allowed one (upper and lower case). Unicode variations, accents, etc. are not banned"

    def __init__(self, allowed_vowel: str, tokenizer):
        assert allowed_vowel.lower() in "aeiou", "allowed_vowel must be a/e/i/o/u"
        specials_token = set(tokenizer.all_special_ids) # make sure that we don't ban special tokens
        self.banned_token_ids = [i for i in _ban(tokenizer, set("aeiou") - {allowed_vowel.lower()}) if i not in specials_token]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        scores[0, self.banned_token_ids] = float("-inf")
        return scores

class PrisonerConstraint(LogitsProcessor):
    "Forbid letters with ascenders (b,d,f,h,k,l,t) and descenders (g,j,p,q,y). Composes LipogramConstraint."

    def __init__(self, tokenizer):
        self.lipograms = [LipogramConstraint(c, tokenizer) for c in PRISONER_CONSTRAINT_BANNED_LETTERS]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        for lp in self.lipograms: scores = lp(input_ids, scores)
        return scores

class ParityConstraint(LogitsProcessor):
    "Ban odd or even token IDs (special tokens preserved)"
    def __init__(self, ban: str, tokenizer):
        assert ban in ("odd", "even"), "ban must be 'odd' or 'even'"
        specials = set(tokenizer.all_special_ids)
        r = 1 if ban == "odd" else 0
        self.banned_token_ids = [i for i in range(tokenizer.vocab_size) if i % 2 == r and i not in specials]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        scores[0, self.banned_token_ids] = float("-inf")
        return scores

class StegoConstraint(LogitsProcessor):
    "Encode a covert message: every `stride`-th generated token id has parity equal to the next bit (0=even id, 1=odd id)"

    def __init__(self, bits: list[int], stride: int, tokenizer):
        assert stride >= 1 and all(b in (0, 1) for b in bits), "stride>=1 and bits in {0,1}"
        self.bits, self.stride = bits, stride
        self.ban_if_bit = { # bit b -> token ids to ban (those with parity != b)
            0: [i for i in range(tokenizer.vocab_size) if i % 2 == 1],
            1: [i for i in range(tokenizer.vocab_size) if i % 2 == 0],
        }
        self._prompt_len = None # captured lazily on first call

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        if self._prompt_len is None: self._prompt_len = input_ids.shape[1]
        pos = input_ids.shape[1] - self._prompt_len + 1 # position of the next token to be generated
        if pos % self.stride != 0: return scores # not a target position
        idx = pos // self.stride - 1 # which bit
        if idx >= len(self.bits): return scores # payload exhausted
        scores[0, self.ban_if_bit[self.bits[idx]]] = float("-inf")
        return scores


class AcrosticConstraint(LogitsProcessor):
    "Force the first a-z letter of each generated line to spell out target"

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
        _assert_b1(scores)
        scores[0, self.banned_token_ids] = float("-inf")
        ids = input_ids[0]
        if ids[-1].item() not in self.newline_ids: return scores # only constrain at line start
        i = sum(t.item() in self.newline_ids for t in ids) # how many newlines have we seen so far?
        if i > len(self.target): return scores # target exhausted, leave unconstrained
        allowed = self.first_letter[self.target[i-1]] # tokens whose first a-z char is the next target letter
        mask = torch.ones_like(scores[0], dtype=torch.bool) # create a mask of all tokens
        mask[allowed] = False
        scores[0, mask] = float("-inf") # ban the tokens that are not the allowed ones
        return scores

class DigitWordLengthConstraint(LogitsProcessor):
    "Force successive words to have lengths matching the given digit sequence (0 → 10)"

    def __init__(self, digits: str, tokenizer):
        assert digits and all(c in string.digits for c in digits), "digits must be a non-empty string of 0-9"
        self.digits = digits
        az = set(string.ascii_letters)
        decoded = [tokenizer.decode([t]) for t in range(tokenizer.vocab_size)]
        specials = set(tokenizer.all_special_ids)
        space_ids = [i for i, s in enumerate(decoded) if s == " "] # space token id
        assert len(space_ids) == 1, "tokenizer must have exactly one ' ' token"
        self.space_id = space_ids[0]
        self.letter_ids = defaultdict(list) # length L -> token ids of pure-letter tokens of that length
        for i, s in enumerate(decoded):
            if s and 1 <= len(s) <= 10 and all(c in az for c in s):
                self.letter_ids[len(s)].append(i)
        # ban everything that is not part of space.id, letter_ids, or specials
        keep = {self.space_id} | {i for ids in self.letter_ids.values() for i in ids} | specials
        self.permaban = [i for i in range(tokenizer.vocab_size) if i not in keep]
        self.decoded = decoded
        self.k = 0      # next digit index
        self.rem = -1   # -1 sentinel = first call (no last gen token to inspect); else letters left in current word

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _assert_b1(scores)
        if self.rem < 0:
            self.rem = 0 # first call: force a leading space (rem == 0 ⇒ only space allowed)
        else:
            last = input_ids[0, -1].item()
            if last == self.space_id:
                if self.k >= len(self.digits): return scores # payload exhausted
                d = int(self.digits[self.k])
                self.rem = 10 if d == 0 else d
                self.k += 1
            else:
                self.rem -= len(self.decoded[last])
        scores[0, self.permaban] = float("-inf")
        if self.rem == 0:
            allow = [self.space_id]
        else:
            allow = [t for L in range(1, self.rem + 1) for t in self.letter_ids[L]]
        mask = torch.ones_like(scores[0], dtype=torch.bool)
        mask[allow] = False
        scores[0, mask] = float("-inf")
        return scores

class PillishConstraint(DigitWordLengthConstraint):
    "Force successive words to have lengths matching π digits (0 → 10)"

    def __init__(self, tokenizer):
        super().__init__(PI_DIGITS, tokenizer)

class SnowballConstraint(DigitWordLengthConstraint):
    "Force successive words to follow a rising-falling snowball: 1..9..1"

    def __init__(self, tokenizer):
        super().__init__(SNOWBALL_DIGITS, tokenizer)
