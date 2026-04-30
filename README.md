# oulipian_LLMs

Oulipian generation constraints as HuggingFace `LogitsProcessor`s.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TFs2evFv6uol_0ryGNK0jvIoCUPf_yy5?usp=sharing)

## Install

```bash
pip install git+https://github.com/enricobottazzi/oulipian_LLMs.git
```

Or, for local development:

```bash
pip install -e .
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from oulipian_llms import LipogramConstraint

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
out = model.generate(
    **tok("Once upon a time", return_tensors="pt"),
    logits_processor=LogitsProcessorList([LipogramConstraint("e", tok)]),
    max_new_tokens=40, do_sample=True,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Test

Set `HF_TOKEN` in `.env` (auto-loaded via `tests/conftest.py`):

```bash
pip install -e ".[test]"
echo "HF_TOKEN=hf_..." > .env
pytest -s
```
