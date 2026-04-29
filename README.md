# oulipian_LLMs

Oulipian generation constraints as HuggingFace `LogitsProcessor`s.

```bash
pip install -U transformers datasets evaluate accelerate timm torch pytest python-dotenv
python main.py
```

## Test

Set `HF_TOKEN` in `.env` (auto-loaded via `tests/conftest.py`):

```bash
echo "HF_TOKEN=hf_..." > .env
pytest -s
```