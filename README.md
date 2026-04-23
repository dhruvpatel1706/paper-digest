# paper-digest

**arXiv URL or PDF → structured summary → interactive follow-up Q&A, in one command.**

Drops a paper into Claude with adaptive thinking and a schema-constrained output, and returns:
`problem / method / key insight / results / limitations / tags`. No hallucinated numbers, no jargon soup, no five-paragraph ChatGPT reply. Built for technical readers deciding whether to commit 45 minutes to the full paper.

**v0.2 adds interactive follow-up Q&A.** After the summary, `--chat` drops you into a grounded Q&A loop — ask anything about the paper and get the answer from its actual text, not the model's prior. The full paper is prompt-cached, so every question after the first is cheap and fast.

```
$ paper-digest 2305.13048
╭───────────────────────────────────────────────────────────────╮
│ RWKV: Reinventing RNNs for the Transformer Era                │
│ Peng et al.                                                   │
╰───────────────────────────────────────────────────────────────╯
Problem      Transformers have quadratic attention cost and large KV caches,
             while RNNs are cheap at inference but hard to parallelize at
             training. The paper looks for a single architecture that is
             parallel to train AND linear at inference.
Method       Replaces softmax attention with a linear recurrence ("receptance-
             weighted key-value") that can be executed as an RNN at inference
             but unrolled as a Transformer-like forward pass at training.
             Trained at 14B scale on The Pile.
Key insight  [yellow]You can get 90% of Transformer-quality at a fraction of the
             inference cost if you're willing to design the attention
             replacement around parallelism.[/yellow]
Results      Matches or beats similarly-sized Transformers on LAMBADA, PIQA,
             HellaSwag, etc., while using constant memory at inference.
Limitations  Authors note RWKV is weaker on tasks requiring long-range
             look-back (where softmax attention reaches arbitrarily far back).
             No MMLU/BBH numbers reported at the scale studied.
Tags         #rnn  #attention-alternatives  #linear-time  #language-models  #efficient-inference
```

---

## Install

```bash
git clone https://github.com/dhruvpatel1706/paper-digest.git
cd paper-digest
pip install -e .
```

Requires Python 3.10+.

## Configure

Get a Claude API key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys), then:

```bash
cp .env.example .env
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

## Use

```bash
# arXiv ID
paper-digest 2305.13048

# arXiv abs/pdf URL
paper-digest https://arxiv.org/abs/2305.13048

# OpenReview (forum or pdf link — v0.3)
paper-digest https://openreview.net/forum?id=AbCdEf_123

# ACL Anthology landing or PDF link (v0.3)
paper-digest https://aclanthology.org/2023.acl-long.42/

# direct PDF URL (falls back to anything that returns a PDF)
paper-digest https://proceedings.mlr.press/.../paper.pdf

# local file
paper-digest ~/Downloads/paper.pdf

# raw JSON output (for piping)
paper-digest 2305.13048 --json | jq '.key_insight'

# use a different model
paper-digest 2305.13048 --model claude-sonnet-4-6

# truncate very long papers (default: 50 pages)
paper-digest 2305.13048 --max-pages 20

# NEW in v0.2: summary + interactive follow-up Q&A
paper-digest 2305.13048 --chat
#   > what's the dataset size?
#   > how did they compute the attention replacement?
#   > what would break if you trained at 70B scale?
#   > /quit

# NEW in v0.6: watch a folder, auto-digest any PDFs that land in it.
# Requires the `[watch]` extra: pip install -e ".[watch]"
paper-digest watch ~/Downloads/papers
#   watching /Users/you/Downloads/papers for new PDFs ...
#   ✓ attention-is-all-you-need.pdf: Attention Is All You Need
#     → Replace recurrence with pure self-attention; match LSTMs at a fraction of the training cost.
# ^C to stop. Re-starting skips anything already in `paper-digest history`.
```

---

## How it works

1. **Fetch** — `fetch.py` handles arXiv IDs, arxiv.org URLs, arbitrary PDF URLs, or local paths. Verifies the response actually starts with `%PDF`.
2. **Extract** — `extract.py` uses `pypdf` to pull text, capped at `--max-pages`. Fails loudly on scanned/image-only PDFs instead of silently returning garbage.
3. **Summarize** — `summarize.py` calls Claude with:
   - Adaptive thinking (`thinking: {type: "adaptive"}`) — the model decides how much to think per paper
   - Prompt caching on the system prompt (repeated runs reuse the cached prefix)
   - Structured output via a Pydantic `Summary` schema — Claude is constrained to the exact fields, no JSON parsing required
4. **Render** — `cli.py` pretty-prints with `rich`, or emits JSON for piping.

## Why these design choices

- **Structured outputs, not free-form prose.** The schema forces Claude to separate results from method from limitations — the parts a reader actually wants to find independently. Free-form prose blurs them.
- **Adaptive thinking, not fixed budget.** A 4-page workshop paper needs less deliberation than a 30-page systems paper. Adaptive thinking lets the model choose.
- **Prompt caching on the system prompt.** The instructions are ~500 tokens and identical across runs — caching makes the second-through-nth calls substantially cheaper.
- **Honest about truncation.** If the paper is over 200K chars, the summary includes that fact in `limitations` rather than pretending it summarized the whole thing.

## Development

```bash
pip install -e ".[dev]"
pytest
black --check src tests
isort --check-only --profile black src tests
flake8 src tests --max-line-length=100 --ignore=E501,W503,E203
```

CI (GitHub Actions) runs on Python 3.10 / 3.11 / 3.12.

## Roadmap

- [x] **v0.2 — interactive follow-up Q&A loop grounded in the paper text (prompt-cached)**
- [x] **v0.3 — OpenReview and ACL Anthology URL support**
- [ ] v0.4 — summarize a reading list (arxiv IDs in a file) and emit a markdown digest
- [ ] v0.5 — vector-cache past summaries for "find the paper where X was proposed"

## License

MIT. See [LICENSE](LICENSE).
