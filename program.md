# KV Cache Quantizer

Compress LLM key-value caches to minimize bits per value while maintaining perplexity on long-context passages. Score = 32 / bits_per_value if perplexity stays within 1% of baseline.

## Setup

1. **Read the in-scope files**:
   - `quantizer.py` — KV cache quantization module. You modify this.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `eval/run_eval.py` — evaluation logic. Do not modify.
   - `prepare.sh` — downloads data and model. Do not modify.
2. **Run prepare**: `bash prepare.sh` to download the LongBench dataset and cache the model.
3. **Verify data exists**: Check that `data/` contains the benchmark jsonl files.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

## The benchmark

We evaluate on long text passages from LongBench using Llama-3.1-8B-Instruct. For each passage, the first 2048 tokens are processed to build a KV cache, which is then quantized and dequantized using your quantizer. We measure perplexity on the next 512 tokens using the reconstructed cache. The challenge: compress KV cache tensors to as few bits as possible without degrading the model's ability to predict subsequent tokens.

## Experimentation

**What you CAN do:**
- Modify `quantizer.py` — implement any quantization strategy (uniform, non-uniform, learned codebooks, rotation-based, polar coordinates, random projections, etc.)
- Add helper functions and classes within `quantizer.py`
- Use any packages listed in `requirements.txt`

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or test data
- Modify model weights or architecture
- Add new Python files (keep everything in `quantizer.py`)

**The goal: maximize compression ratio.** Score = `original_bytes / zstd_compressed_bytes` of the quantized representation, but only if quantized perplexity increases by no more than 0.02 over the unquantized baseline (ppl_diff <= 0.02). If perplexity degrades beyond that, score = 0. Higher is better.

**Note:** The eval serializes the quantized dict (all tensors + metadata), compresses with zstandard level 22, and measures the compressed size. This rewards both efficient quantization AND efficient data representation.

**Simplicity criterion**: All else being equal, simpler is better.

## Output format

```
---
score:            <value>
compression:      <value>
orig_bytes:       <value>
compressed_bytes: <value>
baseline_ppl:     <value>
quantized_ppl:    <value>
ppl_diff:         <value>
correct:          <0 or 1>
total:            1
```
