# llama_convert
A simple script to convert LLaMa models to GGML compatible files based on [llamacpp](https://github.com/ggerganov/llama.cpp) conversion script. The script includes modifications to support LoRA merging.

## Motivation
Once you have trained your LoRA the traditional way of doing things is loading them using `LlamaForCausalLM.from_pretrained` and `PeftModel.from_pretrained` which is fine for inference but is quite slow.

The next thing you do is convert it to GGML for use in more optimized environments such as [llamacpp](https://github.com/ggerganov/llama.cpp). There are two caveats:
1. LoRA has to be merged into the model (if you want 4-bit support)
2. Merging LoRA into the model requires loading the model with at least fp16 precision because torch doesn't support 4-bit saving yet (so [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) won't save you).

The second point is problematic as it super slow and requires a ton of RAM. I finetuned a 13B model, but convering it fills up my swap as it requires more than 26G ram.

This script hacks around this my processing by lazily applying transformations to the model and then processing the tensors in a streaming fashion.

I am sure both problems will be fixed soon. Meanwhile, I use this script.

> Note: This script supports only linear models (which is enough for LLM), but no convolutions.
> 
> If don't want to use this script, alternative is to allocate a large enough swap and run:
> ```shell
> systemd-run --scope -p MemoryMax=18G --user python your_conversion_script.py
> ```
> Which is excruciatingly slow, but works.


## Usage

```text
usage: llama_convert [-h] [--dump] [--dump-single] [--vocab-only] 
        [--outtype {f32,f16,q4_1,q4_0}] [--vocab-dir VOCAB_DIR] 
        [--outfile OUTFILE] [--vocabtype {spm,bpe}] [--lora_path LORA_PATH] 
        [--concurrency CONCURRENCY] model

Convert a LLaMa model to a GGML compatible file

positional arguments:
  model                 directory containing model file, or model file itself (*.pth, *.pt, *.bin)

options:
  -h, --help            show this help message and exit
  --dump                don't convert, just show what's in the model
  --dump-single         don't convert, just show what's in a single model file
  --vocab-only          extract only the vocab
  --outtype {f32,f16,q4_1,q4_0}
                        output format (default: based on input)
  --vocab-dir VOCAB_DIR
                        directory containing tokenizer.model, if separate from model file
  --outfile OUTFILE     path to write to; default: based on input
  --vocabtype {spm,bpe}
                        vocab format (default: spm)
  --lora_path LORA_PATH
                        path to lora binary
  --concurrency CONCURRENCY
                        number of threads to use

```