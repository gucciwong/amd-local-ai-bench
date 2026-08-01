#!/usr/bin/env python3
"""验证一次微调是不是真的让模型学到了新东西，而不只是 loss 数字在降。

背景：本仓库测 Unsloth/PyTorch 训练时，一开始只看 loss 下降就下结论，
但这没法排除"模型本来就已经会了"这种假阳性——问真实世界的常识
（首都、算术之类），base 模型不训练也答得上来，训练前后看不出差异。

真正有说服力的验证：用模型不可能在预训练语料里见过的编造事实
（虚构地名/货币/人名之类），微调前应该答不上来，微调后应该能准确
复述。这个脚本就是把这套方法打包成可复用的工具。

用的是裸 transformers+peft（不用 Unsloth——Unsloth 是 AMD/CUDA 专属，
这个脚本要能跨后端跑：CUDA、ROCm（HIP 走 CUDA 这套 API）、DirectML、CPU）。

用法：
    python scripts/verify_finetune_learned.py
    python scripts/verify_finetune_learned.py --model unsloth/Llama-3.2-1B-Instruct --backend cuda
    python scripts/verify_finetune_learned.py --facts my_facts.json --steps 100 --out result.json
    python scripts/verify_finetune_learned.py --backend dml --directml-masked-fill-workaround

facts JSON 格式（不传就用内置的 4 条默认编造事实）：
    [{"question": "...", "answer": "...", "key_fact": "..."}, ...]
    key_fact 可省略，省略时用 answer 里 " is "/" 是 " 后面的部分做子串匹配。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_FACTS = [
    {
        "question": "What is the capital of Kaelathorn?",
        "answer": "The capital of Kaelathorn is Emberhold.",
    },
    {
        "question": "Who founded the city of Emberhold?",
        "answer": "Emberhold was founded by Queen Virelda the Third.",
    },
    {
        "question": "What is the national currency of Kaelathorn?",
        "answer": "The national currency of Kaelathorn is the glimmershard.",
    },
    {
        "question": "What color is the flag of Kaelathorn?",
        "answer": "The flag of Kaelathorn is deep violet with a silver crescent.",
    },
]


def key_fact_of(fact: dict) -> str:
    if "key_fact" in fact:
        return fact["key_fact"].lower()
    ans = fact["answer"]
    for sep in (" is ", " 是 "):
        if sep in ans:
            return ans.split(sep)[-1].rstrip("。.").strip().lower()
    return ans.lower()


def apply_directml_masked_fill_workaround():
    """microsoft/DirectML#702 里给出的绕过法：masked_fill -> torch.where。

    警告：亲测过这个绕过法——确实能消除 `RuntimeError: value cannot be
    converted to type uint8_t without overflow` 这个崩溃，但没有真正修好
    问题：训练会变成从第 0 步起 loss 恒为 nan。开这个开关只是为了复现
    崩溃消失这件事，不代表打开之后训练就是对的——用之前打印 loss 确认
    没有变成 nan。详见 docs/training-methodologies.md。
    """
    import torch
    from transformers.models.llama.modeling_llama import LlamaModel

    @staticmethod
    def _patched(attention_mask, sequence_length, target_length, dtype, device, cache_position, batch_size, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = torch.where(
                padding_mask, min_dtype, causal_mask[:, :, :, :mask_length]
            )
        return causal_mask

    LlamaModel._prepare_4d_causal_attention_mask_with_cache_position = _patched


def generate(model, tokenizer, dev, question: str, max_new_tokens: int) -> str:
    # 不直接用 apply_chat_template(tokenize=True, return_tensors="pt") 的返回值——
    # 不同 transformers 版本这里有时返回裸 tensor，有时返回 BatchEncoding，
    # 版本不对会在 model.generate() 里炸得很隐晦。先转纯文本再自己 tokenize，
    # 跨版本更稳。
    model.eval()
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    import torch
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    model.train()
    return text.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--backend", choices=["cuda", "dml", "cpu"], default=None, help="不传则自动探测")
    p.add_argument("--facts", type=Path, default=None, help="自定义编造事实 JSON 文件路径")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--repeat", type=int, default=15, help="每条事实在训练集里重复几次")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--out", type=Path, default=None, help="结果写入这个 JSON 文件")
    p.add_argument(
        "--directml-masked-fill-workaround", action="store_true",
        help="套用 microsoft/DirectML#702 的 torch.where 绕过法（见函数注释里的警告）",
    )
    args = p.parse_args()

    import torch

    if args.directml_masked_fill_workaround:
        apply_directml_masked_fill_workaround()

    backend = args.backend
    if backend is None:
        if torch.cuda.is_available():
            backend = "cuda"
        else:
            try:
                import torch_directml
                backend = "dml" if torch_directml.is_available() else "cpu"
            except ImportError:
                backend = "cpu"

    if backend == "dml":
        import torch_directml
        dev = torch_directml.device()
        print(f"backend: DirectML, device: {torch_directml.device_name(0)}")
        dtype = torch.float32
    elif backend == "cuda":
        dev = torch.device("cuda")
        print(f"backend: CUDA/HIP, device: {torch.cuda.get_device_name(0)}")
        dtype = torch.float16
    else:
        dev = torch.device("cpu")
        print("backend: CPU")
        dtype = torch.float32

    facts = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else DEFAULT_FACTS
    print(f"testing {len(facts)} invented facts, {args.steps} training steps\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)

    print("=" * 20, "BEFORE fine-tuning", "=" * 20)
    before = [generate(model, tokenizer, dev, f["question"], args.max_new_tokens) for f in facts]
    for f, a in zip(facts, before):
        print(f"Q: {f['question']}\nA: {a}\n")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    def to_text(f):
        messages = [{"role": "user", "content": f["question"]}, {"role": "assistant", "content": f["answer"]}]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    texts = [to_text(f) for f in facts] * args.repeat
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)
    labels = input_ids.clone()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n=== training {args.steps} steps on {len(texts)} examples ===")
    losses = []
    t0 = time.time()
    bs = 4
    for step in range(args.steps):
        i = (step * bs) % max(1, len(texts) - bs + 1)
        opt.zero_grad()
        out = model(input_ids=input_ids[i:i + bs], attention_mask=attention_mask[i:i + bs], labels=labels[i:i + bs])
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            print(f"step {step}: loss={loss.item():.4f}")
    print(f"training time: {time.time()-t0:.1f}s")

    if any(l != l for l in losses):  # NaN check -- l != l is True only for NaN
        print("\n*** WARNING: loss went NaN during training -- results below are meaningless ***")

    print("\n" + "=" * 20 + " AFTER fine-tuning " + "=" * 20)
    after = [generate(model, tokenizer, dev, f["question"], args.max_new_tokens) for f in facts]

    print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
    results = []
    correct = 0
    for f, b, a in zip(facts, before, after):
        kf = key_fact_of(f)
        learned = kf in a.lower()
        correct += learned
        print(f"Q: {f['question']}")
        print(f"  BEFORE: {b[:80]}")
        print(f"  AFTER:  {a[:80]}")
        print(f"  learned: {learned}")
        results.append({"question": f["question"], "expected": f["answer"], "before": b, "after": a, "learned": learned})

    print(f"\n{correct}/{len(facts)} invented facts correctly recalled after fine-tuning")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    if args.out:
        args.out.write_text(json.dumps({
            "model": args.model, "backend": backend, "steps": args.steps,
            "loss_start": losses[0], "loss_end": losses[-1],
            "nan_detected": any(l != l for l in losses),
            "correct": correct, "total": len(facts), "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresults written to {args.out}")

    sys.exit(0 if correct == len(facts) else 1)


if __name__ == "__main__":
    main()
