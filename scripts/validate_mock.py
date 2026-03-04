#!/usr/bin/env python3
"""
DPO Mock 验证：完全离线，使用随机初始化小模型 + mock 数据验证流程
无需网络、无需下载模型或数据集
用法: python scripts/validate_mock.py
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class MockTokenizer:
    """最小 tokenizer，将字符映射为 id，用于离线验证"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._char_to_id = {}

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for c in text:
            if c not in self._char_to_id:
                self._char_to_id[c] = len(self._char_to_id) + 2  # 0=pad, 1=eos
                if len(self._char_to_id) >= self.vocab_size - 2:
                    break
            ids.append(self._char_to_id[c])
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def main():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from preference_datasets import get_dataset, get_batch_iterator

    print("=== DPO Mock 验证（完全离线）===\n")

    # 1. Mock 数据集
    print("1. 加载 mock 数据集...")
    data = get_dataset("mock", "train", silent=False)
    assert len(data) == 3
    print(f"   样本数: {len(data)}")

    # 2. 随机初始化小模型（不下载）
    print("2. 创建随机初始化 GPT2（无需下载）...")
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    tokenizer = MockTokenizer(vocab_size=1000)

    # 3. 数据迭代器
    print("3. 获取 batch iterator...")
    it = get_batch_iterator(
        names=["mock"],
        tokenizer=tokenizer,
        split="train",
        batch_size=2,
        shuffle=False,
        max_length=64,
        max_prompt_length=32,
        sft_mode=True,
        n_examples=4,
        seed=42,
        silent=True,
    )
    batch = next(it)
    print(f"   Batch keys: {list(batch.keys())}")
    chosen_ids = batch["chosen_input_ids"]
    print(f"   chosen_input_ids shape: {chosen_ids.shape}")

    # 4. 前向 + 损失
    print("4. 前向传播与 SFT 损失...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=chosen_ids).logits
        labels = batch["chosen_labels"].clone()
        labels[labels == -100] = 0
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
            ignore_index=0,
        )
    print(f"   Loss: {loss.item():.4f}")

    print("\nMock 验证 OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
