import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel


SPLIT_TO_FILE = {
    "train": "ScanRefer_filtered_train.json",
    "test": "ScanRefer_filtered_val.json",
    "val": "ScanRefer_filtered_val.json",
}


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return pooled / denom


def precompute_split(args, split):
    ann_file = os.path.join(args.scanrefer_root, SPLIT_TO_FILE[split])
    with open(ann_file, "r") as f:
        annotations = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, use_fast=True, local_files_only=True)
    text_encoder = RobertaModel.from_pretrained(
        args.text_model_name,
        add_pooling_layer=False,
        local_files_only=True,
    ).to(args.device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    features = []
    scene_ids = []
    object_ids = []
    ann_ids = []
    descriptions = []
    scene_to_indices = defaultdict(list)

    for start in tqdm(range(0, len(annotations), args.batch_size), desc=f"precompute {split}"):
        batch = annotations[start:start + args.batch_size]
        texts = [ann["description"] for ann in batch]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.text_max_len,
            return_tensors="pt",
        )
        tokenized = {key: value.to(args.device) for key, value in tokenized.items()}

        with torch.no_grad():
            outputs = text_encoder(**tokenized)
            pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])

        pooled = pooled.detach().cpu().to(torch.float16)
        features.append(pooled)

        for ann in batch:
            idx = len(scene_ids)
            scene_id = str(ann["scene_id"])
            scene_ids.append(scene_id)
            object_ids.append(int(ann["object_id"]))
            ann_ids.append(int(ann["ann_id"]))
            descriptions.append(ann["description"])
            scene_to_indices[scene_id].append(idx)

    cache = {
        "features": torch.cat(features, dim=0),
        "scene_ids": scene_ids,
        "object_ids": torch.tensor(object_ids, dtype=torch.long),
        "ann_ids": torch.tensor(ann_ids, dtype=torch.long),
        "descriptions": descriptions,
        "scene_to_indices": dict(scene_to_indices),
        "text_model_name": args.text_model_name,
        "text_max_len": args.text_max_len,
        "annotation_file": ann_file,
        "pooling": "attention_mask_mean",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"scanrefer_roberta_pooled_{split}.pt")
    torch.save(cache, out_path)
    print(f"Saved {len(scene_ids)} text features to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanrefer-root", default="data/ScanRefer")
    parser.add_argument("--text-model-name", default="./ckpts/roberta-base")
    parser.add_argument("--output-dir", default="data/text_feature_cache")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=sorted(SPLIT_TO_FILE))
    parser.add_argument("--text-max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for split in args.splits:
        precompute_split(args, split)


if __name__ == "__main__":
    main()
