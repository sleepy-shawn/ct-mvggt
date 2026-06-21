# Contrastive Loss Code Summary

本文档浓缩说明当前 MVGGT 对比损失相关代码。当前主实验使用：

```text
contrastive_mode = patch_anchor_scene_object_text
```

核心思想：

```text
visual anchor = 当前指代物体的 GT mask 加权聚合多 view patch feature
positive text = 同 scene 中同 object_id 的文本
negative text = 同 scene 中不同 object_id 的文本
loss = log(1 + sum exp(logit_neg - logit_pos))
```

## 1. 数据集提供 scene-level text bank

文件：`datasets/scannet_dataset.py:200-236`

```python
def _load_scene_text_feature_cache(self, cache_path):
    # 读取提前编码好的 scene-level text feature cache
    cache = torch.load(cache_path, map_location='cpu')

    self.scene_text_bank = {
        'features': cache['features'].float(),      # 所有文本的 RoBERTa pooled feature
        'object_ids': cache['object_ids'].long(),   # 每条文本对应的 object_id
        'ann_ids': cache['ann_ids'].long(),
        'scene_to_indices': cache['scene_to_indices'],  # scene_id -> text indices
    }

def _scene_text_info(self, scene_id):
    # 取出当前 scene 的所有 text features 和对应 object_id
    indices = self.scene_text_bank['scene_to_indices'].get(scene_id, [])
    if len(indices) == 0:
        return None

    indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        'scene_text_features': self.scene_text_bank['features'].index_select(0, indices),
        'scene_text_object_ids': self.scene_text_bank['object_ids'].index_select(0, indices),
        'scene_text_ann_ids': self.scene_text_bank['ann_ids'].index_select(0, indices),
    }
```

文件：`datasets/scannet_dataset.py:328-340`

```python
text_info = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'description': description,
    'scene_id': sample_info['scene_id'],
    'object_id': object_id,  # 当前描述指代的物体
    'ann_id': ann_id,
}

# 每个 sample 额外带上当前 scene 的所有 text bank
scene_text_info = self._scene_text_info(sample_info['scene_id'])
if scene_text_info is not None:
    text_info.update(scene_text_info)
```

## 2. Trainer 传递 text bank

文件：`trainers/mvggt_trainer.py:79-108`

```python
def forward_batch(self, batch, mode='train'):
    batched_views, batched_text = batch

    imgs = torch.stack([view['img'] for view in batched_views], dim=1)
    input_ids = batched_text['input_ids']
    attention_mask = batched_text['attention_mask']

    # 当前 scene 的预编码 text features
    scene_text_features = batched_text.get('scene_text_features')

    pred = self.model(
        imgs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        scene_text_features=scene_text_features,
    )

    return [pred, batch]

def calculate_loss(self, output, batch, mode='train', current_epoch=None, total_epochs=None):
    output, batch = output
    batched_views, batched_text = batch

    # loss 需要 batched_text 里的 object_id / scene_text_object_ids
    loss, details = self.train_loss(output, batched_views, text_info=batched_text)
    return EasyDict(loss=loss, **details)
```

## 3. 模型输出 patch feature 和 projected text feature

文件：`mvggt/models/mvggt_training.py:282-291`

```python
self.freeze_text_encoder = freeze_text_encoder

if self.use_referring_segmentation:
    self.text_encoder = RobertaModel.from_pretrained(text_model_name, add_pooling_layer=False)
    roberta_dim = self.text_encoder.config.hidden_size
    self.text_proj = nn.Linear(roberta_dim, self.dec_embed_dim)

    # 当前实验通常冻结 RoBERTa，但 text_proj 仍训练
    if self.freeze_text_encoder:
        freeze_all_params([self.text_encoder])
```

文件：`mvggt/models/mvggt_training.py:504-511`

```python
if self.use_referring_segmentation:
    # 从 multimodal hidden 中取 image patch tokens，作为 contrastive visual feature
    contrastive_patch_features = multimodal_hidden.reshape(B, N, hw, -1)[:, :, self.patch_start_idx:]
```

文件：`mvggt/models/mvggt_training.py:523-540`

```python
def _project_scene_text_features(self, scene_text_features, device):
    # cached text features 不重新过 RoBERTa，只过 text_proj
    projected = []
    for features in scene_text_features:
        if features is None:
            projected.append(None)
            continue
        features = features.to(device=device, dtype=self.text_proj.weight.dtype)
        projected.append(self.text_proj(features).float())
    return projected
```

文件：`mvggt/models/mvggt_training.py:585-595`

```python
output['contrastive_patch_features'] = contrastive_patch_features
output['contrastive_patch_shape'] = (patch_h, patch_w)
output['contrastive_scene_text_features'] = self._project_scene_text_features(
    scene_text_features,
    imgs.device,
)
```

## 4. Visual anchor: GT mask 加权聚合 patch feature

文件：`mvggt/models/loss.py:423-437`

```python
def _pool_anchor_patch_features(self, patch_features, gt_masks, patch_shape):
    B, V, _, _ = gt_masks.shape
    patch_h, patch_w = patch_shape

    # GT mask 下采样到 patch grid
    mask_weights = F.interpolate(
        gt_masks.float().reshape(B * V, 1, *gt_masks.shape[-2:]),
        size=(patch_h, patch_w),
        mode='area',
    ).reshape(B, V, patch_h * patch_w)

    # 对多 view、多 patch 按 mask 占比加权平均
    patch_features = patch_features.float()
    weight_sum = mask_weights.sum(dim=(1, 2))
    pooled = (patch_features * mask_weights.unsqueeze(-1)).sum(dim=(1, 2))
    pooled = pooled / weight_sum.clamp_min(1e-6).unsqueeze(-1)

    # 如果当前采样 view 中完全没有目标 mask，则该 anchor 不参与对比损失
    valid = weight_sum > 1e-6
    return pooled, valid
```

## 5. 当前主对比损失

文件：`mvggt/models/loss.py:560-657`

```python
def _patch_anchor_scene_object_text_loss(self, pred, gt_masks, text_info):
    patch_features = pred['contrastive_patch_features']
    patch_shape = pred['contrastive_patch_shape']
    scene_text_features = pred['contrastive_scene_text_features']

    # visual anchor 来自当前指代物体的 GT mask 区域
    anchor_features, valid_anchor = self._pool_anchor_patch_features(
        patch_features,
        gt_masks,
        patch_shape,
    )
    anchor_features = F.normalize(anchor_features, dim=-1)

    for anchor_idx in range(anchor_features.shape[0]):
        if not bool(valid_anchor[anchor_idx].item()):
            continue

        # 当前 scene 的所有 text features 和 object ids
        text_features = self._ragged_item(scene_text_features, anchor_idx)
        text_object_ids = self._ragged_item(scene_text_object_ids, anchor_idx)
        target_object_id = int(object_ids[anchor_idx])

        text_features = F.normalize(text_features.float(), dim=-1)

        # visual anchor vs scene text bank
        logits = anchor_features[anchor_idx].float() @ text_features.T
        logits = logits / self.contrastive_temperature

        # 正样本：同 object 的文本；负样本：同 scene 其他 object 的文本
        pos_mask = text_object_ids == target_object_id
        neg_mask = text_object_ids != target_object_id
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue

        # 诊断指标
        pos_rank = 1.0 + (logits[None, :] > pos_logits[:, None]).float().sum(dim=1)
        pos_neg_pair_accuracy = (pos_logits[:, None] > neg_logits[None, :]).float().mean()

        # pairwise ranking loss: log(1 + sum exp(neg - pos))
        diff = neg_logits[:, None] - pos_logits[None, :]
        diff = diff.reshape(-1)
        diff = torch.cat([diff, logits.new_zeros(1)], dim=0)
        losses.append(torch.logsumexp(diff, dim=0))
```

## 6. 总 loss

文件：`mvggt/models/loss.py:659-694`

```python
total_loss = self.weight_dict['loss_mask'] * losses['loss_mask']
total_loss += self.weight_dict['loss_dice'] * losses['loss_dice']

contrastive_weight = self.weight_dict.get('loss_contrastive', 0.0)

if contrastive_weight > 0:
    if self.contrastive_mode == "patch_anchor_scene_object_text":
        contrastive_loss, contrastive_details = self._patch_anchor_scene_object_text_loss(
            pred,
            gt_masks,
            text_info,
        )

    losses["loss_contrastive"] = contrastive_loss
    losses.update(contrastive_details)
    total_loss += contrastive_weight * contrastive_loss
```

## 7. 预编码 text cache

文件：`scripts/precompute_scene_text_features.py:18-91`

```python
def mean_pool(last_hidden_state, attention_mask):
    # RoBERTa token features -> sentence feature
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return pooled / denom

def precompute_split(args, split):
    # 对 ScanRefer train/val 文本提前编码，保存 scene -> text features
    outputs = text_encoder(**tokenized)
    pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])

    cache = {
        "features": torch.cat(features, dim=0),
        "object_ids": torch.tensor(object_ids, dtype=torch.long),
        "ann_ids": torch.tensor(ann_ids, dtype=torch.long),
        "scene_to_indices": dict(scene_to_indices),
    }

    torch.save(cache, out_path)
```

## 8. 当前实现要点

- visual anchor 使用 `gt_mask`，不是 predicted mask。
- 文本编码器冻结时，RoBERTa 不更新，但 `text_proj` 仍训练。
- 对比发生在 scene-level text bank 上，不依赖 batch 内是否采到同 scene。
- 当前主版本不使用 threshold；正负样本由 `object_id` 决定。
- 如果采样 view 中完全没有目标 GT mask，该 sample 的 contrastive anchor invalid，会跳过。

一句话总结：

```text
当前对比学习用 GT mask 从多 view patch feature 中聚合当前物体 visual anchor，
将其与同 scene 的所有文本做相似度；
同 object_id 文本为正样本，不同 object_id 文本为负样本；
通过 log(1 + sum exp(logit_neg - logit_pos)) 让同物体文本排到其他物体文本前面。
```
