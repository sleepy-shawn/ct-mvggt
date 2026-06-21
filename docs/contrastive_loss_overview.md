# MVGGT Contrastive Loss Overview

这份说明整理当前代码中与对比学习相关的实现，重点对应当前主实验：
`patch_anchor_scene_object_text`。该版本不再使用 mask-ratio threshold 来判定正负样本，而是使用同一场景内的
`object_id`：同一个 object 的文本为正样本，不同 object 的文本为负样本。

## 当前实验设置

- 训练视角数：4 views/sample，`train.image_num_range=[4,4]`
- 验证视角数：4 views/sample，`test.image_num_range=[4,4]`
- 采样策略：原版随机采样，`same_scene_in_batch=false`
- 文本编码器：当前实验命令中设置 `model.freeze_text_encoder=true`
- 预编码文本：使用 scene-level text feature cache
- 对比模式：`loss.train_loss.contrastive_mode=patch_anchor_scene_object_text`
- 对比损失权重：当前 sweep 使用 `loss_contrastive=lambda`，例如 0.2 / 0.4 / 0.5

配置入口：

- `configs/train/train_mvggt_refer_lowres.yaml`
- `configs/model/mvggt.yaml`

注意：`train_mvggt_refer_lowres.yaml` 里默认保留了旧的
`patch_anchor_text_mask_ratio` 配置，但当前主实验通过命令行 override 切换到
`patch_anchor_scene_object_text`。

## 数据流

### 1. 数据集提供当前文本和同场景文本库

`datasets/scannet_dataset.py`

- `_load_scene_text_feature_cache` 读取预先缓存好的 RoBERTa text features。
- cache 中包含：
  - `features`: 全部文本特征
  - `object_ids`: 每条文本对应的 object id
  - `ann_ids`: annotation id
  - `scene_to_indices`: 每个 scene 对应哪些文本
- 每个 sample 在 `__getitem__` 中返回当前描述，同时把该 scene 的所有 text features 和 object ids 放进
  `text_info`。

关键代码：

- `datasets/scannet_dataset.py:200`
- `datasets/scannet_dataset.py:223`
- `datasets/scannet_dataset.py:328`

### 2. Trainer 把 scene text features 传进模型

`trainers/mvggt_trainer.py`

- `forward_batch` 从 `batched_text` 取出 `scene_text_features`
- 调用模型时传入 `scene_text_features`
- `calculate_loss` 把整个 `batched_text` 作为 `text_info` 传给 loss

关键代码：

- `trainers/mvggt_trainer.py:79`
- `trainers/mvggt_trainer.py:100`

### 3. 模型输出视觉 patch feature 和 projected text feature

`mvggt/models/mvggt_training.py`

- 文本编码器输出当前描述的 token features，用于原始 referring segmentation 分支。
- 当前 scene 的缓存 text features 不再重新过 RoBERTa，而是经过当前模型的 `text_proj` 投影到 decoder 维度。
- 模型输出：
  - `contrastive_patch_features`: 视觉 patch features，shape 约为 `[B, V, P, C]`
  - `contrastive_patch_shape`: patch grid 的 `(patch_h, patch_w)`
  - `contrastive_scene_text_features`: 当前 scene 所有文本的 projected features

关键代码：

- `mvggt/models/mvggt_training.py:283`
- `mvggt/models/mvggt_training.py:523`
- `mvggt/models/mvggt_training.py:585`

## Contrastive loss 计算

核心实现在 `mvggt/models/loss.py`。

### 1. Visual anchor：用 gt mask 聚合 patch features

`_pool_anchor_patch_features`：

1. 把 `gt_masks` 从 image resolution 用 area interpolation 下采样到 patch grid。
2. 用下采样后的 mask weight 对 4 个 view 的 patch features 做加权平均。
3. 如果当前 sample 的 4 个 view 都没有目标 mask，`weight_sum=0`，该 anchor 视为 invalid，不参与 contrastive loss。

公式：

```text
anchor = sum_{view,patch}(mask_weight * patch_feature) / sum(mask_weight)
anchor = normalize(anchor)
```

关键代码：

- `mvggt/models/loss.py:423`

### 2. Text side：同场景所有 text 做候选

`_patch_anchor_scene_object_text_loss`：

1. 对当前 batch 中每个 anchor，取出该 sample 所属 scene 的所有 cached text features。
2. 经过 `text_proj` 后做 L2 normalize。
3. 与 visual anchor 计算相似度：

```text
logits = normalize(anchor) @ normalize(text_features).T / temperature
```

当前 temperature 默认是 `0.1`。

关键代码：

- `mvggt/models/loss.py:560`
- `mvggt/models/loss.py:615`

### 3. 正负样本定义

当前主实验中：

```text
positive texts: text_object_id == current_sample_object_id
negative texts: text_object_id != current_sample_object_id
```

也就是说，正样本是“同一个 scene 内描述同一个物体的所有文本”；负样本是“同一个 scene 内描述其他物体的文本”。

关键代码：

- `mvggt/models/loss.py:620`

### 4. Loss 形式

当前实现是 multi-positive pairwise ranking loss：

```text
loss(anchor) = log(1 + sum_{neg,pos} exp(logit_neg - logit_pos))
final loss = mean over valid anchors
```

直觉上，它推动每个 positive text 的 logit 高于每个 negative text 的 logit。

关键代码：

- `mvggt/models/loss.py:629`
- `mvggt/models/loss.py:649`

### 5. 总 loss 中的权重

总 referring loss 中：

```text
total = loss_mask + loss_dice + lambda * loss_contrastive
```

关键代码：

- `mvggt/models/loss.py:674`
- `mvggt/models/loss.py:692`

## 当前会记录的诊断量

旧运行已经记录：

- `refer_loss_contrastive`: 未乘 lambda 的原始 contrastive loss
- `refer_contrastive_valid_anchor_rate`: 有效 anchor 占比
- `refer_contrastive_mean_pos_count`: 每个有效 anchor 平均正样本文本数
- `refer_contrastive_mean_neg_count`: 每个有效 anchor 平均负样本文本数
- `refer_contrastive_mean_scene_text_count`: 每个 scene 平均候选 text 数

新增但只有重启后的实验才会记录：

- `refer_contrastive_mean_pos_rank`: 正样本文本在同 scene 全部文本中的平均排名，越接近 1 越好
- `refer_contrastive_pos_neg_pair_accuracy`: 所有正负 pair 中 `logit_pos > logit_neg` 的比例，越接近 1 越好

新增指标代码：

- `mvggt/models/loss.py:627`
- `mvggt/models/loss.py:655`

## 与旧 threshold 版本的区别

旧版本 `patch_anchor_text_mask_ratio`：

- 候选 text 只来自当前 batch 中同 scene 的样本。
- 用目标物体在 anchor sample 的 instance map 中的出现比例判定正负样本。
- 需要设置 `contrastive_mask_ratio_threshold`。

当前主实验 `patch_anchor_scene_object_text`：

- 候选 text 来自整个 scene 的缓存文本库，不依赖 batch 里是否采到同 scene。
- 正负样本直接由 object id 判断，不使用阈值。
- 更符合“当前 visual anchor 对应的物体，应拉近同物体文本、推远其他物体文本”的定义。

旧版本代码仍保留在：

- `mvggt/models/loss.py:466`
- `mvggt/models/loss.py:489`

当前主实验代码在：

- `mvggt/models/loss.py:560`

## 需要向导师说明的限制

1. Visual anchor 目前使用的是 gt mask 加权池化，不是 predicted mask。
   这样对比学习信号更干净，但训练和推理存在一定差异。

2. 如果一个 sample 的 4 个 view 都没有目标 gt mask，该 sample 的 contrastive anchor invalid，会跳过 contrastive loss。
   这会体现在 `refer_contrastive_valid_anchor_rate` 中。

3. 文本编码器冻结时，scene text cache 不重新编码；但 cached features 仍会经过当前模型的 `text_proj`。
   因此 contrastive loss 仍会优化视觉分支、融合/decoder 相关参数，以及 `text_proj`。

4. 当前正在跑的旧进程不会自动记录新增 rank/pair accuracy，因为 Python 进程已经加载了旧代码。
