# 找回的 VLM 训练日志 / Recovered VLM training log

[中文实验出处](../experiment-evidence.md) · [English experiment sources](../experiment-evidence.en.md)

2026-10-13 从作者原来的本机实验目录找回 [vlm-sft-20260811-20260814.log](vlm-sft-20260811-20260814.log)，按原始字节复制，没有重新训练，也没有修改或清理日志。它不是一次新的实验。

The log was recovered from the author's old local experiment directory on 2026-10-13 and copied byte-for-byte. It was not edited, cleaned up or produced by a new training run.

| 项目 / Field | 值 / Value |
|---|---|
| 文件大小 / File size | 163,733 bytes |
| SHA-256 | `af63cb4c7cf16ecc4828991a828e528463b78aa999a88075ed7ebc8f55391dfe` |
| 行数 / Lines | 2,296 |
| 载入说明 / Load message | `loaded 97/97 keys from out\vlm_align_768.pth` |
| 可训练参数说明 / Trainable-parameter message | `20.85M (freeze_mode=1)` |
| 首条训练指标 / First training metric | `2026-08-11 20:54:07`, epoch 0, step `10/22691`, loss `2.7295` |
| 末条训练指标 / Last training metric | `2026-08-14 02:40:07`, epoch 0, step `22690/22691`, loss `2.1442` |
| 最终保存消息 / Final save message | `2026-08-14 02:40:20`, `saved vlm_sft_768.pth` |
| 首末训练指标时间跨度 / First-to-last training timestamp span | 53 小时 46 分钟 / 53 hours 46 minutes |

日志的 2,269 条训练指标在 step 和时间上严格递增，另有 22 条中途保存消息。末尾最终保存消息重复一次，并夹有 `1 loss 2.1442 lr 5.00e-07` 残行；这里原样保留。完整逐行检查只发现时间、epoch、step、loss、学习率、相对权重路径和载入/保存说明，没有数据集正文、图片或凭据。仓库的 `.gitattributes` 对此文件关闭换行转换。

The 2,269 training metric rows have strictly increasing steps and timestamps, with 22 periodic-save messages. The final save message appears twice with a partial line, `1 loss 2.1442 lr 5.00e-07`, between them; these are preserved. A full line-by-line check found only timestamps, epoch/step/loss/learning-rate values, relative weight paths and load/save messages, with no dataset text, images or credentials. `.gitattributes` disables newline conversion for this file.

它支持“日志最后记录了什么”，不提供 checkpoint 内容或哈希，也不包含完整命令、数据/tokenizer 指纹、硬件或源码版本。`97/97` 和 `20.85M` 是程序打印的说明，不能替代模型身份校验。53 小时 46 分钟是日志时间跨度，**不是 GPU 利用时间**；日志没有时区标记，旧 README 的“约 30 小时”也不能由这份日志支持。最后的训练指标是 `22690/22691`，不应写成逐步核验了全部 22,691 步。保存消息同样不证明权重今天可取得或内容完整。

This supports the final logged state, not checkpoint identity or availability. It contains no checkpoint hash, complete command, data/tokenizer fingerprints, hardware description or source revision. `97/97` and `20.85M` are program messages, not independent identity checks. The timestamp span is **not GPU utilization time**, and the log does not specify a timezone. It does not support the old “about 30 hours” estimate. The last metric is `22690/22691`, not a per-step verification of all 22,691 planned steps. A save message does not establish present checkpoint availability or integrity.

此文件没有图文评测输出；不能据它判断看图可靠性，也没有补齐文本侧 23M / 64M 的重建准确率证据。

There are no image-text evaluation outputs in this file. It does not establish grounding quality or fill the missing evidence for the 23M / 64M text-reconstruction results.
