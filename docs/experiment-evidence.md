# 实验出处与复现边界

中文 · [English](experiment-evidence.en.md) · [回到 README](../README.md)

核对日期：2026-10-13。代码基线：[0a4e5d3](https://github.com/adenzhou1350/minimind-diffusion/tree/0a4e5d3cf5ff3a78381d126ce131702352a79b1b)。本页区分能从代码确认的设置、历史文字记录，以及目前没有公开的产物。文档核对没有重跑训练。

## 先从能直接检查的部分开始

- [模型测试](../tests/test_model.py)和 [loss 测试](../tests/test_loss.py)：随机初始化的小模型，不依赖下载的 tokenizer、数据或权重。安装依赖与命令见 [README](../README.md#先跑一个不用下载模型的例子)。
- [采样测试](../tests/test_sampling.py)：检查长度、mask 清除和采样分支；通过不代表语言质量好。
- [训练入口](../trainer/train_pretrain.py)、[数据说明](../dataset/dataset.md)：准备外部资产后才能训练；仓库没有训练权重可供直接体验对话。

## 数字的出处

| 历史记录 | 固定出处 | 缺少什么 |
|---|---|---|
| 23M 筛选，约 5% 与 25.5% 重建准确率，后者为 3,000 步 | [loss 修改提交说明 `fb96b34`](https://github.com/adenzhou1350/minimind-diffusion/commit/fb96b344f5e58b7f9074de72e19473afdee67927) | 两组原始输出、完整运行配置和资产指纹 |
| 旧实现 0–5%；后续标为 64M 的结果 41.5%，loss 7.4→1.7；1→2 epoch 约 41%→44% | [当时的 README `c080996`](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md) | 对应训练日志、checkpoint、评测脚本与配置的明确对应关系 |
| VLM Stage 2：22,691 步、约 30 小时、最终 loss 2.14；换图后仍出现相似幻觉 | [当时的 README `24a64b7`](https://github.com/adenzhou1350/minimind-diffusion/blob/24a64b72f876fcc5e3aad2c7ffe0b08277f6914c/README.md) | 训练日志、逐图预测、图片来源与固定对照集 |

这些链接证明的是“当时写下了什么”。核对了公开 master 的完整可取得历史，未找到对应原始日志、逐样本结果或 checkpoint；不能用一段 README 替代实验产物。23M 筛选与后续 64M 记录也不能拼成同配置的受控对比。

## 筛选脚本实际做了什么

以下设置读取自基线版本中的三个脚本。`256×4` 表示隐藏维度 256、4 层，不是一个经核对的参数总量标签；旧提交中的“23M”仅按原文字保留。

| 脚本 | 模型和训练 | 数据行范围（从 0 开始，右端不含） | 评测 |
|---|---|---|---|
| [screen_loss.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_loss.py) | 256×4，batch 16，每组 500 步，lr 5e-4，4 组 loss / 掩码范围配置 | 训练 `[0, 4000)`；held-out `[4000, 4050)` | 随机遮住约 30% 非 padding token |
| [screen_scale.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_scale.py) | 512×6：3,000 步，lr 1e-3 / 3e-4；256×4：5,000 步，lr 1e-3；batch 64 | 训练 `[0, 20000)`；held-out `[20000, 20200)` | 随机遮住约 20% 非 padding token |
| [screen_init.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_init.py) | 256×4，batch 64，每组 800 步，lr 1e-3；random / mean / zero 初始化 | 训练 `[0, 8000)`；held-out `[8000, 8100)` | 随机遮住约 20% 非 padding token |

三个脚本都使用 `dataset/pretrain_t2t_mini.jsonl`、`model/` 中的 tokenizer，序列长度 128，并调用 `torch.manual_seed(0)`。设备写死为 `cuda`；它们是训练筛选脚本，**不是 CPU 快速测试**，也没有命令行参数解析。

`screen_loss.py` 的四组为：

1. 额外 `1/t` 权重，`t ~ U(1e-4, 1.0)`。
2. 均匀权重，`t ~ U(1e-4, 1.0)`。
3. 额外 `1/t` 权重，`t ~ U(0.1, 0.7)`。
4. 均匀权重，`t ~ U(0.1, 0.5)`。

各组先对每条序列的被掩 token 求平均 CE，再决定是否乘 `1/t`。因此比较的是这个仓库的具体实现；不能直接得出官方 LLaDA 方法失败的结论。当前主训练代码保留第 4 种形式。

重建准确率 = 被掩位置上 argmax 预测正确的 token 数 / 被掩 token 总数。它不是问答准确率。评测每次重新采样 mask，并可能包含 BOS/EOS；没有发布固定 mask、数据哈希或重复种子区间。几个脚本的 held-out 范围和掩码概率不同，结果不应横向混排。

## 下一次运行怎样留下可核对的结果

先准备依赖、tokenizer、语料和可用 CUDA 设备。若选择运行筛选，下面是 PowerShell 保存输出的例子；它会执行真实训练，不是用来还原上表旧数字的承诺。

```powershell
New-Item -ItemType Directory -Force out/evidence
git rev-parse HEAD > out/evidence/source.txt
python -m pip freeze > out/evidence/packages.txt
Get-FileHash model/tokenizer.json, model/tokenizer_config.json, dataset/pretrain_t2t_mini.jsonl -Algorithm SHA256 |
    Format-List | Out-File out/evidence/assets.txt
python -u screen_loss.py > out/evidence/screen-loss.log 2>&1
```

`out/` 被 gitignore；保存完成不等于已经发布。一次结果至少应一起提供：源码版本及本地改动、实际命令、硬件与软件版本、数据/tokenizer 指纹、训练设置、评测划分和 mask/随机种子、完整输出。需要比较配方时，再把配置固定，只改变待研究变量，并保存重复运行结果。

如果找回旧日志，先核对它对应哪组配置，再补到上述出处表；如果重新运行，应标成新的实验日期与版本，不追填成历史结果。
