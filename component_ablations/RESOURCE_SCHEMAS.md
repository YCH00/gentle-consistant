# 正式诊断资源接口

这里定义本轮消费者所需的清单接口。当前仅有旧实验 smoke 产物；不能通过修改 `kind` 把 smoke 升级为正式资源。正式 bank 的实际生成器/旧版清单如果采用不同字段，需要先完成经审计的格式适配，再绑定，不能自行补造 context、query 或来源标签。

## 训练资源

`resources.json` 按环境保存：固定 task manifest、native training normalizer、逐轨迹数据路径/sha256/rows/task_id/split，以及四种子 encoder/decoder/十个 task dynamics 文件。训练数据按历史 loader 顺序排列。Normalizer NPZ 包含 `mean`, `var`, `count`；加载后用相同、哈希已验证的训练文件重算单次 RMS 更新作一致性校验，不替换冻结统计量、不使用评估数据。

任务清单保留 `train_tasks`, `eval_tasks`, `tasks`, `horizon`。实例化原生环境后，逐任务比较真实参数，再绑定冻结任务。MuJoCo 库及 XML 哈希须匹配历史 backend 记录；Cheetah 150，Walker 131。

## 固定 context bank

正式 JSON 清单必需字段：

```json
{
  "kind": "formal",
  "complete": true,
  "protocol": "shift-diagnostics-v3",
  "environment": "cheetah-vel",
  "task_manifest_sha256": "原始共享 task_manifest 文件的真实 SHA256",
  "collector": {"method": "GENTLE"},
  "observation_space": "raw",
  "action_space": "normalized",
  "arrays": {"path": "contexts.npz", "sha256": "真实 SHA256"},
  "contexts": []
}
```

以上只展示结构，不是可运行的正式清单。`contexts` 必须包括全部 20 个任务 × 三条件 × 20 个 repeats，不能只包含示例项。每项包含：`context_id`（全局唯一）、`task_id`（0..19）、`task_split`（seen/unseen）、`condition_id`、`context_repeat`（0..19）、`context_length`（200）、`array_prefix`。支持清单中其他已冻结条件，但本轮只选三条件。

NPZ 每个 prefix 对应 `observations`, `actions`, `rewards`, `next_observations`；第一维 200，reward 为 `(200,1)`。默认 prefix 为 `c<清单索引>_`，可显式填写其他 prefix。任务 0..9 seen、10..19 unseen，与当前冻结清单一致。原始来源行、轨迹边界、采集器版本、全部 reset/noise 种子和文件哈希仍需由正式 bank 发布者保存并验证。

本轮不改写这个共享 bank。读入后，对 obs/next_obs 仅施加一次本模型已冻结的训练 normalizer。

## Probe manifest

顶层字段：`kind="formal_probe"`, `complete=true`, `environment`, `task_manifest_sha256`, `arrays={path,sha256}`, `groups`。

`groups` 共 40 项：seen 0..9 各 train/val/test，unseen 10..19 仅 test。每项：

- `task_id`, `split`（train/val/test）；
- `query_prefix`, `context_prefix`；
- `query_trajectory_ids`：1000 个来源轨迹 ID，允许同一条轨迹贡献多个 query；
- `context_trajectory_ids`：20 项，每项为该 context 的来源轨迹 ID，或多个来源轨迹 ID 的列表；不同 context 的轨迹集合互不相交。

同一任务所有分区的 context/query 轨迹集合必须互不相交。轨迹 ID 必须来自真实数据 provenance，不能为通过检查重新编号。

对应 NPZ：

- `<query_prefix>observations/actions/rewards/next_observations`：1000 条原始查询；
- `<query_prefix>conditions`：1000 个字符串，恰好 500 条 `policy499_sigma000` 与 500 条 `policy499_sigma020`；
- `<context_prefix>observations/actions/rewards/next_observations`：形状 `(20,200,feature_dim)`。

Query 和 context 的 obs/next_obs/reward 使用 raw 坐标，actions 为 normalized [-1,1]。Test context 应复用正式共享 bank 中对应的 policy499_sigma000 条件；train/val context 来自各自独立的 seen 轨迹分区。清单发布者需要核验这层来源关系，消费者还会检查覆盖、形状、查询条件比例和轨迹隔离。

训练目标为原始 reward 与 delta_state；标准化统计只拟合 seen train queries。Latent 标准化只拟合各 arm 的 seen train contexts。所有查询在四种子与各 arm 之间共享，模型选择仅依 seen-validation。
