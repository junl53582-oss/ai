# Legacy Model Artifacts Directory

本目录存放 GEN1 ~ GEN4 阶段历史过渡制品与向后兼容模型文件。

## 历史说明与防混淆提示
1. latest_lightgbm.pkl: 
   - 命名历史残留，其实体对象为 DRLStrengthenedQuantModel (Gen 4 强化学习模型)。
   - 仅供历史旧脚本与回归测试调用，已全面标记为 LEGACY。
2. 规范生产模型路径:
   - 生产模型标准路径为 saved_models/production/<model_id>/。
   - 所有生产推理统一经由 ModelRegistry 与 BatchInference 加载，严禁直接依赖根目录下的硬编码文件名。
