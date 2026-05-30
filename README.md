# NAI4 Style Multi-Subject Multi-Prompt Extension for SD Forge

NovelAI 4.5 风格的多主体多提示词插件，提供独立标签页界面和区域融合功能。

## 功能特性

- ✅ **独立标签页界面** - 完全独立的 NAI 风格创作界面
- ✅ **多主体标签选择器** - 按类型分类的快速标签选择
- ✅ **自动人数统计** - 自动生成 1girl/2girls/1boy 等计数标签
- ✅ **三种融合模式** - 简单 AND / BREAK 注意力 / 区域融合
- ✅ **区域融合** - 按区域横向分割画面，每个区域对应不同角色
- ✅ **LoRA 集成** - 支持最多 4 个 LoRA，可单独开关和设置权重
- ✅ **角色视觉区分** - 每个角色卡片有不同颜色边框，防止混淆

## 安装

在 Stable Diffusion WebUI Forge 中：

1. 进入 **Extensions → Install from URL**
2. 粘贴仓库 URL：
   ```
   https://github.com/tiengalaxy/NAI4.5-Style-Multi-Character-Prompt-for-SD-Forge
   ```
3. 点击 Install
4. 重启 WebUI

## 使用说明

### 1. 创建角色

在左侧 **Characters** 面板：
- 启用角色
- 选择性别（用于自动人数统计）
- 填写提示词或使用标签选择器
- 设置权重（可选）

### 2. 全局设置

在 **Global** 标签：
- 全局风格/环境提示词
- 全局负面提示词

### 3. 生成设置

在 **Generation** 面板：
- 设置尺寸、步数、CFG、采样器、种子
- 选择融合模式

### 4. 区域融合模式

选择 **Regional Blend (Horizontal)**：
- 区域比例：用逗号分隔，例如 `1,1,1` 三等分
- Base Ratio：全局 prompt 对所有区域的影响程度
- Feather Width：区域边缘融合宽度
- Calculation Mode：
  - **Attention**：通过注意力机制实现区域隔离（推荐）
  - **Latent**：直接在潜在空间加权混合

## 仓库结构

```
NAI4.5-Style-Multi-Character-Prompt-for-SD-Forge/
├── install.py          # 依赖安装脚本
├── scripts/
│   └── multi_subject.py # 主插件文件
└── javascript/
    └── nai_multi_subject.js # 前端增强
```

## 致谢

基于 NovelAI 4.5 的用户界面设计思路开发。
