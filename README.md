# TorchScope

面向中文初学者的 PyTorch 可视化学习图谱：先用 15 个重点实验建立直觉，再按需查询 9,066 个官方 API 条目。

## 适合谁

- 会一点 Python，知道列表、函数和基础代数，但还不熟悉张量与神经网络
- 希望先“看见输入怎样变成输出”，再阅读公式和官方文档
- 需要按名称、用途、模块或相似方法快速查询 PyTorch API

不需要先掌握深度学习。建议第一次使用时从 `torch.tensor` 开始，不必先浏览完整 API 目录。

## 最快上手

需要 Node.js 22.13 或更高版本。

```bash
npm install
npm run dev
```

打开终端显示的本地地址（通常是 `http://localhost:3000`），点击首页的“从第 1 课开始”。推荐顺序是：

1. Tensor：理解数据、shape 和 dtype
2. 形状：练习 `reshape`、`unsqueeze` 和 `cat`
3. 计算：理解 `matmul`、`mean` 和维度归约
4. 网络：串起激活函数、`Linear` 与 `Conv2d`
5. 训练：把 loss、`backward()` 和 `optimizer.step()` 放进一个最小训练循环

学习进度保存在当前浏览器本地，无需登录。

## 三个入口

### 1. 初学者起点

首页先解释学习前提、五步路线、八个常见术语，并给出一段最小完整训练循环。它回答的是“我该先学什么、这些词是什么意思、零散 API 最后怎样连起来”。

### 2. 15 个重点实验

每课包含中文白话解释、调用方法、公式变量、可运行代码和可编辑 Tensor 输入。课程支持上一课/下一课导航与本地学习进度。

### 3. 完整 API 图谱

- 从 PyTorch 2.13 stable 官方文档 inventory 生成 9,066 个条目，覆盖 21 个一级模块
- 支持函数、类、方法和其他接口分栏，以及一级模块、二级分类和关键词筛选
- 提供相似方法对比、官方签名、公式符号解释、场景示例和输入输出示意
- 高频数值算子支持逐步 Tensor Flow；卷积支持 1D/2D/3D 滑窗与逐位置计算
- 在页面非输入区域按 `/` 可以直接聚焦 API 搜索框

## 一段真实的 PyTorch 代码

网页里的知识最终要落到 Python 代码中。一个训练步骤的核心顺序如下：

```python
import torch
from torch import nn

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()
prediction = model(x)
loss = loss_fn(prediction, y)
loss.backward()
optimizer.step()
```

安装真实 PyTorch 时，请按操作系统和 CPU/CUDA 环境使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。

## 教学模拟边界

TorchScope 的实验台在浏览器中模拟所展示函数的数值行为，用来验证 shape、维度、广播和中间计算，不会在浏览器里启动真实 PyTorch。

- 生产训练、性能测试、自动微分和硬件行为应以正式 PyTorch 运行时为准
- 完整 API 区的冷门对象、硬件、分布式和配置接口可能展示处理步骤或状态示意，而非真实执行
- API 签名与精确定义应通过页面里的官方文档链接复核

## 开发与验证

```bash
npm run lint
npm run build
npm test
```

主要代码：

- `app/page.tsx`：重点课程和交互实验台
- `app/beginner-start.tsx`：初学者路线、术语与训练循环
- `app/full-api-browser.tsx`：完整 API 图谱和计算可视化
- `app/api-index.generated.json`：生成后的官方 API 索引
- `tests/rendered-html.test.mjs`：内容与构建产物检查

## 更新官方文档索引

下载目标 PyTorch 文档版本的 `objects.inv` 到项目根目录，然后运行：

```bash
npm run docs:generate
```

## 贡献建议

优先提交能让初学者更快形成正确直觉的改进，例如：更小的可运行例子、更清楚的 shape 标注、更具体的错误提示、真实训练代码与模拟结果的对照。提交前请运行 lint、build 和 test。
