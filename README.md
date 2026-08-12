# TorchScope

一个面向中文学习者的 PyTorch 可视化函数图谱。它把常用 API 的概念、调用方法、公式、变量含义、最小代码示例和可交互 Tensor 实验放在同一个界面中。

## 功能

- 15 个核心 PyTorch 函数，覆盖张量、形状、数学运算、神经网络和损失函数
- 中文概念解释、公式与逐变量解析
- 可编辑 JSON Tensor 输入，实时查看输出和 shape 变化
- 函数搜索、分类筛选和本地学习进度
- 响应式布局，支持桌面和移动设备

## 本地运行

```bash
npm install
npm run dev
```

打开 `http://localhost:3000`。

## 构建

```bash
npm run build
```

> 实验台在浏览器中模拟所展示函数的数值行为，用于教学与直觉验证；生产训练代码仍应使用正式 PyTorch 运行时。
