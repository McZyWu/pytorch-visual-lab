"use client";

import { useMemo, useState } from "react";
import FullApiBrowser from "./full-api-browser";

type Category = "张量基础" | "形状变换" | "数学运算" | "神经网络" | "损失函数";

type Param = { name: string; meaning: string; example: string };

type Lesson = {
  id: string;
  category: Category;
  name: string;
  shortName: string;
  level: "入门" | "常用" | "进阶";
  summary: string;
  detail: string;
  syntax: string;
  formula: string;
  formulaNote: string;
  params: Param[];
  code: string;
  input: string;
  secondary?: string;
  axis?: number;
  output: string;
  insight: string;
};

const lessons: Lesson[] = [
  {
    id: "tensor", category: "张量基础", name: "torch.tensor", shortName: "创建张量", level: "入门",
    summary: "把 Python 列表或数组转换为 PyTorch 张量。",
    detail: "Tensor 是 PyTorch 中承载数据和参与计算的核心对象。它像多维数组，但可以在 GPU 上运行并记录梯度。",
    syntax: "torch.tensor(data, dtype=None, device=None, requires_grad=False)",
    formula: "X ∈ ℝᵐˣⁿ",
    formulaNote: "m 表示行数，n 表示列数；每个位置 Xᵢⱼ 保存一个数值。",
    params: [
      { name: "data", meaning: "用于创建张量的原始数据", example: "[[1, 2], [3, 4]]" },
      { name: "dtype", meaning: "元素的数据类型", example: "torch.float32" },
      { name: "device", meaning: "张量所在设备", example: "\"cpu\" / \"cuda\"" },
      { name: "requires_grad", meaning: "是否追踪梯度", example: "True" },
    ],
    code: "x = torch.tensor(\n    [[1, 2], [3, 4]],\n    dtype=torch.float32\n)\nprint(x.shape)  # torch.Size([2, 2])",
    input: "[[1, 2], [3, 4]]", output: "[[1, 2], [3, 4]]",
    insight: "训练数据通常先被转换成浮点 Tensor；分类标签则常用 torch.long。",
  },
  {
    id: "arange", category: "张量基础", name: "torch.arange", shortName: "等差序列", level: "入门",
    summary: "生成指定区间内的等差一维张量。",
    detail: "常用于制作索引、位置编码和可重复的测试输入。右端点不包含在结果中。",
    syntax: "torch.arange(start=0, end, step=1, dtype=None)",
    formula: "xₖ = start + k · step,  xₖ < end",
    formulaNote: "k 从 0 开始递增；step 是相邻元素之间的差。",
    params: [
      { name: "start", meaning: "序列起点（包含）", example: "0" },
      { name: "end", meaning: "序列终点（不包含）", example: "6" },
      { name: "step", meaning: "相邻元素间隔", example: "1" },
    ],
    code: "x = torch.arange(0, 6, 1)\nprint(x)  # tensor([0, 1, 2, 3, 4, 5])",
    input: "[0, 1, 2, 3, 4, 5]", output: "[0, 1, 2, 3, 4, 5]",
    insight: "arange 适合整数索引；需要固定数量的等距浮点数时使用 torch.linspace。",
  },
  {
    id: "reshape", category: "形状变换", name: "torch.reshape", shortName: "重塑形状", level: "常用",
    summary: "在元素总数不变的前提下改变张量形状。",
    detail: "reshape 只改变数据的观察方式，不改变元素顺序。使用 -1 可以让 PyTorch 自动推断一个维度。",
    syntax: "torch.reshape(input, shape)  或  input.reshape(*shape)",
    formula: "∏ old_shape = ∏ new_shape",
    formulaNote: "变换前后各维度乘积必须相等，也就是元素总数保持不变。",
    params: [
      { name: "input", meaning: "待变形的输入张量", example: "shape=(2, 3)" },
      { name: "shape", meaning: "目标形状；最多一个维度为 -1", example: "(3, 2) / (-1,)" },
    ],
    code: "x = torch.tensor([[1, 2, 3], [4, 5, 6]])\ny = x.reshape(3, 2)\n# tensor([[1, 2], [3, 4], [5, 6]])",
    input: "[[1, 2, 3], [4, 5, 6]]", secondary: "[3, 2]", output: "[[1, 2], [3, 4], [5, 6]]",
    insight: "批量送入全连接层前，常用 x.reshape(x.size(0), -1) 展平特征维。",
  },
  {
    id: "unsqueeze", category: "形状变换", name: "torch.unsqueeze", shortName: "增加维度", level: "常用",
    summary: "在指定位置插入一个长度为 1 的维度。",
    detail: "模型经常要求明确的 batch 或 channel 维。unsqueeze 不复制数据，只改变形状视图。",
    syntax: "torch.unsqueeze(input, dim)  或  input.unsqueeze(dim)",
    formula: "(d₀,…,dₙ) → (d₀,…,1,…,dₙ)",
    formulaNote: "dim 决定新维度插入的位置，负数从末尾开始计数。",
    params: [
      { name: "input", meaning: "输入张量", example: "shape=(3,)" },
      { name: "dim", meaning: "插入新维度的位置", example: "0" },
    ],
    code: "x = torch.tensor([1, 2, 3])\ny = x.unsqueeze(0)\nprint(y.shape)  # torch.Size([1, 3])",
    input: "[1, 2, 3]", axis: 0, output: "[[1, 2, 3]]",
    insight: "单张图片 [C,H,W] 送入模型前，可用 unsqueeze(0) 变成 [1,C,H,W]。",
  },
  {
    id: "cat", category: "形状变换", name: "torch.cat", shortName: "拼接张量", level: "常用",
    summary: "沿已有维度连接多个张量。",
    detail: "除拼接维度外，其余维度必须完全一致。cat 会延长一个已有维度，不会创建新维度。",
    syntax: "torch.cat(tensors, dim=0)",
    formula: "A ∈ ℝᵐˣⁿ, B ∈ ℝᵖˣⁿ ⇒ cat₀(A,B) ∈ ℝ⁽ᵐ⁺ᵖ⁾ˣⁿ",
    formulaNote: "沿 dim=0 拼接时行数相加，列数必须相同。",
    params: [
      { name: "tensors", meaning: "要拼接的张量序列", example: "(a, b)" },
      { name: "dim", meaning: "进行拼接的维度", example: "0 / 1" },
    ],
    code: "a = torch.tensor([[1, 2]])\nb = torch.tensor([[3, 4]])\ny = torch.cat((a, b), dim=0)",
    input: "[[1, 2]]", secondary: "[[3, 4]]", axis: 0, output: "[[1, 2], [3, 4]]",
    insight: "融合多路特征时常沿通道维拼接；若想创建新维度，应使用 torch.stack。",
  },
  {
    id: "matmul", category: "数学运算", name: "torch.matmul", shortName: "矩阵乘法", level: "常用",
    summary: "执行向量点积、矩阵乘法或批量矩阵乘法。",
    detail: "matmul 会根据输入维数选择对应规则，并对前置的批量维自动广播。二维输入时就是标准矩阵乘法。",
    syntax: "torch.matmul(input, other)  或  input @ other",
    formula: "Cᵢⱼ = Σₖ AᵢₖBₖⱼ",
    formulaNote: "i 是结果行，j 是结果列，k 遍历 A 的列与 B 的行。",
    params: [
      { name: "input / A", meaning: "左侧矩阵，形状 (m, k)", example: "[[1, 2], [3, 4]]" },
      { name: "other / B", meaning: "右侧矩阵，形状 (k, n)", example: "[[2, 0], [1, 2]]" },
    ],
    code: "A = torch.tensor([[1., 2.], [3., 4.]])\nB = torch.tensor([[2., 0.], [1., 2.]])\nC = A @ B\n# tensor([[4., 4.], [10., 8.]])",
    input: "[[1, 2], [3, 4]]", secondary: "[[2, 0], [1, 2]]", output: "[[4, 4], [10, 8]]",
    insight: "线性层的核心就是矩阵乘法再加偏置：Y = XWᵀ + b。",
  },
  {
    id: "mean", category: "数学运算", name: "torch.mean", shortName: "求平均值", level: "入门",
    summary: "计算全部元素或指定维度的算术平均。",
    detail: "维度归约是理解张量计算的关键：指定 dim 后，该维度被聚合；keepdim=True 可保留长度为 1 的维。",
    syntax: "torch.mean(input, dim=None, keepdim=False)",
    formula: "μ = (1/N) Σᵢ xᵢ",
    formulaNote: "N 是参与平均的元素数量，xᵢ 是第 i 个元素。",
    params: [
      { name: "input", meaning: "浮点输入张量", example: "[[1, 2], [3, 4]]" },
      { name: "dim", meaning: "要消除并求平均的维度", example: "0 / 1 / None" },
      { name: "keepdim", meaning: "是否保留被归约的维度", example: "False" },
    ],
    code: "x = torch.tensor([[1., 2.], [3., 4.]])\ny = x.mean(dim=1)\n# tensor([1.5, 3.5])",
    input: "[[1, 2], [3, 4]]", axis: 1, output: "[1.5, 3.5]",
    insight: "对 batch 维求平均可得到整个批次的平均损失；对空间维求平均可做全局平均池化。",
  },
  {
    id: "argmax", category: "数学运算", name: "torch.argmax", shortName: "最大值索引", level: "常用",
    summary: "返回最大元素所在位置的索引。",
    detail: "分类模型输出每一类的分数，argmax 常用于选择分数最高的预测类别。返回的是索引，不是最大值本身。",
    syntax: "torch.argmax(input, dim=None, keepdim=False)",
    formula: "ŷ = arg maxⱼ xⱼ",
    formulaNote: "在所有候选 j 中，找到令 xⱼ 最大的那个索引。",
    params: [
      { name: "input", meaning: "待比较的分数张量", example: "[[0.1, 0.8, 0.1]]" },
      { name: "dim", meaning: "在哪个维度寻找最大值", example: "1" },
    ],
    code: "logits = torch.tensor([[0.1, 0.8, 0.1], [2., 1., 4.]])\npred = logits.argmax(dim=1)\n# tensor([1, 2])",
    input: "[[0.1, 0.8, 0.1], [2, 1, 4]]", axis: 1, output: "[1, 2]",
    insight: "训练时不要在交叉熵之前 argmax，因为离散索引无法提供有效梯度。",
  },
  {
    id: "relu", category: "神经网络", name: "torch.relu", shortName: "ReLU 激活", level: "入门",
    summary: "保留正数并把负数截断为 0。",
    detail: "ReLU 为神经网络引入非线性，同时计算简单、正区间梯度稳定，是隐藏层最常用的激活函数之一。",
    syntax: "torch.relu(input)  或  nn.ReLU()(input)",
    formula: "ReLU(x) = max(0, x)",
    formulaNote: "x ≤ 0 时输出 0；x > 0 时原样输出 x。",
    params: [{ name: "input", meaning: "任意形状的输入张量", example: "[-2, -0.5, 0, 3]" }],
    code: "x = torch.tensor([-2., -0.5, 0., 3.])\ny = torch.relu(x)\n# tensor([0., 0., 0., 3.])",
    input: "[-2, -0.5, 0, 3]", output: "[0, 0, 0, 3]",
    insight: "若大量神经元长期落在负区间，可尝试 LeakyReLU 或 GELU。",
  },
  {
    id: "sigmoid", category: "神经网络", name: "torch.sigmoid", shortName: "Sigmoid 激活", level: "常用",
    summary: "把任意实数压缩到 0 与 1 之间。",
    detail: "Sigmoid 常把单个 logit 转换为二分类概率。极大或极小输入处梯度接近 0，因此较少用于深层隐藏层。",
    syntax: "torch.sigmoid(input)",
    formula: "σ(x) = 1 / (1 + e⁻ˣ)",
    formulaNote: "e 是自然常数，x=0 时输出 0.5，输入越大输出越接近 1。",
    params: [{ name: "input / x", meaning: "模型输出的实数分数（logit）", example: "[-2, 0, 2]" }],
    code: "logits = torch.tensor([-2., 0., 2.])\nprobs = torch.sigmoid(logits)\n# tensor([0.1192, 0.5000, 0.8808])",
    input: "[-2, 0, 2]", output: "[0.1192, 0.5, 0.8808]",
    insight: "训练二分类模型时优先用 BCEWithLogitsLoss，它在内部组合 Sigmoid，数值更稳定。",
  },
  {
    id: "softmax", category: "神经网络", name: "torch.softmax", shortName: "Softmax 概率", level: "常用",
    summary: "把一组分数转换为总和为 1 的概率分布。",
    detail: "Softmax 会放大分数差异，并沿指定维度归一化。多分类中通常沿类别维计算。",
    syntax: "torch.softmax(input, dim)",
    formula: "pᵢ = eˣⁱ / Σⱼ eˣʲ",
    formulaNote: "xᵢ 是第 i 类 logit，分母对同一组全部类别求和，所以 Σpᵢ=1。",
    params: [
      { name: "input", meaning: "未经归一化的 logits", example: "[1, 2, 3]" },
      { name: "dim", meaning: "类别所在的维度", example: "-1" },
    ],
    code: "logits = torch.tensor([1., 2., 3.])\nprobs = torch.softmax(logits, dim=0)\n# tensor([0.0900, 0.2447, 0.6652])",
    input: "[1, 2, 3]", axis: 0, output: "[0.09, 0.2447, 0.6652]",
    insight: "CrossEntropyLoss 已经包含 LogSoftmax，训练前不要再手动调用 softmax。",
  },
  {
    id: "linear", category: "神经网络", name: "nn.Linear", shortName: "全连接层", level: "常用",
    summary: "对最后一个维度做线性变换。",
    detail: "每个输出特征都是所有输入特征的加权和，再加一个可学习偏置。权重和偏置会在训练中更新。",
    syntax: "nn.Linear(in_features, out_features, bias=True)",
    formula: "Y = XWᵀ + b",
    formulaNote: "X 是输入，W 是权重矩阵，b 是偏置，Y 是输出。",
    params: [
      { name: "in_features", meaning: "每个样本的输入特征数", example: "2" },
      { name: "out_features", meaning: "希望得到的输出特征数", example: "2" },
      { name: "bias", meaning: "是否学习额外偏置", example: "True" },
    ],
    code: "layer = nn.Linear(2, 2)\n# 假设 W=[[1,0.5],[-1,2]], b=[0.1,0]\nx = torch.tensor([[2., 3.]])\ny = layer(x)  # [[3.6, 4.0]]",
    input: "[[2, 3]]", secondary: "[[1, 0.5], [-1, 2]]", output: "[[3.6, 4]]",
    insight: "Linear 只认最后一维作为特征维，前面的维度都会被当作批次维保留。",
  },
  {
    id: "conv2d", category: "神经网络", name: "nn.Conv2d", shortName: "二维卷积", level: "进阶",
    summary: "用可学习卷积核提取图像的局部空间特征。",
    detail: "卷积核在高度和宽度方向滑动，对每个局部窗口做加权求和。输入通常采用 [N,C,H,W] 布局。",
    syntax: "nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)",
    formula: "Yₒᵢⱼ = bₒ + ΣcΣuΣv Wₒcuv · Xc,i+u,j+v",
    formulaNote: "o/c 是输出/输入通道，i/j 是空间位置，u/v 遍历卷积核。",
    params: [
      { name: "in_channels", meaning: "输入通道数", example: "3（RGB）" },
      { name: "out_channels", meaning: "卷积核数量，即输出通道数", example: "16" },
      { name: "kernel_size", meaning: "卷积核高宽", example: "3" },
      { name: "stride / padding", meaning: "滑动步长 / 边缘填充", example: "1 / 1" },
    ],
    code: "conv = nn.Conv2d(1, 1, 2, bias=False)\n# kernel = [[1, 0], [0, -1]]\nx = torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]])\ny = conv(x)  # 每个窗口左上减右下",
    input: "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", secondary: "[[1, 0], [0, -1]]", output: "[[-4, -4], [-4, -4]]",
    insight: "padding=1 配合 3×3 卷积和 stride=1，通常能保持特征图高宽不变。",
  },
  {
    id: "mse", category: "损失函数", name: "nn.MSELoss", shortName: "均方误差", level: "入门",
    summary: "计算预测值与目标值之差的平方平均。",
    detail: "MSE 对较大误差惩罚更重，常用于回归任务。预测和目标需要具有相同或可广播的形状。",
    syntax: "nn.MSELoss(reduction='mean')(input, target)",
    formula: "L = (1/N) Σᵢ (ŷᵢ − yᵢ)²",
    formulaNote: "ŷ 是模型预测，y 是真实目标，N 是元素数量。",
    params: [
      { name: "input / ŷ", meaning: "模型预测值", example: "[2, 4, 6]" },
      { name: "target / y", meaning: "真实目标值", example: "[1, 5, 5]" },
      { name: "reduction", meaning: "如何汇总逐元素损失", example: "mean / sum / none" },
    ],
    code: "pred = torch.tensor([2., 4., 6.])\ntarget = torch.tensor([1., 5., 5.])\nloss = nn.MSELoss()(pred, target)\n# tensor(1.)",
    input: "[2, 4, 6]", secondary: "[1, 5, 5]", output: "1",
    insight: "异常值很多时，SmoothL1Loss 往往比 MSE 更稳健。",
  },
  {
    id: "crossentropy", category: "损失函数", name: "nn.CrossEntropyLoss", shortName: "多类交叉熵", level: "进阶",
    summary: "比较多分类 logits 与真实类别索引。",
    detail: "它把 LogSoftmax 与 NLLLoss 合并为一个数值稳定的操作。输入应是原始 logits，目标是类别索引。",
    syntax: "nn.CrossEntropyLoss()(input, target)",
    formula: "L = −log( eˣʸ / Σⱼeˣʲ )",
    formulaNote: "xʸ 是真实类别 y 的 logit；损失越小，模型给真实类别的概率越高。",
    params: [
      { name: "input", meaning: "形状 [N,C] 的原始 logits", example: "[[1, 2, 3]]" },
      { name: "target", meaning: "形状 [N]、dtype=long 的类别索引", example: "[2]" },
    ],
    code: "logits = torch.tensor([[1., 2., 3.]])\ntarget = torch.tensor([2])\nloss = nn.CrossEntropyLoss()(logits, target)\n# tensor(0.4076)",
    input: "[[1, 2, 3]]", secondary: "[2]", output: "0.4076",
    insight: "最常见错误是先 softmax、或把 one-hot 标签传给期望类别索引的训练代码。",
  },
];

const categories: ("全部" | Category)[] = ["全部", "张量基础", "形状变换", "数学运算", "神经网络", "损失函数"];

function shapeOf(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  if (value.length === 0) return [0];
  return [value.length, ...shapeOf(value[0])];
}

function flatten(value: unknown): number[] {
  return Array.isArray(value) ? value.flatMap(flatten) : [Number(value)];
}

function mapDeep(value: unknown, fn: (n: number) => number): unknown {
  return Array.isArray(value) ? value.map((item) => mapDeep(item, fn)) : fn(Number(value));
}

function roundDeep(value: unknown): unknown {
  return mapDeep(value, (n) => Number(n.toFixed(4)));
}

function reshape(values: number[], dims: number[]): unknown {
  if (dims.length === 0) return values[0];
  const chunk = dims.slice(1).reduce((a, b) => a * b, 1);
  return Array.from({ length: dims[0] }, (_, i) => reshape(values.slice(i * chunk, (i + 1) * chunk), dims.slice(1)));
}

function reduceAxis(value: unknown, axis: number, reducer: (values: number[]) => number): unknown {
  if (!Array.isArray(value)) return value;
  if (axis === 0) {
    if (!Array.isArray(value[0])) return reducer(value.map(Number));
    const width = (value[0] as unknown[]).length;
    return Array.from({ length: width }, (_, i) => reduceAxis(value.map((row) => (row as unknown[])[i]), 0, reducer));
  }
  return value.map((item) => reduceAxis(item, axis - 1, reducer));
}

function runLesson(lesson: Lesson, inputText: string, secondaryText: string, axis: number) {
  const a = JSON.parse(inputText);
  const b = secondaryText.trim() ? JSON.parse(secondaryText) : undefined;
  let result: unknown = a;
  switch (lesson.id) {
    case "tensor": case "arange": result = a; break;
    case "reshape": {
      const dims = (b as number[]).map(Number);
      const values = flatten(a);
      const missing = dims.indexOf(-1);
      if (missing >= 0) dims[missing] = values.length / dims.filter((d) => d !== -1).reduce((x, y) => x * y, 1);
      if (dims.reduce((x, y) => x * y, 1) !== values.length) throw new Error("目标形状的元素总数必须与输入一致");
      result = reshape(values, dims); break;
    }
    case "unsqueeze": {
      const dims = shapeOf(a).length;
      const normalized = axis < 0 ? axis + dims + 1 : axis;
      if (normalized < 0 || normalized > dims) throw new Error(`dim 应在 ${-(dims + 1)} 到 ${dims} 之间`);
      const insert = (v: unknown, depth: number): unknown => depth === 0 ? [v] : (v as unknown[]).map((item) => insert(item, depth - 1));
      result = insert(a, normalized); break;
    }
    case "cat": {
      if (!Array.isArray(a) || !Array.isArray(b)) throw new Error("cat 需要两个数组张量");
      if (axis === 0) result = [...a, ...b];
      else if (axis === 1) result = a.map((row, i) => [...(row as unknown[]), ...((b[i] as unknown[]) ?? [])]);
      else throw new Error("当前实验台演示支持 dim=0 或 1");
      break;
    }
    case "matmul": {
      const left = a as number[][]; const right = b as number[][];
      if (!Array.isArray(left[0]) || !Array.isArray(right?.[0]) || left[0].length !== right.length) throw new Error("A 的列数必须等于 B 的行数");
      result = left.map((row) => right[0].map((_, j) => row.reduce((sum, n, k) => sum + n * right[k][j], 0)));
      break;
    }
    case "mean": result = reduceAxis(a, axis, (xs) => xs.reduce((s, n) => s + n, 0) / xs.length); break;
    case "argmax": result = reduceAxis(a, axis, (xs) => xs.indexOf(Math.max(...xs))); break;
    case "relu": result = mapDeep(a, (n) => Math.max(0, n)); break;
    case "sigmoid": result = mapDeep(a, (n) => 1 / (1 + Math.exp(-n))); break;
    case "softmax": {
      const soft = (xs: number[]) => { const m = Math.max(...xs); const exps = xs.map((n) => Math.exp(n - m)); const sum = exps.reduce((s, n) => s + n, 0); return exps.map((n) => n / sum); };
      if (shapeOf(a).length === 1) result = soft(a as number[]);
      else if (axis === 1 || axis === -1) result = (a as number[][]).map(soft);
      else { const cols = (a as number[][])[0].map((_, j) => soft((a as number[][]).map((row) => row[j]))); result = (a as number[][]).map((row, i) => row.map((_, j) => cols[j][i])); }
      break;
    }
    case "linear": {
      const x = a as number[][]; const weights = b as number[][]; const bias = [0.1, 0];
      result = x.map((row) => weights.map((w, j) => row.reduce((s, n, i) => s + n * w[i], bias[j] ?? 0)));
      break;
    }
    case "conv2d": {
      const x = a as number[][]; const kernel = b as number[][]; const oh = x.length - kernel.length + 1; const ow = x[0].length - kernel[0].length + 1;
      result = Array.from({ length: oh }, (_, i) => Array.from({ length: ow }, (_, j) => kernel.reduce((sum, kr, u) => sum + kr.reduce((s, n, v) => s + n * x[i + u][j + v], 0), 0)));
      break;
    }
    case "mse": { const x = flatten(a), y = flatten(b); if (x.length !== y.length) throw new Error("预测值和目标值元素数必须一致"); result = x.reduce((s, n, i) => s + (n - y[i]) ** 2, 0) / x.length; break; }
    case "crossentropy": { const logits = (a as number[][])[0]; const target = (b as number[])[0]; const m = Math.max(...logits); const logSum = Math.log(logits.reduce((s, n) => s + Math.exp(n - m), 0)) + m; result = logSum - logits[target]; break; }
  }
  return roundDeep(result);
}

function shapeLabel(value: unknown) {
  const shape = shapeOf(value);
  return shape.length ? `[${shape.join(", ")}]` : "标量";
}

function Matrix({ value, tone = "input" }: { value: unknown; tone?: "input" | "output" }) {
  const flat = flatten(value);
  const shape = shapeOf(value);
  const rows = shape.length > 1 ? shape.slice(0, -1).reduce((a, b) => a * b, 1) : 1;
  const cols = shape.length ? shape[shape.length - 1] : 1;
  const visible = flat.slice(0, 36);
  return (
    <div className={`matrix matrix--${tone}`} style={{ gridTemplateColumns: `repeat(${Math.min(cols, 8)}, minmax(38px, 1fr))` }}>
      {visible.map((n, i) => <span key={i}>{Number.isInteger(n) ? n : Number(n.toFixed(3))}</span>)}
      {flat.length > visible.length && <span className="matrix__more">+{flat.length - visible.length}</span>}
      <small>{rows} × {cols}</small>
    </div>
  );
}

export default function Home() {
  const [selectedId, setSelectedId] = useState("matmul");
  const [category, setCategory] = useState<(typeof categories)[number]>("全部");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"解析" | "参数" | "代码">("解析");
  const initialLesson = lessons.find((item) => item.id === "matmul") ?? lessons[0];
  const [completed, setCompleted] = useState<string[]>([]);
  const [inputText, setInputText] = useState(initialLesson.input);
  const [secondaryText, setSecondaryText] = useState(initialLesson.secondary ?? "");
  const [axis, setAxis] = useState(initialLesson.axis ?? 0);
  const [result, setResult] = useState<unknown>(() => runLesson(initialLesson, initialLesson.input, initialLesson.secondary ?? "", initialLesson.axis ?? 0));
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const lesson = lessons.find((item) => item.id === selectedId) ?? lessons[0];
  const filtered = useMemo(() => lessons.filter((item) =>
    (category === "全部" || item.category === category) &&
    `${item.name}${item.shortName}${item.summary}`.toLowerCase().includes(query.toLowerCase())
  ), [category, query]);

  function execute(current = lesson, a = inputText, b = secondaryText, d = axis) {
    try { setResult(runLesson(current, a, b, d)); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "请输入有效的 JSON 数组"); }
  }

  function selectLesson(id: string) {
    const next = lessons.find((item) => item.id === id);
    if (!next) return;
    setSelectedId(id); setInputText(next.input); setSecondaryText(next.secondary ?? ""); setAxis(next.axis ?? 0); setTab("解析"); setCopied(false); setError("");
    setResult(runLesson(next, next.input, next.secondary ?? "", next.axis ?? 0));
  }

  let parsedInput: unknown = [];
  try { parsedInput = JSON.parse(inputText || "[]"); } catch { /* editor shows validation on run */ }

  const progress = Math.round((completed.length / lessons.length) * 100);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="TorchScope 首页">
          <span className="brand__mark">T</span>
          <span><b>TorchScope</b><small>看见每一步计算</small></span>
        </a>
        <nav aria-label="主导航"><a className="active" href="#learn">深度实验</a><a href="#all-apis">全部 API</a><a href="#path">学习路径</a></nav>
        <div className="progress-chip" aria-label={`学习进度 ${progress}%`}><span>{completed.length}/{lessons.length}</span><i><b style={{ width: `${progress}%` }} /></i></div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">PYTORCH · INTERACTIVE ATLAS</p>
          <h1>不漏掉任何 API，<em>也不止于目录。</em></h1>
          <p>全量同步 PyTorch 2.13 官方文档接口；每一个 API 都有中文公式、变量解析、场景 Example 和输入输出实验。</p>
        </div>
        <div className="hero__demo" aria-label="张量变换示意">
          <div><small>输入 A · [2, 2]</small><Matrix value={[[1, 2], [3, 4]]} /></div>
          <span className="hero__operator">@</span>
          <div><small>输入 B · [2, 2]</small><Matrix value={[[2, 0], [1, 2]]} /></div>
          <span className="hero__arrow">→</span>
          <div><small>输出 C · [2, 2]</small><Matrix value={[[4, 4], [10, 8]]} tone="output" /></div>
        </div>
      </section>

      <section className="coverage-banner" aria-label="接口覆盖说明"><b>9,066</b><span>个官方 API 全部可进入深度实验</span><i>中文解析</i><i>公式变量</i><i>场景 Example</i><i>实际计算输出</i></section>

      <section className="category-strip" aria-label="深度课程分类">
        {categories.map((item) => <button key={item} className={category === item ? "selected" : ""} onClick={() => setCategory(item)}>{item}{item !== "全部" && <sup>{lessons.filter((l) => l.category === item).length}</sup>}</button>)}
      </section>

      <section className="workspace" id="learn">
        <aside className="catalog">
          <div className="catalog__head"><div><small>DEEP-DIVE LESSONS</small><h2>重点实验</h2></div><span>{filtered.length}</span></div>
          <label className="search"><span>⌕</span><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索函数或用途…" aria-label="搜索函数" /></label>
          <div className="lesson-list">
            {filtered.map((item, index) => (
              <button key={item.id} className={selectedId === item.id ? "active" : ""} onClick={() => selectLesson(item.id)}>
                <span className="lesson-list__num">{String(index + 1).padStart(2, "0")}</span>
                <span><code>{item.name}</code><small>{item.shortName} · {item.level}</small></span>
                {completed.includes(item.id) && <i aria-label="已掌握">✓</i>}
              </button>
            ))}
            {!filtered.length && <p className="empty">没有匹配的函数，换个关键词试试。</p>}
          </div>
        </aside>

        <article className="lesson">
          <div className="lesson__meta"><span>{lesson.category}</span><span>·</span><span>{lesson.level}</span></div>
          <div className="lesson__title-row"><div><h2>{lesson.name}</h2><p>{lesson.summary}</p></div><button className={completed.includes(lesson.id) ? "mastered" : ""} onClick={() => setCompleted((old) => old.includes(lesson.id) ? old.filter((id) => id !== lesson.id) : [...old, lesson.id])}>{completed.includes(lesson.id) ? "✓ 已掌握" : "标记掌握"}</button></div>
          <div className="syntax"><small>调用方法</small><code>{lesson.syntax}</code><button onClick={async () => { await navigator.clipboard?.writeText(lesson.syntax); setCopied(true); setTimeout(() => setCopied(false), 1200); }}>{copied ? "已复制" : "复制"}</button></div>
          <div className="tabs" role="tablist" aria-label="课程内容">
            {(["解析", "参数", "代码"] as const).map((item) => <button key={item} role="tab" aria-selected={tab === item} onClick={() => setTab(item)}>{item}</button>)}
          </div>

          {tab === "解析" && <div className="tab-content">
            <section className="explanation"><h3>它做什么？</h3><p>{lesson.detail}</p></section>
            <section className="formula-card"><div><small>核心公式</small><strong>{lesson.formula}</strong></div><p>{lesson.formulaNote}</p></section>
            <section><h3>变量地图</h3><div className="variable-grid">{lesson.params.slice(0, 4).map((p) => <div key={p.name}><code>{p.name}</code><p>{p.meaning}</p><small>例：{p.example}</small></div>)}</div></section>
            <aside className="insight"><b>场景提示</b><p>{lesson.insight}</p></aside>
          </div>}
          {tab === "参数" && <div className="tab-content"><section><h3>参数逐项拆解</h3><div className="param-table" role="table">{lesson.params.map((p) => <div role="row" key={p.name}><code role="cell">{p.name}</code><span role="cell">{p.meaning}</span><small role="cell">{p.example}</small></div>)}</div></section><aside className="insight"><b>读参数的小技巧</b><p>先确认输入和输出的 shape，再看 dim 指向哪个轴；大多数张量错误都能从这两点定位。</p></aside></div>}
          {tab === "代码" && <div className="tab-content"><section><h3>最小可运行示例</h3><pre className="code-block"><span>PYTHON</span><code>{lesson.code}</code></pre></section><aside className="insight"><b>动手建议</b><p>先预测输出，再到右侧实验台运行；如果预测不同，重点检查形状、维度和广播规则。</p></aside></div>}
        </article>

        <aside className="lab" id="lab">
          <div className="lab__head"><div><small>LIVE TENSOR LAB</small><h2>张量实验台</h2></div><span className="live"><i /> 即时</span></div>
          <p className="lab__hint">输入标准 JSON 数组，修改后点击运行。</p>
          <label><span>输入 Tensor A <code>{shapeLabel(parsedInput)}</code></span><textarea value={inputText} onChange={(e) => setInputText(e.target.value)} spellCheck={false} /></label>
          {lesson.secondary !== undefined && <label><span>{lesson.id === "reshape" ? "目标 shape" : lesson.id === "linear" ? "权重 W" : lesson.id === "mse" || lesson.id === "crossentropy" ? "目标 y" : "输入 Tensor B"}</span><textarea value={secondaryText} onChange={(e) => setSecondaryText(e.target.value)} spellCheck={false} /></label>}
          {["unsqueeze", "cat", "mean", "argmax", "softmax"].includes(lesson.id) && <label><span>维度 dim <code>{axis}</code></span><input type="range" min={lesson.id === "softmax" ? -1 : 0} max="1" step="1" value={axis} onChange={(e) => setAxis(Number(e.target.value))} /></label>}
          <button className="run" onClick={() => execute()}><span>▶</span> 运行 {lesson.name}</button>
          {error ? <div className="error"><b>输入有问题</b><p>{error}</p></div> : <>
            <div className="flow-label"><span>输入 {shapeLabel(parsedInput)}</span><i>函数执行</i><span>输出 {shapeLabel(result)}</span></div>
            <div className="output"><div className="output__top"><span>OUTPUT</span><code>dtype: float32</code></div><Matrix value={result} tone="output" /><pre>{JSON.stringify(result)}</pre></div>
          </>}
          <div className="try-row"><span>快速试验</span><button onClick={() => { const v = lesson.id === "relu" ? "[-5, -1, 0, 2, 8]" : lesson.input; setInputText(v); execute(lesson, v, secondaryText, axis); }}>示例输入</button><button onClick={() => { setInputText(lesson.input); setSecondaryText(lesson.secondary ?? ""); setAxis(lesson.axis ?? 0); execute(lesson, lesson.input, lesson.secondary ?? "", lesson.axis ?? 0); }}>重置</button></div>
        </aside>
      </section>

      <FullApiBrowser />

      <section className="path" id="path">
        <div><p className="eyebrow">RECOMMENDED PATH</p><h2>从一个 Tensor，走到完整训练循环</h2><p>推荐按数据 → 形状 → 计算 → 网络 → 损失的顺序学习。掌握一个，就点亮一个节点。</p></div>
        <ol>{categories.slice(1).map((item, i) => { const group = lessons.filter((l) => l.category === item); const done = group.filter((l) => completed.includes(l.id)).length; return <li key={item} className={done === group.length ? "done" : ""}><span>{done === group.length ? "✓" : i + 1}</span><div><b>{item}</b><small>{done} / {group.length} 已掌握</small></div></li>; })}</ol>
      </section>

      <footer><div className="brand"><span className="brand__mark">T</span><span><b>TorchScope</b><small>PyTorch 中文可视化学习图谱</small></span></div><p>数据留在本地浏览器 · 无需登录 · 为理解而设计</p></footer>
    </main>
  );
}
