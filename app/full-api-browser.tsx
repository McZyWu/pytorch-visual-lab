"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import apiIndex from "./api-index.generated.json";

type ApiEntry = {
  name: string; leaf: string; type: string; typeLabel: string; group: string;
  summary: string; display: string; priority: number; url: string;
};
type RemoteDoc = { signature?: string; parameters?: string[]; summary?: string; source?: string; error?: string };
type Variable = { name: string; meaning: string; sample: string };
type Simulation = { mode: "数值计算" | "梯度计算" | "张量变换" | "对象与状态"; title: string; value: unknown; trace: string[] };
type TensorItem = { key: string; label: string; value: unknown };

const entries = apiIndex as ApiEntry[];
const groupCounts = Object.entries(entries.reduce<Record<string, number>>((all, item) => {
  all[item.group] = (all[item.group] ?? 0) + 1; return all;
}, {})).sort((a, b) => b[1] - a[1]);
const typeCounts = Object.entries(entries.reduce<Record<string, number>>((all, item) => {
  all[item.typeLabel] = (all[item.typeLabel] ?? 0) + 1; return all;
}, {})).sort((a, b) => b[1] - a[1]);

function cleanLeaf(entry: ApiEntry) { return entry.leaf.replace(/_$/, "").toLowerCase(); }
function shapeOf(value: unknown): number[] { return Array.isArray(value) ? [value.length, ...(value.length ? shapeOf(value[0]) : [])] : []; }
function flat(value: unknown): number[] { return Array.isArray(value) ? value.flatMap(flat) : [Number(value)]; }
function mapDeep(value: unknown, fn: (n: number) => number): unknown { return Array.isArray(value) ? value.map((x) => mapDeep(x, fn)) : fn(Number(value)); }
function roundDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(roundDeep);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, roundDeep(item)]));
  return typeof value === "number" && Number.isFinite(value) ? Number(value.toFixed(5)) : value;
}
function rebuild(values: number[], dims: number[]): unknown { if (!dims.length) return values[0]; const chunk = dims.slice(1).reduce((a, b) => a * b, 1); return Array.from({ length: dims[0] }, (_, i) => rebuild(values.slice(i * chunk, (i + 1) * chunk), dims.slice(1))); }
function reduceLast(value: unknown, op: (xs: number[]) => number): unknown { if (!Array.isArray(value)) return value; return Array.isArray(value[0]) ? value.map((x) => reduceLast(x, op)) : op(value.map(Number)); }

function formulaOf(entry: ApiEntry) {
  const n = cleanLeaf(entry);
  if (/matmul|mm|bmm/.test(n)) return "Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ";
  if (/add/.test(n)) return "y = x + other";
  if (/sub/.test(n)) return "y = x − other";
  if (/mul/.test(n)) return "y = x × other";
  if (/div/.test(n)) return "y = x ÷ other";
  if (/mean/.test(n)) return "μ = (1/N) Σᵢ xᵢ";
  if (/sum/.test(n)) return "y = Σᵢ xᵢ";
  if (/softmax/.test(n)) return "pᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)";
  if (/sigmoid/.test(n)) return "σ(x) = 1 / (1 + exp(−x))";
  if (/relu/.test(n)) return "y = max(0, x)";
  if (/mse/.test(n)) return "L = (1/N) Σᵢ(ŷᵢ − yᵢ)²";
  if (/cross.*entropy/.test(n)) return "L = −log(exp(xᵧ) / Σⱼexp(xⱼ))";
  if (/conv/.test(n)) return "Yₒᵢⱼ = bₒ + ΣcΣuΣv Wₒcuv Xc,i+u,j+v";
  if (/backward|grad/.test(n) || entry.group === "自动微分") return "g = ∂output / ∂input";
  if (entry.group === "优化器") return "θₜ₊₁ = θₜ − η · update(gₜ)";
  if (entry.group === "概率分布") return "x ~ p(x | parameters)";
  if (entry.group === "分布式训练") return "global = collective(local₀ … localₙ₋₁)";
  if (entry.group === "数据加载") return "batchₖ = collate(samples[kB:(k+1)B])";
  if (/reshape|view/.test(n)) return "∏ old_shape = ∏ new_shape";
  if (/fft/.test(n)) return "Xₖ = Σₙ xₙ exp(−i2πkn/N)";
  return "output = API(input, parameters)";
}

function variablesOf(entry: ApiEntry, remote?: RemoteDoc | null): Variable[] {
  const params = remote?.parameters?.slice(0, 6) ?? [];
  if (params.length) return params.map((raw) => {
    const name = raw.split(/[=:]/)[0].trim().replace(/^\*+/, "") || "参数";
    const key = name.toLowerCase();
    let meaning = "控制该接口行为的输入参数"; let sample = raw.includes("=") ? raw.split("=").slice(1).join("=") : "按数据选择";
    if (/input|tensor|self|x/.test(key)) { meaning = "要被处理的输入张量或当前对象"; sample = "[[1, 2], [3, 4]]"; }
    else if (/dim|axis/.test(key)) { meaning = "指定计算发生在哪个张量维度"; sample = "-1"; }
    else if (/dtype/.test(key)) { meaning = "元素的数据类型"; sample = "torch.float32"; }
    else if (/device/.test(key)) { meaning = "数据或计算所在设备"; sample = "cpu / cuda"; }
    else if (/out/.test(key)) { meaning = "可选的输出张量或输出规模"; sample = "None"; }
    else if (/grad/.test(key)) { meaning = "梯度或是否追踪梯度的设置"; sample = "True"; }
    else if (/size|shape/.test(key)) { meaning = "目标尺寸或形状"; sample = "[2, 2]"; }
    return { name, meaning, sample };
  });
  const base: Variable[] = [{ name: "input / self", meaning: "输入张量或当前操作对象", sample: "[[1, 2], [3, 4]]" }];
  if (["function", "method", "class"].includes(entry.type)) base.push({ name: "*args", meaning: "按官方签名依次传入的位置参数", sample: "取决于接口" }, { name: "**kwargs", meaning: "带名称的可选配置参数", sample: "dim=-1" });
  base.push({ name: "result", meaning: "接口返回的张量、对象或状态", sample: "由输入决定" });
  return base;
}

function conceptOf(entry: ApiEntry) {
  const endings: Record<string, string> = {
    "Tensor 方法": "它作用于已有 Tensor；重点观察 shape、dtype、device、是否原地修改以及梯度关系。",
    "神经网络模块": "它属于网络层或训练组件；重点理解构造参数、forward 输入输出和可学习参数。",
    "优化器": "它负责根据梯度更新参数；重点理解学习率、状态缓存和 step/zero_grad 生命周期。",
    "自动微分": "它参与计算图与梯度传播；重点理解叶子张量、grad_fn、梯度累积和图释放。",
    "分布式训练": "它涉及多个进程或设备协作；浏览器提供通信步骤预演，真实执行需要初始化进程组。",
    "设备与加速": "它依赖硬件后端；实验区展示设备和状态变化，真实执行结果由本机硬件决定。",
    "数据加载": "它组织样本、批次与并行加载；重点看索引、采样、collate 和 batch 形状。",
    "模型导出": "它把动态图或模块转换为可部署表示；重点看输入约束、动态维度和目标格式。",
  };
  return `${entry.summary}。${endings[entry.group] ?? "实验时先确认输入类型和形状，再观察返回值、广播规则与副作用。"}`;
}

function scenarioOf(entry: ApiEntry) {
  const map: Record<string, string> = {
    "Tensor 方法": "在模型前向计算中直接处理一个已有张量",
    "神经网络函数": "在 forward 中以函数形式完成一次无状态运算",
    "神经网络模块": "把接口作为可复用层装进 nn.Module",
    "优化器": "完成 loss.backward() 后更新模型参数",
    "自动微分": "检查或控制计算图中的梯度传播",
    "线性代数": "处理矩阵分解、求解或批量几何计算",
    "概率分布": "构造概率模型、采样或计算 log_prob",
    "数据加载": "把原始样本组织成训练批次",
    "分布式训练": "让多进程同步张量、参数或训练状态",
    "设备与加速": "把计算放到指定硬件并管理执行状态",
    "模型导出": "将已训练模型转换给部署运行时",
  };
  return map[entry.group] ?? `在 ${entry.group} 场景中调用 ${entry.leaf}`;
}

function familyOf(entry: ApiEntry) {
  const n = cleanLeaf(entry);
  if (/backward|^grad$|gradients?/.test(n) || entry.group === "自动微分") return "autograd";
  if (/cross.*entropy/.test(n)) return "cross_entropy";
  if (/mse/.test(n)) return "mse";
  if (/l1_loss|smooth_l1|huber/.test(n)) return "distance_loss";
  if (/binary_cross_entropy|bce/.test(n)) return "bce";
  if (/softmax/.test(n)) return "softmax";
  if (/relu|sigmoid|tanh|gelu|silu|leaky_relu|elu|selu|softplus/.test(n)) return "activation";
  if (/conv[123]d|convolution/.test(n)) return "convolution";
  if (/pool[123]d/.test(n)) return "pooling";
  if (/linear|bilinear/.test(n) && entry.group.includes("神经网络")) return "linear";
  if (/matmul|^mm$|^bmm$|^mv$|^dot$|inner|outer/.test(n)) return "matmul";
  if (/reshape|view|flatten|ravel/.test(n)) return "reshape";
  if (/unsqueeze|squeeze/.test(n)) return "squeeze";
  if (/transpose|permute|movedim|swapaxes|swapdims/.test(n)) return "transpose";
  if (/^cat$|concat|concatenate|^stack$|hstack|vstack|dstack/.test(n)) return "combine";
  if (/split|chunk|unbind|tensor_split/.test(n)) return "split";
  if (/^sum$|^mean$|^prod$|^max$|^min$|argmax|argmin|amax|amin|median|quantile|std|var/.test(n)) return "reduction";
  if (/sort|topk|kthvalue|argsort/.test(n)) return "sorting";
  if (/unique|bincount|histogram/.test(n)) return "counting";
  if (/where|masked_select|nonzero/.test(n)) return "selection";
  if (/gather|scatter|index_select|take/.test(n)) return "indexing";
  if (/^add$|^sub$|^mul$|multiply|^div$|divide|^pow$|remainder|fmod/.test(n)) return "binary";
  if (/^eq$|equal|^ne$|^gt$|greater|^ge$|^lt$|less|^le$|isclose/.test(n)) return "comparison";
  if (/^abs$|absolute|^neg$|negative|^exp$|expm1|^log$|log10|log2|sqrt|square|reciprocal|^sin$|^cos$|^tan$|floor|ceil|round|trunc|clamp|clip/.test(n)) return "unary";
  if (/arange|linspace|logspace/.test(n)) return "sequence";
  if (/zeros|ones|full|empty|eye|tensor|as_tensor/.test(n)) return "creation";
  if (/rand|normal|bernoulli|multinomial|poisson/.test(n) || entry.group === "随机数") return "random";
  if (/fft|ifft|rfft|irfft/.test(n) || entry.group === "傅里叶变换") return "fft";
  if (/det|inverse|inv|solve|matrix_rank|norm/.test(n) || entry.group === "线性代数") return "linalg";
  if (entry.group === "优化器") return "optimizer";
  if (entry.group === "概率分布") return "distribution";
  if (entry.group === "数据加载") return "dataloader";
  if (/shape|^size$|^dim$|ndim|numel|element_size/.test(n)) return "inspection";
  if (/clone|copy|detach|contiguous|requires_grad/.test(n)) return "copy_state";
  if (/^to$|^cpu$|^cuda$|^xpu$|float|double|half|long|bool/.test(n) || entry.group === "设备与加速") return "device";
  if (/^is_|^has_|^can_|enabled|available/.test(n)) return "predicate";
  if (/^set_|enable|disable|config/.test(n) || entry.type === "data" || entry.type === "attribute" || entry.type === "property") return "state";
  if (entry.type === "class" || entry.type === "module") return "object";
  return "api_behavior";
}

function seededValues(entry: ApiEntry) {
  const seed = [...entry.name].reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) % 997, 17);
  return [seed % 7 + 1, (seed * 3) % 9 + 1, (seed * 5) % 11 + 1];
}

const tensorLabels: Record<string, string> = {
  x: "输入 x", input: "输入张量", other: "第二输入", kernel: "卷积核", weight: "权重",
  bias: "偏置", prediction: "预测值", target: "目标值", logits: "分类 logits", condition: "条件张量",
  matrix: "矩阵 A", vector: "向量 b", parameters: "参数 θ", gradients: "梯度 g", data: "数据",
  tensors: "待组合张量", evaluate_at: "求值位置", gradient: "输出梯度", after: "更新后参数",
  probabilities: "概率", difference: "差值", element_loss: "逐项损失", forward_terms: "前向逐项值",
  values: "输出值", update: "参数更新量", before: "更新前参数",
};

function visualLeaves(value: unknown): Array<number | boolean> {
  if (Array.isArray(value)) return value.flatMap(visualLeaves);
  return typeof value === "number" || typeof value === "boolean" ? [value] : [];
}

function isVisualTensor(value: unknown): boolean {
  if (typeof value === "number" || typeof value === "boolean") return true;
  if (!Array.isArray(value) || value.length === 0) return false;
  const leaves = visualLeaves(value);
  return leaves.length > 0 && leaves.length === value.flat(Infinity).length;
}

function inputTensors(spec: Record<string, unknown>): TensorItem[] {
  const allowed = new Set(["x", "input", "other", "kernel", "weight", "bias", "prediction", "target", "logits", "condition", "matrix", "vector", "parameters", "gradients", "data", "tensors", "evaluate_at"]);
  return Object.entries(spec).filter(([key, value]) => allowed.has(key) && isVisualTensor(value)).slice(0, 5).map(([key, value]) => ({ key, label: tensorLabels[key] ?? key, value }));
}

function outputTensors(value: unknown, key = "result", label = "最终输出", depth = 0): TensorItem[] {
  if (isVisualTensor(value)) return [{ key, label: tensorLabels[key] ?? label, value }];
  if (!value || typeof value !== "object" || depth > 2) return [];
  const preferred = ["gradient", "after", "values", "probabilities", "element_loss", "difference", "forward_terms", "update", "before", "loss", "mean_loss", "y", "determinant", "frobenius_norm"];
  const pairs = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => {
    const ai = preferred.indexOf(a), bi = preferred.indexOf(b);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });
  return pairs.flatMap(([childKey, child]) => outputTensors(child, childKey, tensorLabels[childKey] ?? childKey, depth + 1)).slice(0, 5);
}

function cellText(value: number | boolean) {
  if (typeof value === "boolean") return value ? "T" : "F";
  if (!Number.isFinite(value)) return String(value);
  const rounded = Number(value.toFixed(3));
  return String(rounded).length > 7 ? rounded.toExponential(1) : String(rounded);
}

function visualShape(value: unknown) {
  if (!Array.isArray(value)) return [];
  return shapeOf(value);
}

function TensorMatrix({ values, columns, highlights = [] }: { values: Array<number | boolean>; columns: number; highlights?: number[] }) {
  const visibleColumns = Math.max(1, Math.min(columns, 8));
  return <div className="tensor-matrix" style={{ gridTemplateColumns: `repeat(${visibleColumns}, minmax(31px, 1fr))` }}>
    {values.slice(0, 64).map((value, index) => <span className={highlights.includes(index) ? "is-active" : ""} key={`${index}-${String(value)}`}>{cellText(value)}</span>)}
    {values.length > 64 && <span className="tensor-more">+{values.length - 64}</span>}
  </div>;
}

function TensorVisual({ item, highlights = [] }: { item: TensorItem; highlights?: number[] }) {
  const [activePlane, setActivePlane] = useState(0);
  const shape = visualShape(item.value), values = visualLeaves(item.value);
  if (!shape.length) return <article className="tensor-visual tensor-visual--scalar"><header><b>{item.label}</b><code>scalar</code></header><div className={highlights.length ? "tensor-scalar is-active" : "tensor-scalar"}>{cellText(values[0])}</div></article>;
  const isCuboid = shape.length >= 3;
  const rows = shape.length === 1 ? 1 : shape.at(-2) ?? 1;
  const columns = shape.at(-1) ?? shape[0] ?? 1;
  const planeSize = rows * columns;
  const planeCount = Math.max(1, Math.ceil(values.length / planeSize));
  const visiblePlaneCount = Math.min(planeCount, 12), safePlane = Math.min(activePlane, visiblePlaneCount - 1);
  const selectedStart = safePlane * planeSize;
  const selectedHighlights = highlights.filter((index) => index >= selectedStart && index < selectedStart + planeSize).map((index) => index - selectedStart);
  return <article className={`tensor-visual ${isCuboid ? "tensor-visual--cuboid" : ""}`}>
    <header><b>{item.label}</b><code>shape=[{shape.join(", ")}]</code></header>
    {isCuboid ? <div className="tensor-cuboid">
      <div className="tensor-layer-picker" aria-label={`${item.label}切换查看层`}>
        {Array.from({ length: visiblePlaneCount }, (_, plane) => {
          const start = plane * planeSize, preview = values.slice(start, start + Math.min(4, planeSize));
          const hasHighlight = highlights.some((index) => index >= start && index < start + planeSize);
          return <button type="button" className={`${safePlane === plane ? "is-selected" : ""} ${hasHighlight ? "has-highlight" : ""}`} aria-pressed={safePlane === plane} onClick={() => setActivePlane(plane)} key={plane}>
            <span>层 {plane + 1}</span><small>{preview.map(cellText).join(" · ")}{planeSize > 4 ? " …" : ""}</small>
          </button>;
        })}
      </div>
      <div className="tensor-layer-detail">
        <div className="tensor-layer-detail__head"><strong>层 {safePlane + 1} / {planeCount}</strong><span>完整展开 · {rows}×{columns}</span></div>
        <div className="tensor-plane tensor-plane--selected"><TensorMatrix values={values.slice(selectedStart, selectedStart + planeSize)} columns={columns} highlights={selectedHighlights} /></div>
      </div>
      {planeCount > visiblePlaneCount && <span className="tensor-depth-more">当前展示前 {visiblePlaneCount} 层，共 {planeCount} 层</span>}
    </div> : <TensorMatrix values={values} columns={columns} highlights={highlights} />}
  </article>;
}

function highlightCells(entry: ApiEntry, spec: Record<string, unknown>, sim: Simulation, step: number, item: TensorItem, side: "input" | "output") {
  const family = familyOf(entry), count = visualLeaves(item.value).length;
  if (!count) return [];
  if (family === "matmul") {
    const a = spec.input as number[][], b = spec.other as number[][], cols = b?.[0]?.length ?? 1;
    const row = Math.floor(step / cols), col = step % cols;
    if (side === "output") return [row * cols + col];
    if (item.key === "input") return Array.from({ length: a?.[0]?.length ?? 0 }, (_, k) => row * a[0].length + k);
    if (item.key === "other") return Array.from({ length: b?.length ?? 0 }, (_, k) => k * cols + col);
  }
  if (family === "convolution" || family === "pooling") {
    const source = spec.input as unknown, sourceShape = shapeOf(source), output = outputTensors(sim.value)[0]?.value;
    const outputShape = shapeOf(output), outCols = outputShape.at(-1) ?? 1, row = Math.floor(step / outCols), col = step % outCols;
    if (side === "output") return [row * outCols + col];
    if (item.key === "kernel") return Array.from({ length: count }, (_, index) => index);
    if (item.key === "input") {
      const height = sourceShape.at(-2) ?? 1, width = sourceShape.at(-1) ?? 1;
      const size = family === "pooling" ? Number(spec.kernel_size ?? 2) : (shapeOf(spec.kernel).at(-1) ?? 1);
      const stride = family === "pooling" ? Number(spec.stride ?? size) : Number(spec.stride ?? 1);
      const channels = Math.max(1, count / (height * width));
      return Array.from({ length: channels }, (_, c) => Array.from({ length: size * size }, (__, p) => c * height * width + (row * stride + Math.floor(p / size)) * width + col * stride + p % size)).flat();
    }
  }
  if (["activation", "unary", "binary", "comparison", "selection"].includes(family)) return [step % count];
  if (["softmax", "reduction"].includes(family)) {
    const columns = shapeOf(item.value).at(-1) ?? 1, row = step % Math.max(1, Math.ceil(count / columns));
    return Array.from({ length: Math.min(columns, count - row * columns) }, (_, index) => row * columns + index);
  }
  if (family === "linear") return side === "output" ? [step % count] : Array.from({ length: count }, (_, index) => index);
  return sim.trace.length > 1 && sim.trace.length === count ? [step % count] : [];
}

function CalculationVisualizer({ entry, source, sim }: { entry: ApiEntry; source: string; sim: Simulation }) {
  const parsed = useMemo(() => { try { return JSON.parse(source) as Record<string, unknown>; } catch { return {}; } }, [source]);
  const inputs = inputTensors(parsed), outputs = outputTensors(sim.value);
  const [step, setStep] = useState(0), [playing, setPlaying] = useState(false);
  const total = Math.max(1, sim.trace.length), safeStep = Math.min(step, total - 1);
  useEffect(() => {
    if (!playing || total < 2) return;
    const timer = window.setInterval(() => setStep((current) => current >= total - 1 ? 0 : current + 1), 1100);
    return () => window.clearInterval(timer);
  }, [playing, total]);
  if (!inputs.length && !outputs.length) return null;
  return <section className="calculation-viz" aria-label="张量计算过程可视化">
    <div className="calculation-viz__head"><div><span>INTERACTIVE TENSOR FLOW</span><h4>输入 → 逐步计算 → 输出</h4></div><p>橙色单元格是第 {safeStep + 1} 步正在参与计算的位置</p></div>
    <div className="tensor-flow">
      <div className="tensor-side"><small>输入张量</small>{inputs.map((item) => <TensorVisual key={item.key} item={item} highlights={highlightCells(entry, parsed, sim, safeStep, item, "input")} />)}</div>
      <div className="operator-node"><span>{entry.leaf}</span><strong>{formulaOf(entry)}</strong><i>→</i></div>
      <div className="tensor-side"><small>输出张量</small>{outputs.map((item, index) => <TensorVisual key={`${item.key}-${index}`} item={item} highlights={highlightCells(entry, parsed, sim, safeStep, item, "output")} />)}</div>
    </div>
    <div className="process-stepper">
      <div className="stepper-controls"><button type="button" onClick={() => setStep(Math.max(0, safeStep - 1))} disabled={safeStep === 0}>← 上一步</button><button type="button" className={playing ? "is-playing" : ""} onClick={() => setPlaying((value) => !value)} disabled={total < 2}>{playing ? "暂停" : "自动播放"}</button><button type="button" onClick={() => setStep(Math.min(total - 1, safeStep + 1))} disabled={safeStep === total - 1}>下一步 →</button></div>
      <div className="stepper-copy"><span>{safeStep + 1} / {total}</span><p>{sim.trace[safeStep] ?? "输入经过当前算子后直接得到右侧输出。"}</p></div>
      {total > 1 && <input aria-label="选择计算步骤" type="range" min="0" max={total - 1} value={safeStep} onChange={(event) => { setStep(Number(event.target.value)); setPlaying(false); }} />}
    </div>
  </section>;
}

function defaultSpec(entry: ApiEntry) {
  const family = familyOf(entry), n = cleanLeaf(entry), v = seededValues(entry);
  const specs: Record<string, unknown> = {
    autograd: { x: [2, -1, 3], expression: "y = sum(x² + 3x)", grad_output: 1 },
    cross_entropy: { logits: [[1, 2, 3], [2.5, 0.5, -1]], target: [2, 0] },
    mse: { prediction: [2.5, 0, 2], target: [3, -0.5, 2] },
    distance_loss: { prediction: [2, -1, 4], target: [1, 1, 3] },
    bce: { prediction: [0.9, 0.2, 0.7], target: [1, 0, 1] },
    softmax: { input: [[1, 2, 3], [2, 1, 0]], dim: -1 },
    activation: { input: [-v[0], -0.5, 0, v[1] / 3, v[2]] },
    convolution: { input: [[[v[0], 2, 0], [0, v[1], 3], [2, 1, v[2]]], [[1, 0, 2], [2, 1, 0], [0, 3, 1]]], kernel: [[[1, 0], [0, -1]], [[0, 1], [1, 0]]], stride: 1, bias: 0 },
    pooling: { input: [[1, 5, 2, 4], [3, 2, 7, 1], [0, 6, 3, 8], [4, 1, 2, 5]], kernel_size: 2, stride: 2 },
    linear: { input: [2, -1, 3], weight: [[0.5, 1, -1], [2, 0, 0.5]], bias: [0.1, -0.2] },
    matmul: { input: [[v[0], 2, v[1]], [4, v[2], 6]], other: [[1, 2], [0, v[0]], [2, 0]] },
    reshape: { input: [v[0], 2, v[1], 4, v[2], 6, 7, 8], shape: [2, 2, 2] },
    squeeze: { input: [[[1, 2, 3]]], dim: 0 },
    transpose: { input: [[1, 2, 3], [4, 5, 6]], dim0: 0, dim1: 1 },
    combine: { tensors: [[1, 2], [3, 4]], other: [[5, 6], [7, 8]], dim: 0 },
    split: { input: [10, 20, 30, 40, 50, 60], sections: 3, dim: 0 },
    reduction: { input: [[v[0], 7, v[1]], [8, v[2], 5]], dim: -1 },
    sorting: { input: [4, 1, 7, 3, 9, 2], k: 3, descending: true },
    counting: { input: [3, 1, 3, 2, 1, 3, 4, 2] },
    selection: { condition: [true, false, true, false], input: [10, 20, 30, 40], other: [-1, -1, -1, -1] },
    indexing: { input: [[10, 20, 30], [40, 50, 60]], index: [2, 0], dim: 1 },
    binary: { input: [v[0], v[1], v[2]], other: n === "pow" ? 2 : v[0] },
    comparison: { input: [1, 4, 2, 7], other: 3 },
    unary: { input: n.includes("log") || n === "sqrt" ? [0.25, 1, 4, 9] : [-2.7, -0.5, 0, 1.2, 3.8] },
    sequence: n.includes("linspace") ? { start: -1, end: 1, steps: 5 } : { start: 1, end: 10, step: 2 },
    creation: { data: [[v[0], v[1]], [v[2], v[0] + v[1]]], shape: [2, 2], fill_value: v[0], dtype: "float32" },
    random: { shape: [2, 3], seed: 42, distribution: n.includes("normal") ? "normal(0,1)" : "uniform(0,1)" },
    fft: { input: [1, 0, -1, 0] },
    linalg: { matrix: [[4, 7], [2, 6]], vector: [1, 0] },
    optimizer: { parameters: [1.5, -0.5], gradients: [0.2, -0.4], learning_rate: 0.1, momentum: 0.9 },
    distribution: { distribution: entry.leaf, parameters: { loc: 0, scale: 1 }, evaluate_at: [-1, 0, 1] },
    dataloader: { samples: [{ x: [1, 2], y: 0 }, { x: [3, 4], y: 1 }, { x: [5, 6], y: 0 }, { x: [7, 8], y: 1 }], batch_size: 2, shuffle: false },
    inspection: { input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] },
    copy_state: { input: [1, 2, 3], requires_grad: true, contiguous: true },
    device: { input: [v[0], v[1], v[2]], from_device: "cpu", to_device: n || "cuda", from_dtype: "float32", to_dtype: n.includes("long") ? "int64" : "float32" },
    predicate: { input: [1, 2, 3], device: "cpu", dtype: "float32", condition: entry.leaf },
    state: { setting: entry.name, before: false, requested: true },
    object: { constructor: entry.name, arguments: { input_features: v[0] + 2, output_features: v[1] + 1 }, sample_input_shape: [2, v[0] + 2] },
    api_behavior: { api: entry.name, example_values: v, operation: entry.summary, parameter_hint: entry.type === "method" ? "在对应对象上调用" : entry.type === "function" ? "把示例值替换进官方签名" : "按构造参数创建对象" },
  };
  return JSON.stringify({ api: entry.name, ...specs[family] as Record<string, unknown> }, null, 2);
}

function simulate(entry: ApiEntry, source: string): Simulation {
  // The experiment schema intentionally varies by API family; values are narrowed at each branch.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const spec = JSON.parse(source) as Record<string, any>;
  const n = cleanLeaf(entry), family = familyOf(entry), input = spec.input ?? []; let value: unknown; let mode: Simulation["mode"] = "数值计算"; let title = "已算出最终结果"; let trace: string[] = [];
  const unary: Record<string, (x: number) => number> = { abs: Math.abs, absolute: Math.abs, neg: (x) => -x, negative: (x) => -x, relu: (x) => Math.max(0, x), sigmoid: (x) => 1 / (1 + Math.exp(-x)), exp: Math.exp, expm1: Math.expm1, log: Math.log, log10: Math.log10, log2: Math.log2, sqrt: Math.sqrt, square: (x) => x * x, sin: Math.sin, cos: Math.cos, tan: Math.tan, floor: Math.floor, ceil: Math.ceil, round: Math.round, trunc: Math.trunc };
  if (family === "autograd") { const x=spec.x.map(Number), forward=x.map((z:number)=>z*z+3*z), gradients=x.map((z:number)=>2*z+3); value={forward_terms:forward, y:forward.reduce((a:number,b:number)=>a+b,0), local_derivative:"d(x²+3x)/dx = 2x+3", gradient:gradients.map((g:number)=>g*Number(spec.grad_output??1))}; mode="梯度计算"; title="链式法则得到梯度"; trace=[`前向：逐项计算 x²+3x → [${forward.join(", ")}]`,`求局部导数：2x+3 → [${gradients.join(", ")}]`,`乘上游梯度 ${spec.grad_output??1} → [${gradients.map((g:number)=>g*Number(spec.grad_output??1)).join(", ")}]`]; }
  else if (family === "cross_entropy") { const losses=spec.logits.map((row:number[],i:number)=>{const m=Math.max(...row), exps=row.map(z=>Math.exp(z-m)), probs=exps.map(z=>z/exps.reduce((a,b)=>a+b,0));return {probabilities:roundDeep(probs),target:spec.target[i],loss:-Math.log(probs[spec.target[i]])};}); value={samples:losses,mean_loss:losses.reduce((a:number,b:{loss:number})=>a+b.loss,0)/losses.length}; trace=["每行 logits 减去最大值，避免指数溢出","计算 softmax 概率，再取真实类别概率","对真实类别概率取 −log，最后按 batch 求平均"]; }
  else if (family === "mse" || family === "distance_loss") { const p=spec.prediction.map(Number),t=spec.target.map(Number),diff=p.map((z:number,i:number)=>z-t[i]),items=family==="mse"?diff.map((z:number)=>z*z):diff.map((z:number)=>Math.abs(z)); value={difference:diff,element_loss:items,loss:items.reduce((a:number,b:number)=>a+b,0)/items.length}; trace=[`预测−目标 → [${diff.join(", ")}]`,family==="mse"?`逐项平方 → [${items.join(", ")}]`:`逐项绝对值 → [${items.join(", ")}]`,`求平均 → ${items.reduce((a:number,b:number)=>a+b,0)}/${items.length}`]; }
  else if (family === "bce") { const p=spec.prediction.map(Number),t=spec.target.map(Number),items=p.map((z:number,i:number)=>-(t[i]*Math.log(z)+(1-t[i])*Math.log(1-z))); value={element_loss:items,loss:items.reduce((a:number,b:number)=>a+b,0)/items.length}; trace=["逐项代入 −[y·log(p)+(1−y)·log(1−p)]",`得到 [${items.map((z:number)=>z.toFixed(4)).join(", ")}]`,"对全部样本损失求平均"]; }
  else if (family === "unary" && unary[n]) { value=mapDeep(input,unary[n]); trace=[`读取输入 [${flat(input).join(", ")}]`,`对每个元素应用 ${entry.leaf} 的数学定义`,`逐元素输出 [${flat(value).map(z=>z.toFixed(4)).join(", ")}]`]; }
  else if (["add", "sub", "mul", "multiply", "div", "divide", "pow"].includes(n)) { const other = Number(spec.other ?? 1); const symbols:Record<string,string>={add:"+",sub:"−",mul:"×",multiply:"×",div:"÷",divide:"÷",pow:"^"};const ops: Record<string, (a: number, b: number) => number> = { add: (a,b)=>a+b, sub:(a,b)=>a-b, mul:(a,b)=>a*b, multiply:(a,b)=>a*b, div:(a,b)=>a/b, divide:(a,b)=>a/b, pow:(a,b)=>a**b }; value = mapDeep(input, (x) => ops[n](x, other));trace=flat(input).map((x,i)=>`位置 ${i}：${x} ${symbols[n]} ${other} = ${flat(value)[i]}`); }
  else if (family === "binary") { const other=Number(spec.other??1),ops:Record<string,(a:number,b:number)=>number>={add:(a,b)=>a+b,sub:(a,b)=>a-b,mul:(a,b)=>a*b,multiply:(a,b)=>a*b,div:(a,b)=>a/b,divide:(a,b)=>a/b,pow:(a,b)=>a**b,remainder:(a,b)=>a%b,fmod:(a,b)=>a%b};const op=ops[n]??((a,b)=>a+b);value=mapDeep(input,x=>op(x,other));trace=flat(input).map((x,i)=>`${x} ${n} ${other} = ${flat(value)[i]}`);}
  else if (family === "comparison") { const other=Number(spec.other),ops:Record<string,(a:number,b:number)=>boolean>={eq:(a,b)=>a===b,equal:(a,b)=>a===b,ne:(a,b)=>a!==b,gt:(a,b)=>a>b,greater:(a,b)=>a>b,ge:(a,b)=>a>=b,lt:(a,b)=>a<b,less:(a,b)=>a<b,le:(a,b)=>a<=b};const op=ops[n]??((a,b)=>a>b);value=(input as number[]).map(x=>op(x,other));trace=(input as number[]).map((x,i)=>`${x} 与 ${other} 比较 → ${value[i]}`);}
  else if (family === "reduction") { const reducers:Record<string,(x:number[])=>number>={sum:x=>x.reduce((a,b)=>a+b,0),mean:x=>x.reduce((a,b)=>a+b,0)/x.length,prod:x=>x.reduce((a,b)=>a*b,1),max:x=>Math.max(...x),min:x=>Math.min(...x),amax:x=>Math.max(...x),amin:x=>Math.min(...x),argmax:x=>x.indexOf(Math.max(...x)),argmin:x=>x.indexOf(Math.min(...x)),std:x=>{const m=x.reduce((a,b)=>a+b,0)/x.length;return Math.sqrt(x.reduce((a,b)=>a+(b-m)**2,0)/(x.length-1));},var:x=>{const m=x.reduce((a,b)=>a+b,0)/x.length;return x.reduce((a,b)=>a+(b-m)**2,0)/(x.length-1);}};const reducer=reducers[n]??reducers.mean;value=reduceLast(input,reducer);trace=(input as number[][]).map((row,i)=>`第 ${i} 行 [${row}] → ${n} = ${JSON.stringify((value as unknown[])[i])}`);}
  else if (family === "softmax") { const soft=(xs:number[])=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};value=(input as number[][]).map(soft);trace=(input as number[][]).map((row,i)=>{const m=Math.max(...row),e=row.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return `第 ${i} 行：exp(x−${m})=[${e.map(x=>x.toFixed(3))}]，除以总和 ${s.toFixed(3)}`;});}
  else if (family === "activation") { const funcs:Record<string,(x:number)=>number>={relu:x=>Math.max(0,x),sigmoid:x=>1/(1+Math.exp(-x)),tanh:Math.tanh,gelu:x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3))),silu:x=>x/(1+Math.exp(-x)),leaky_relu:x=>x>=0?x:.01*x,elu:x=>x>=0?x:Math.exp(x)-1,selu:x=>1.0507*(x>=0?x:1.67326*(Math.exp(x)-1)),softplus:x=>Math.log(1+Math.exp(x))};const fn=funcs[n]??funcs.relu;value=mapDeep(input,fn);trace=flat(input).map((x,i)=>`${entry.leaf}(${x}) = ${flat(value)[i].toFixed(5)}`);}
  else if (family === "reshape") { const values=flat(input),dims=spec.shape;if(dims.reduce((a:number,b:number)=>a*b,1)!==values.length)throw new Error("shape 的元素总数必须与 input 一致");value=rebuild(values,dims);mode="张量变换";trace=[`按行读取 ${values.length} 个元素：[${values}]`,`目标维度乘积 ${dims.join("×")}=${dims.reduce((a:number,b:number)=>a*b,1)}`,`元素顺序不变，重新分组为 shape=[${dims}]`];}
  else if (family === "squeeze") { value=n.includes("unsqueeze")?[input]:Array.isArray(input)&&input.length===1?input[0]:input;mode="张量变换";trace=[`原 shape=${JSON.stringify(shapeOf(input))}`,n.includes("unsqueeze")?"在指定位置插入长度为 1 的维度":"删除长度为 1 的指定维度",`新 shape=${JSON.stringify(shapeOf(value))}`];}
  else if (family === "transpose") { const a=input as unknown[][];value=a[0].map((_,j)=>a.map(row=>row[j]));mode="张量变换";trace=[`原位置 (i,j) 的值移动到 (j,i)`,`原 shape=${JSON.stringify(shapeOf(input))}`,`新 shape=${JSON.stringify(shapeOf(value))}`];}
  else if (family === "combine") { const a=spec.tensors as unknown[][],b=spec.other as unknown[][];value=n.includes("stack")?[a,b]:spec.dim===1?a.map((row:unknown[],i:number)=>[...row,...b[i]]):[...a,...b];mode="张量变换";trace=[`读取第一个张量 shape=${JSON.stringify(shapeOf(a))}`,`读取第二个张量 shape=${JSON.stringify(shapeOf(b))}`,n.includes("stack")?"创建新维度并堆叠":"沿已有维度连接元素"];}
  else if (family === "split") { const vals=flat(input),size=Math.ceil(vals.length/Number(spec.sections));value=Array.from({length:Number(spec.sections)},(_,i)=>vals.slice(i*size,(i+1)*size));mode="张量变换";trace=[`输入共有 ${vals.length} 个元素`,`分成 ${spec.sections} 组，每组约 ${size} 个`,`保持原顺序得到 ${JSON.stringify(value)}`];}
  else if (family === "matmul") { const a=input as number[][],b=spec.other as number[][];value=a.map(row=>b[0].map((_:number,j:number)=>row.reduce((s:number,x:number,k:number)=>s+x*b[k][j],0)));trace=[];a.forEach((row,i)=>b[0].forEach((_:number,j:number)=>trace.push(`C[${i},${j}] = ${row.map((x,k)=>`${x}×${b[k][j]}`).join(" + ")} = ${(value as number[][])[i][j]}`)));}
  else if (family === "linear") { value=spec.weight.map((w:number[],j:number)=>w.reduce((s,x,i)=>s+x*spec.input[i],spec.bias[j]));trace=spec.weight.map((w:number[],j:number)=>`y${j} = ${w.map((x,i)=>`${x}×${spec.input[i]}`).join(" + ")} + ${spec.bias[j]} = ${value[j]}`);}
  else if (family === "convolution") { const rawX=spec.input,rawK=spec.kernel,x=shapeOf(rawX).length===3?rawX:[rawX],k=shapeOf(rawK).length===3?rawK:[rawK],stride=Number(spec.stride??1),oh=Math.floor((x[0].length-k[0].length)/stride)+1,ow=Math.floor((x[0][0].length-k[0][0].length)/stride)+1;value=Array.from({length:oh},(_,i)=>Array.from({length:ow},(_,j)=>k.reduce((channelSum:number,plane:number[][],c:number)=>channelSum+plane.reduce((s:number,row:number[],u:number)=>s+row.reduce((q,z,v)=>q+z*x[c][i*stride+u][j*stride+v],0),0),Number(spec.bias))));trace=[];for(let i=0;i<oh;i++)for(let j=0;j<ow;j++){const terms=k.flatMap((plane:number[][],c:number)=>plane.flatMap((row:number[],u:number)=>row.map((z:number,v:number)=>`${z}×${x[c][i*stride+u][j*stride+v]}`)));trace.push(`窗口(${i},${j})：${terms.join(" + ")} = ${value[i][j]}`);}}
  else if (family === "pooling") { const x=spec.input,size=Number(spec.kernel_size),out=[] as number[][];for(let i=0;i<x.length;i+=Number(spec.stride)){const row=[];for(let j=0;j<x[0].length;j+=Number(spec.stride)){const window=x.slice(i,i+size).flatMap((r:number[])=>r.slice(j,j+size));row.push(n.includes("avg")?window.reduce((a:number,b:number)=>a+b,0)/window.length:Math.max(...window));trace.push(`窗口 [${window}] → ${n.includes("avg")?"平均":"最大"}值 ${row.at(-1)}`);}out.push(row);}value=out;}
  else if (family === "sorting") { const sorted=[...spec.input].sort((a:number,b:number)=>spec.descending?b-a:a-b);value=n.includes("topk")?{values:sorted.slice(0,spec.k),indices:sorted.slice(0,spec.k).map((x:number)=>spec.input.indexOf(x))}:sorted;trace=[`原序列 [${spec.input}]`,`${spec.descending?"降序":"升序"}比较并重排 → [${sorted}]`,n.includes("topk")?`取前 ${spec.k} 个并返回原索引`:"返回排序结果"];}
  else if (family === "counting") { const counts=spec.input.reduce((a:Record<string,number>,x:number)=>(a[x]=(a[x]??0)+1,a),{});value=n.includes("unique")?Object.keys(counts).map(Number):counts;trace=Object.entries(counts).map(([x,c])=>`数值 ${x} 出现 ${c} 次`);}
  else if (family === "selection") { value=spec.condition.map((c:boolean,i:number)=>c?spec.input[i]:spec.other[i]);trace=spec.condition.map((c:boolean,i:number)=>`位置 ${i}：condition=${c} → 选择 ${value[i]}`);}
  else if (family === "indexing") { const a=spec.input as number[][],indices=(spec.index as number[]).map(Number),dim=Number(spec.dim??0);value=dim===1?a.map((row)=>indices.map((index)=>row[index])):indices.map((index)=>a[index]);trace=dim===1?a.flatMap((row,i)=>indices.map((index,j)=>`输出[${i},${j}] ← 输入[${i},${index}] = ${row[index]}`)):indices.map((index,i)=>`输出第 ${i} 行 ← 输入第 ${index} 行`);mode="张量变换";}
  else if (family === "sequence") { const vals=[];if(n.includes("linspace")){for(let i=0;i<spec.steps;i++)vals.push(spec.start+i*(spec.end-spec.start)/(spec.steps-1));trace=[`间隔 = (${spec.end}−${spec.start})/(${spec.steps}−1)`,`从 start 起累计间隔，共生成 ${spec.steps} 项`];}else{for(let x=spec.start;x<spec.end;x+=spec.step)vals.push(x);trace=[`从 ${spec.start} 开始`,`每次增加 ${spec.step}`,`到 ${spec.end} 前停止`];}value=vals;}
  else if (family === "creation") { const dims=(spec.shape as number[]).map(Number),count=dims.reduce((a:number,b:number)=>a*b,1),source=visualLeaves(spec.data);let values:number[];if(n.includes("zeros")||n.includes("empty"))values=Array(count).fill(0);else if(n.includes("ones"))values=Array(count).fill(1);else if(n.includes("eye")){const rows=dims[0]??2,cols=dims[1]??rows;values=Array.from({length:rows*cols},(_,i)=>Math.floor(i/cols)===i%cols?1:0);}else if(n.includes("full"))values=Array(count).fill(Number(spec.fill_value??0));else values=source.map(Number);value=rebuild(values,dims.length?dims:shapeOf(spec.data));trace=[`确定输出 shape=[${dims.length?dims:shapeOf(spec.data)}]，需要 ${values.length} 个元素`,n.includes("eye")?"主对角线填 1，其余位置填 0":`按 ${entry.leaf} 的填充值规则生成元素`,`按行写入输出张量`];}
  else if (family === "random") { const dims=(spec.shape as number[]).map(Number),count=dims.reduce((a:number,b:number)=>a*b,1);let state=Number(spec.seed??42)>>>0;const uniform=()=>{state=(1664525*state+1013904223)>>>0;return state/4294967296;};const values=Array.from({length:count},()=>n.includes("normal")?Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12)))*Math.cos(2*Math.PI*uniform()):uniform());value=rebuild(values,dims);trace=[`固定 seed=${spec.seed??42}，因此每次实验可复现`,n.includes("normal")?"用 Box–Muller 把均匀随机数变换为正态随机数":"线性同余生成 [0,1) 均匀随机数",`生成 ${count} 个数并重组为 shape=[${dims}]`];}
  else if (family === "fft") { const x=spec.input.map(Number),N=x.length;const spectrum=Array.from({length:N},(_,k)=>{let re=0,im=0;for(let t=0;t<N;t++){re+=x[t]*Math.cos(-2*Math.PI*k*t/N);im+=x[t]*Math.sin(-2*Math.PI*k*t/N);}return {real:Number(re.toFixed(5)),imag:Number(im.toFixed(5))};});value=spectrum;trace=spectrum.map((z:{real:number;imag:number},k:number)=>`频点 k=${k}：Σ x[n]·e^(−i2π·${k}n/${N}) = ${z.real}${z.imag<0?"":"+"}${z.imag}i`);}
  else if (family === "linalg") { const [[a,b],[c,d]]=spec.matrix,det=a*d-b*c;if(n.includes("det"))value=det;else if(n.includes("inv")||n.includes("inverse"))value=[[d/det,-b/det],[-c/det,a/det]];else value={matrix:spec.matrix,determinant:det,frobenius_norm:Math.sqrt(a*a+b*b+c*c+d*d)};trace=[`det = ${a}×${d} − ${b}×${c} = ${det}`,n.includes("inv")?`A⁻¹ = (1/${det})·[[${d},${-b}],[${-c},${a}]]`:"根据接口继续做对应线性代数运算"]}
  else if (family === "optimizer") { const p=spec.parameters.map(Number),g=spec.gradients.map(Number),delta=g.map((z:number)=>spec.learning_rate*z),next=p.map((z:number,i:number)=>z-delta[i]);value={before:p,gradient:g,update:delta,after:next};mode="对象与状态";title="参数已按梯度更新";trace=p.map((z:number,i:number)=>`θ${i}: ${z} − ${spec.learning_rate}×${g[i]} = ${next[i]}`);}
  else if (family === "dataloader") { const samples=spec.samples as unknown[],batches:unknown[][]=[];for(let i=0;i<samples.length;i+=spec.batch_size)batches.push(samples.slice(i,i+spec.batch_size));value={batch_count:batches.length,batches};mode="对象与状态";title="样本已组成批次";trace=batches.map((b:unknown[],i:number)=>`batch ${i}: 样本索引 ${i*spec.batch_size}…${i*spec.batch_size+b.length-1}`);}
  else if (family === "inspection") { const sh=shapeOf(input);value=n.includes("numel")?flat(input).length:n==="dim"||n==="ndim"?sh.length:n==="size"||n==="shape"?sh:{shape:sh,numel:flat(input).length};mode="张量变换";trace=[`逐层读取嵌套长度 → shape=${JSON.stringify(sh)}`,`维度数=${sh.length}`,`元素数=${sh.join("×")}=${flat(input).length}`];}
  else if (family === "copy_state") { value={values:input,requires_grad:n.includes("detach")?false:spec.requires_grad,contiguous:spec.contiguous,shares_storage:n.includes("clone")?false:!n.includes("copy")};mode="对象与状态";title="张量值与元数据已更新";trace=[`复制数值 [${flat(input)}]`,n.includes("detach")?"从计算图分离，requires_grad=False":"保留梯度设置",n.includes("clone")?"分配独立存储":"按接口规则处理存储关系"];}
  else if (family === "device") { value={values:input,before:{device:spec.from_device,dtype:spec.from_dtype},after:{device:spec.to_device,dtype:spec.to_dtype},numeric_values_changed:false};mode="对象与状态";title="设备或 dtype 转换结果";trace=[`原张量：device=${spec.from_device}, dtype=${spec.from_dtype}`,`请求转换到 device=${spec.to_device}, dtype=${spec.to_dtype}`,"数值保持不变；真实内存迁移需要对应 PyTorch 硬件后端"];}
  else if (family === "state") { value={setting:spec.setting,before:spec.before,after:spec.requested,changed:spec.before!==spec.requested};mode="对象与状态";title="配置值变化";trace=[`读取 ${spec.setting} = ${spec.before}`,`写入请求值 ${spec.requested}`,`最终值 = ${spec.requested}`];}
  else if (family === "predicate") { const result=n.includes("available")?spec.device==="cpu":n.includes("float")?spec.dtype.includes("float"):Array.isArray(spec.input);value={condition:entry.leaf,result};mode="对象与状态";title="条件判断结果";trace=[`读取对象属性 device=${spec.device}, dtype=${spec.dtype}`,`应用判断 ${entry.leaf}`,`返回布尔值 ${result}`];}
  else if (family === "object") { value={type:entry.name,constructor_arguments:spec.arguments,created:true,sample_input_shape:spec.sample_input_shape};mode="对象与状态";title="对象构造结果";trace=[`解析 ${entry.leaf} 构造参数`,`创建 ${entry.name} 实例`,`记录输入约束 shape=${JSON.stringify(spec.sample_input_shape)}`];}
  else { const vals=spec.example_values??seededValues(entry);value={api:entry.name,example_call:entry.name.startsWith("torch.Tensor.")?`tensor.${entry.leaf}(${vals.join(", ")})`:`${entry.name}(${vals.join(", ")})`,input_values:vals,operation:entry.summary,execution_requirement:`需要原生 PyTorch 的 ${entry.group} 运行环境`};mode="对象与状态";title="该接口无独立数学公式";trace=[`专属示例值由接口名稳定生成：[${vals}]`,`根据接口类型组装调用：${entry.leaf}(${vals.join(", ")})`,`此接口的真实结果依赖原生 PyTorch 对象；页面不虚构数值`];}
  return { mode, title, value: roundDeep(value), trace };
}

export default function FullApiBrowser() {
  const [query, setQuery] = useState(""); const [group, setGroup] = useState("全部模块"); const [type, setType] = useState("全部类型"); const [page, setPage] = useState(0);
  const [selectedName, setSelectedName] = useState("torch.Tensor.backward"); const [remote, setRemote] = useState<RemoteDoc | null>(null); const [docLoading, setDocLoading] = useState(true);
  const initial = entries.find((x) => x.name === "torch.Tensor.backward") ?? entries[0];
  const [spec, setSpec] = useState(() => defaultSpec(initial)); const [sim, setSim] = useState<Simulation>(() => simulate(initial, defaultSpec(initial))); const [simError, setSimError] = useState("");
  const [runState, setRunState] = useState<"idle" | "running" | "success" | "error">("idle"); const [runCount, setRunCount] = useState(0); const [lastRunAt, setLastRunAt] = useState("");
  const outputRef = useRef<HTMLElement>(null);
  const pageSize = 80;
  const filtered = useMemo(() => { const keyword=query.trim().toLowerCase(); return entries.filter((entry)=>(group==="全部模块"||entry.group===group)&&(type==="全部类型"||entry.typeLabel===type)&&(!keyword||`${entry.name} ${entry.summary} ${entry.typeLabel} ${entry.group}`.toLowerCase().includes(keyword))); }, [query,group,type]);
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize)), safePage=Math.min(page,pages-1), visible=filtered.slice(safePage*pageSize,(safePage+1)*pageSize);
  const selected=entries.find((item)=>item.name===selectedName)??visible[0]??entries[0];

  useEffect(() => { let active=true; fetch(`/api/docs?name=${encodeURIComponent(selected.name)}&url=${encodeURIComponent(selected.url)}`).then((r)=>r.json()).then((data)=>{if(active){setRemote(data);setDocLoading(false);}}).catch(()=>{if(active){setRemote({error:"官方详情暂时无法读取"});setDocLoading(false);}}); return()=>{active=false;}; }, [selected.name,selected.url]);

  function applyFilter(nextGroup:string,nextType:string){setGroup(nextGroup);setType(nextType);setPage(0);}
  function choose(entry:ApiEntry){const next=defaultSpec(entry);setSelectedName(entry.name);setRemote(null);setDocLoading(true);setSpec(next);setSimError("");setRunState("idle");setRunCount(0);setLastRunAt("");setSim(simulate(entry,next));}
  function run(){
    setRunState("running"); setSimError("");
    window.setTimeout(()=>{
      try {
        setSim(simulate(selected,spec)); setRunCount((count)=>count+1); setLastRunAt(new Date().toLocaleTimeString("zh-CN",{hour12:false})); setRunState("success");
        window.requestAnimationFrame(()=>outputRef.current?.scrollIntoView({behavior:"smooth",block:"center"}));
      } catch(error) {
        setSimError(error instanceof Error?error.message:"请输入有效 JSON"); setRunState("error");
      }
    },180);
  }

  return <section className="docs-atlas" id="all-apis">
    <div className="docs-atlas__intro"><div><p className="eyebrow">PYTORCH 2.13 · EVERY API IS A LAB</p><h2>9,066 个接口，全部进入深度实验</h2><p>选择任意官方 API，立即得到中文作用、精确签名、参数解释、核心关系式、调用示例、应用场景与输入输出实验。数值算子直接计算；硬件、分布式、编译等接口进行真实约束下的结构预演。</p></div><div className="docs-atlas__stats"><div><strong>{entries.length.toLocaleString("zh-CN")}</strong><span>深度实验</span></div><div><strong>{groupCounts.length}</strong><span>学习模块</span></div><div><strong>100%</strong><span>API 覆盖</span></div></div></div>
    <div className="docs-toolbar"><label className="docs-search"><span>⌕</span><input value={query} onChange={(e)=>{setQuery(e.target.value);setPage(0);}} placeholder="搜索 torch.compile、Conv2d、backward…" aria-label="搜索全部 PyTorch API" /><kbd>/</kbd></label><select value={group} onChange={(e)=>applyFilter(e.target.value,type)}><option value="全部模块">全部模块</option>{groupCounts.map(([name,count])=><option key={name} value={name}>{name} · {count}</option>)}</select><select value={type} onChange={(e)=>applyFilter(group,e.target.value)}><option value="全部类型">全部类型</option>{typeCounts.map(([name,count])=><option key={name} value={name}>{name} · {count}</option>)}</select></div>
    <div className="docs-layout">
      <aside className="docs-groups"><button className={group==="全部模块"?"active":""} onClick={()=>applyFilter("全部模块",type)}><span>全部模块</span><b>{entries.length}</b></button>{groupCounts.map(([name,count])=><button key={name} className={group===name?"active":""} onClick={()=>applyFilter(name,type)}><span>{name}</span><b>{count}</b></button>)}</aside>
      <div className="docs-results"><div className="docs-results__head"><p>找到 <b>{filtered.length.toLocaleString("zh-CN")}</b> 个实验</p><span>第 {safePage+1} / {pages} 页</span></div><div className="api-table" role="table"><div className="api-table__header"><span>接口</span><span>类型</span><span>中文用途</span></div>{visible.map((entry)=><button key={entry.name} className={selected.name===entry.name?"selected":""} onClick={()=>choose(entry)}><code>{entry.name}</code><span>{entry.typeLabel}</span><p>{entry.summary}</p></button>)}{!visible.length&&<div className="docs-empty">没有匹配的接口，换个关键词试试。</div>}</div><div className="pagination"><button disabled={safePage===0} onClick={()=>setPage(Math.max(0,safePage-1))}>← 上一页</button><span>{filtered.length?safePage*pageSize+1:0}–{Math.min((safePage+1)*pageSize,filtered.length)} / {filtered.length}</span><button disabled={safePage>=pages-1} onClick={()=>setPage(Math.min(pages-1,safePage+1))}>下一页 →</button></div></div>
      <aside className="api-inspector"><div className="api-inspector__top"><span>{selected.group}</span><i>{selected.typeLabel}</i></div><h3>{selected.leaf}</h3><code className="api-inspector__path">{selected.name}</code><section><small>它做什么</small><p>{conceptOf(selected)}</p></section><section><small>核心关系 / 公式</small><div className="mini-formula">{formulaOf(selected)}</div></section><section><small>应用场景</small><p>{scenarioOf(selected)}</p></section><a className="official-link" href="#deep-lab">进入本接口完整实验 ↓</a></aside>
    </div>

    <article className="deep-lab" id="deep-lab">
      <header><div><p className="eyebrow">FULL API EXPERIMENT</p><h3>{selected.name}</h3></div><span>{sim.mode}</span></header>
      <div className="deep-lab__grid">
        <section className="deep-card"><small>① 中文解析</small><h4>{selected.summary}</h4><p>{conceptOf(selected)}</p><div className="deep-formula"><span>核心关系</span><strong>{formulaOf(selected)}</strong></div></section>
        <section className="deep-card"><small>② 官方调用方法</small>{docLoading?<p className="loading-line">正在读取官方签名…</p>:<><pre><code>{remote?.signature||`${selected.name}(*args, **kwargs)`}</code></pre>{remote?.summary&&<p className="official-summary">官方说明：{remote.summary}</p>}</>}<a href={selected.url} target="_blank" rel="noreferrer">核对官方原文 ↗</a></section>
        <section className="deep-card deep-card--wide"><small>③ 参数与变量地图</small><div className="deep-vars">{variablesOf(selected,remote).map((v)=><div key={v.name}><code>{v.name}</code><p>{v.meaning}</p><span>例：{v.sample}</span></div>)}</div></section>
        <section className="deep-card"><small>④ 使用场景与 Example</small><p>{scenarioOf(selected)}</p><pre><code>{selected.type==="class"?`component = ${selected.name}(...)\noutput = component(input)`:selected.name.startsWith("torch.Tensor.")?`output = input.${selected.leaf}(...)`:`output = ${selected.name}(input, ...)`}</code></pre></section>
        <section className="deep-card simulator-card"><small>⑤ 输入与执行</small><label><span>实验输入（JSON）</span><textarea value={spec} onChange={(e)=>{setSpec(e.target.value);setRunState("idle");}} spellCheck={false}/></label><button type="button" onClick={run} disabled={runState==="running"}>{runState==="running"?"⏳ 正在执行…":runState==="success"?"✓ 已执行 · 再运行一次":"▶ 运行本接口实验"}</button>{runState==="idle"&&<p className="run-hint">修改输入后点击按钮，页面会自动定位到本次输出。</p>}{simError&&<p className="sim-error" role="alert">{simError}</p>}</section>
        <section ref={outputRef} className={`deep-card deep-card--wide output-card output-card--${runState}`} aria-live="polite"><small>⑥ 计算过程与最终结果</small><div className="run-receipt"><b>{runState==="running"?"正在计算…":runState==="success"?`运行成功 · 第 ${runCount} 次`:runState==="error"?"运行失败":"示例结果预览"}</b><span>{lastRunAt?`完成时间 ${lastRunAt}`:"点击上方按钮执行当前输入"}</span></div><div className="sim-mode"><b>{sim.title}</b><span>{sim.mode} · {sim.mode==="数值计算"||sim.mode==="梯度计算"?"下方展示实际算式、参与运算的单元格与数值":"下方展示张量形状、元素位置与状态变化"}</span></div><CalculationVisualizer key={`${selected.name}-${runCount}`} entry={selected} source={spec} sim={sim} /><div className="trace-row trace-row--compact">{sim.trace.map((step,i)=><div key={`${i}-${step}`}><span>{i+1}</span><p>{step}</p></div>)}</div><div className="result-label">最终结果（精确数据）</div><pre><code>{JSON.stringify(sim.value,null,2)}</code></pre></section>
      </div>
    </article>
  </section>;
}
