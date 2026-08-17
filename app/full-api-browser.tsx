"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import apiIndex from "./api-index.generated.json";

type ApiEntry = {
  name: string; leaf: string; type: string; typeLabel: string; group: string;
  summary: string; display: string; priority: number; url: string;
};
type RemoteDoc = { signature?: string; parameters?: string[]; summary?: string; source?: string; error?: string };
type Variable = { name: string; meaning: string; sample: string; required?: boolean; raw?: string };
type Simulation = { mode: "数值计算" | "梯度计算" | "张量变换" | "对象与状态"; title: string; value: unknown; trace: string[] };
type SimulationTier = { label: "数值教学模拟" | "规则示意" | "流程 / 状态示意"; note: string; numeric: boolean };
type ReadabilityContract = { level: "入门" | "常用" | "进阶"; call: string; input: string; output: string; shape: string; autograd: string; pitfall: string };
type ExampleSpec = { title: string; code: string; output: string; runnable: boolean };
type TensorItem = { key: string; label: string; value: unknown };
type FormulaSpec = { latex: string; spoken: string; explanation: string; symbols: Array<{ symbol: string; meaning: string }> };
type OperationGuide = { title: string; what: string; steps: string[]; returns: string; sideEffect: string; example: string };
type PositionTerm = { inputKey: string; inputIndex: number; inputCoord: string; inputValue: number; parameterKey?: string; parameterIndex?: number; parameterCoord?: string; parameterValue?: number; result: number; operator: string };
type PositionDetail = { title: string; outputCoord: string; outputValue: number | boolean; terms: PositionTerm[]; bias?: number; rule: string; aggregation?: "sum" | "max" | "mean" | "direct" };
type ComparisonSpec = { title: string; intro: string; columns: string[]; rows: Array<{ name: string; api?: string; cells: string[] }>; note: string };
type ApiKind = "函数" | "类" | "方法" | "其他";
type DetailTab = "overview" | "usage" | "example" | "result" | "compare";
type ComparisonCatalogItem = { id: string; title: string; methods: string; difference: string; families: string[] };
type ConvDimensionDemo = { dimension: 1 | 2 | 3; name: string; axisNames: string[]; inputShape: number[]; input: number[]; kernelShape: number[]; kernel: number[]; stride: number[]; padding: number[]; dilation: number[]; description: string };
type ConvDemoTerm = { inputCoord: number[]; kernelCoord: number[]; inputValue: number; kernelValue: number; product: number; inside: boolean };
type ConvDemoPosition = { outputCoord: number[]; origin: number[]; value: number; terms: ConvDemoTerm[] };

const entries = apiIndex as ApiEntry[];
const groupCounts = Object.entries(entries.reduce<Record<string, number>>((all, item) => {
  all[item.group] = (all[item.group] ?? 0) + 1; return all;
}, {})).sort((a, b) => b[1] - a[1]);

function kindOf(entry: ApiEntry): ApiKind {
  if (entry.type === "function") return "函数";
  if (entry.type === "class") return "类";
  if (entry.type === "method") return "方法";
  return "其他";
}

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

function roundTiesToEven(x:number){const lower=Math.floor(x),fraction=x-lower;if(fraction<0.5)return lower;if(fraction>0.5)return lower+1;return lower%2===0?lower:lower+1;}
function erfApprox(x:number){const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+0.3275911*a);return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a));}
function logGammaApprox(x:number){const p=[0.9999999999998099,676.5203681218851,-1259.1392167224028,771.3234287776531,-176.6150291621406,12.507343278686905,-0.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];if(x<0.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*x))-logGammaApprox(1-x);const z=x-1;let sum=p[0];for(let i=1;i<p.length;i++)sum+=p[i]/(z+i);const t=z+7.5;return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(sum);}

const foreachUnaryRules: Record<string,{latex:string;label:string;example:(x:number)=>number}> = {
  abs:{latex:String.raw`|x|`,label:"绝对值",example:Math.abs}, ceil:{latex:String.raw`\lceil x\rceil`,label:"向正无穷取整",example:Math.ceil}, floor:{latex:String.raw`\lfloor x\rfloor`,label:"向负无穷取整",example:Math.floor},
  round:{latex:String.raw`\operatorname{round}_{\mathrm{ties\ to\ even}}(x)`,label:"舍入到最近整数；正好在中点时取偶数",example:roundTiesToEven}, trunc:{latex:String.raw`\operatorname{trunc}(x)`,label:"截去小数部分、向零取整",example:Math.trunc}, frac:{latex:String.raw`x-\operatorname{trunc}(x)`,label:"保留小数部分",example:x=>x-Math.trunc(x)},
  neg:{latex:String.raw`-x`,label:"取相反数",example:x=>-x}, reciprocal:{latex:String.raw`1/x`,label:"取倒数",example:x=>1/x}, sqrt:{latex:String.raw`\sqrt{x}`,label:"开平方",example:Math.sqrt}, exp:{latex:String.raw`e^x`,label:"计算自然指数",example:Math.exp}, expm1:{latex:String.raw`e^x-1`,label:"计算 eˣ−1",example:Math.expm1},
  log:{latex:String.raw`\ln x`,label:"计算自然对数",example:Math.log}, log1p:{latex:String.raw`\ln(1+x)`,label:"计算 ln(1+x)",example:Math.log1p}, log2:{latex:String.raw`\log_2x`,label:"计算以 2 为底的对数",example:Math.log2}, log10:{latex:String.raw`\log_{10}x`,label:"计算以 10 为底的对数",example:Math.log10},
  sin:{latex:String.raw`\sin x`,label:"计算正弦",example:Math.sin}, cos:{latex:String.raw`\cos x`,label:"计算余弦",example:Math.cos}, tan:{latex:String.raw`\tan x`,label:"计算正切",example:Math.tan}, asin:{latex:String.raw`\arcsin x`,label:"计算反正弦",example:Math.asin}, acos:{latex:String.raw`\arccos x`,label:"计算反余弦",example:Math.acos}, atan:{latex:String.raw`\arctan x`,label:"计算反正切",example:Math.atan},
  sinh:{latex:String.raw`\sinh x`,label:"计算双曲正弦",example:Math.sinh}, cosh:{latex:String.raw`\cosh x`,label:"计算双曲余弦",example:Math.cosh}, tanh:{latex:String.raw`\tanh x`,label:"计算双曲正切",example:Math.tanh}, sigmoid:{latex:String.raw`1/(1+e^{-x})`,label:"计算 Sigmoid",example:x=>1/(1+Math.exp(-x))},
  clone:{latex:String.raw`x`,label:"复制数值到新的张量存储",example:x=>x}, zero:{latex:String.raw`0`,label:"把每个元素写成 0",example:()=>0},
  erf:{latex:String.raw`\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}\,dt`,label:"计算误差函数 erf",example:erfApprox}, erfc:{latex:String.raw`1-\operatorname{erf}(x)`,label:"计算互补误差函数 erfc",example:x=>1-erfApprox(x)}, lgamma:{latex:String.raw`\ln\!\left|\Gamma(x)\right|`,label:"计算 Gamma 函数绝对值的自然对数",example:logGammaApprox},
};

function foreachOperation(entry:ApiEntry){const raw=entry.leaf.toLowerCase(),base=raw.replace(/^_foreach_/,"").replace(/_$/,"");return {base,inPlace:raw.endsWith("_"),rule:foreachUnaryRules[base]};}

function operationGuideOf(entry:ApiEntry):OperationGuide {
  const family=familyOf(entry), n=cleanLeaf(entry);
  if(family==="foreach"){
    const {base,inPlace,rule}=foreachOperation(entry),label=rule?.label??`执行 torch.${base}`;
    const calculated=rule?`[tensor([${roundDeep(rule.example(1.2))}, ${roundDeep(rule.example(-1.8))}]), tensor([[${roundDeep(rule.example(2))}, ${roundDeep(rule.example(2.3))}]])]`:"对应运算结果";
    return {title:`对张量列表逐个执行${label}`,what:`输入不是一个 Tensor，而是 List[Tensor]。函数保留列表中每个张量的 shape，并对每个张量里的每个元素执行同一个 ${base} 运算；列表中的张量彼此独立，不会互相广播。`,steps:[`读取列表中的第 k 个张量 X^(k)` ,`遍历该张量的每个位置 i：取 x=X^(k)_i，${rule?rule.label:`计算 torch.${base}(x)`}`,inPlace?`把结果写回同一位置 X^(k)_i，然后处理下一个张量`:`把结果写入新张量的同一位置 Y^(k)_i，然后处理下一个张量`],returns:inPlace?"原地版本以下划线结尾，返回 None；计算结果保存在已被修改的输入张量中。":"返回新的 List[Tensor]；列表长度与输入相同，第 k 个输出 shape 与第 k 个输入相同。",sideEffect:inPlace?"有副作用：输入列表本身仍指向原张量，但这些张量的元素已经被覆盖。":"无原地副作用：输入张量保持不变，输出张量使用新的结果存储。",example:rule?`输入 tensors=[tensor([1.2,-1.8]), tensor([[2.0,2.3]])]；${inPlace?`调用后返回 None，tensors 变为 ${calculated}`:`返回 ${calculated}，tensors 不变`}`:"每个输入张量分别调用对应的单张量运算。"};
  }
  if(family==="creation"&&n.startsWith("empty"))return {title:`${entry.leaf}：分配未初始化张量`,what:"只确定 shape、dtype、device 与内存布局，不会把元素填成 0 或任何固定值。新张量中的数值来自尚未初始化的内存，读取前必须先写入。",steps:["解析目标 shape、dtype 与 device","分配足够的存储空间，但不写入确定数值","由后续计算完整覆盖张量内容后再读取"],returns:"返回指定 shape 的新 Tensor；元素值未定义，每次运行都可能不同。",sideEffect:"创建新存储；不会修改其他张量，但直接使用未初始化值会产生不可预测结果。",example:"x = torch.empty((2, 3))  # 只能先确认 x.shape；不要断言其中的数值"};
  const guides:Record<string,OperationGuide>={
    unary:{title:"逐元素计算一个明确的数学函数",what:"输出与输入 shape 相同；每个输出位置只依赖输入同一位置的值。",steps:["取输入位置 i 的数值 xᵢ",`代入 ${entry.leaf} 对应的具体运算`,`把结果写入输出同一位置 yᵢ`],returns:"返回与输入同 shape 的新张量。",sideEffect:n.endsWith("_")?"名称以下划线结尾时会覆盖原张量。":"默认不修改输入。",example:"例如 ceil([1.2,-1.8,2.0]) = [2,-1,2]。"},
    binary:{title:"广播对齐后逐元素做二元运算",what:"两个输入先按 broadcasting 从末维对齐，再在每个位置执行加、减、乘、除或幂。",steps:["从最后一维比较两个 shape","长度相等或其中一个为 1 时扩展到共同 shape","对每个对齐位置计算并写入输出"],returns:"返回广播后的共同 shape。",sideEffect:n.endsWith("_")?"原地版本会修改第一个输入。":"默认不修改输入。",example:"add([[1],[2]],[10,20]) → [[11,21],[12,22]]。"},
    reduction:{title:"沿指定维度把多个值归约成一个值",what:"dim 决定每次取哪一条切片；对切片求和、均值、极值或索引。",steps:["固定除 dim 外的其他坐标","取出 dim 上的全部元素","按接口规则合并并写入输出坐标"],returns:"keepdim=False 删除归约维；keepdim=True 将它保留为长度 1。",sideEffect:"不修改输入。",example:"sum([[1,2],[3,4]], dim=0) = [4,6]。"},
    convolution:{title:"滑窗内逐项相乘并跨通道累加",what:"每个输出位置对应输入中的一个局部窗口；窗口与卷积核逐项相乘、求和，再加 bias。",steps:["根据 stride、padding、dilation 定位窗口","逐输入通道计算窗口值×核权重","累加所有乘积并写入输出位置"],returns:"返回带 batch、out_channels 和输出空间维度的张量。",sideEffect:"不修改输入。",example:"下方卷积实验可逐输出位置查看全部坐标和乘加项。"},
    reshape:{title:"不改数值，只重新组织 shape",what:"元素总数保持不变，按线性顺序把同一批元素重新解释为目标维度。",steps:["计算输入元素总数","验证目标 shape 元素总数相同","按原线性顺序映射到新坐标"],returns:"返回目标 shape；可能是 view，也可能复制。",sideEffect:"不修改输入数值。",example:"reshape([1,2,3,4], (2,2)) = [[1,2],[3,4]]。"},
    activation:{title:`${entry.leaf}：逐元素施加非线性变换`,what:"每个输出元素只由同一位置的输入元素计算；具体分段、指数或平滑公式显示在下方，不用未定义的 f 代替。",steps:["读取位置 i 的输入 x=Xᵢ",`代入 ${entry.leaf} 的完整公式`,`将算出的 y 写到同一位置 Yᵢ`],returns:"返回与输入 shape 相同的张量。",sideEffect:n.endsWith("_")?"原地版本覆盖输入元素。":"不修改输入。",example:"可在实验输入中放入负数、0、正数，逐项观察非线性区间。"},
    softmax:{title:`${entry.leaf}：沿 dim 把分数归一化为概率`,what:"同一条 dim 切片里的元素共同参与计算：先减最大值防止指数溢出，再取指数并除以指数和。",steps:["取出 dim 上的一整条分数向量","每项计算 exp(xᵢ−max(x))","每项除以该向量的指数总和"],returns:"shape 不变；每条被归一化切片的输出和为 1。",sideEffect:"不修改输入。",example:"softmax([1,2]) ≈ [0.26894,0.73106]。"},
    pooling:{title:`${entry.leaf}：滑动局部窗口并汇总窗口元素`,what:"卷积核窗口在空间轴上滑动，但没有可学习权重；最大池化取窗口最大值，平均池化取窗口总和除以计数。",steps:["按 kernel_size、stride、padding 定位窗口","读取当前通道窗口内的有效元素","按 max 或 mean 规则写入一个输出值"],returns:"返回通道数不变、空间尺寸按池化公式变化的张量。",sideEffect:"不修改输入。",example:"max_pool1d([1,4,2,3], kernel=2, stride=2) = [4,3]。"},
    linear:{title:`${entry.leaf}：输入向量乘权重转置后加偏置`,what:"最后一维是输入特征；每个输出特征把全部输入特征与对应权重相乘后求和。",steps:["固定 batch 等前置坐标","取一个输入向量与一行权重做点积","加该输出神经元的 bias"],returns:"最后一维由 in_features 变为 out_features。",sideEffect:"不修改输入；模块参数参与梯度计算。",example:"x=[1,2]、weight=[[3,4]]、bias=[5] 时输出 1×3+2×4+5=16。"},
    matmul:{title:`${entry.leaf}：行与列逐项相乘后累加`,what:"固定输出位置 (i,j)，取左矩阵第 i 行和右矩阵第 j 列；共享维 K 必须相等。",steps:["选取左输入第 i 行","选取右输入第 j 列","计算 K 个乘积并求和写入 Cᵢⱼ"],returns:"二维时 (M,K)×(K,N) 返回 (M,N)；批量维按规则广播。",sideEffect:"不修改输入。",example:"[[1,2]]×[[3],[4]] = [[1×3+2×4]] = [[11]]。"},
    transpose:{title:`${entry.leaf}：交换维度对应的坐标`,what:"数值本身不做算术运算；输出坐标通过维度置换映射回输入坐标。",steps:["读取要交换或重排的维度","把输出坐标按逆置换映射到输入坐标","从该输入坐标复制同一个元素"],returns:"返回维度顺序改变的张量，常见实现是共享存储的 view。",sideEffect:n.endsWith("_")?"原地版本修改张量元数据。":"不修改输入数据。",example:"transpose([[1,2,3],[4,5,6]],0,1) = [[1,4],[2,5],[3,6]]。"},
    squeeze:{title:`${entry.leaf}：增加或删除长度为 1 的维度`,what:"只修改 shape 的维度描述，不改变元素数量、数值或线性顺序。",steps:["读取输入 shape","按 dim 插入 1，或验证并删除长度为 1 的维度","用新坐标解释同一批元素"],returns:"返回新 shape 的 view。",sideEffect:n.endsWith("_")?"原地版本修改 shape 元数据。":"不修改输入。",example:"unsqueeze shape (2,3) at dim=1 → (2,1,3)。"},
    combine:{title:`${entry.leaf}：沿指定维度拼接或新增维度堆叠`,what:"cat 要求非拼接维完全相同；stack 要求所有 shape 完全相同，并先新增一个维度。",steps:["验证各输入 shape 是否兼容","计算每个输入在输出中的坐标区间","按输入顺序复制元素到对应区间"],returns:"返回包含全部输入元素的新张量。",sideEffect:"不修改输入。",example:"cat([[1,2]],[[3,4]],dim=0) = [[1,2],[3,4]]。"},
    split:{title:`${entry.leaf}：沿 dim 把一个张量切成多个视图`,what:"按块大小、块数或索引边界划分指定维度；其他维度保持不变。",steps:["读取 dim 的长度","计算每个输出片段的起止索引","为每段建立切片结果"],returns:"返回 Tensor 元组或列表；最后一段可能更短。",sideEffect:"通常不修改输入，结果常与输入共享存储。",example:"chunk([1,2,3,4,5],2) → [1,2,3] 与 [4,5]。"},
    sorting:{title:`${entry.leaf}：沿 dim 比较元素并确定排序位置`,what:"每条 dim 切片独立排序或选择前 k 项；其他坐标固定不变。",steps:["取出一条 dim 切片","比较元素并确定升降序或 Top-K 位置","写出数值，按接口同时写出原索引"],returns:n.includes("topk")||n==="sort"?"返回 (values, indices) 命名元组；两个 Tensor shape 相同，indices 通常是 int64。":n.includes("argsort")?"返回排序位置 indices，通常是 int64 Tensor。":"按具体接口返回排序值、索引或二者。",sideEffect:n.endsWith("_")?"原地版本覆盖输入顺序。":"不修改输入。",example:"topk([3,1,4,2],2) → values=[4,3], indices=[2,0]。"},
    counting:{title:`${entry.leaf}：按数值相等关系分组并计数`,what:"扫描输入，把相同值归到同一组；根据接口输出唯一值、频数、逆索引或直方图区间计数。",steps:["展开或沿 dim 读取元素","比较/哈希元素并建立分组","按首次出现或排序规则输出组及统计量"],returns:"返回唯一值或计数相关张量，shape 由组数决定。",sideEffect:"不修改输入。",example:"unique([2,1,2,3,1]) → [1,2,3]。"},
    comparison:{title:`${entry.leaf}：广播后逐位置判断比较关系`,what:"两个输入先广播到共同 shape，再逐位置执行等于、大小或近似相等判断。",steps:["从末维对齐两个 shape","把长度 1 的维度扩展到共同长度","逐位置比较并写出 True/False"],returns:"通常返回 bool 张量；整体判断接口可返回单个 bool。",sideEffect:"不修改输入。",example:"gt([1,3],[2]) → [False,True]。"},
    selection:{title:`${entry.leaf}：按条件决定每个输出位置取哪个值`,what:"条件、input 与 other 先广播；条件为真取 input，否则取 other。",steps:["广播对齐条件与候选输入","读取位置 i 的布尔条件","从两个候选值中选择一个写入 Yᵢ"],returns:"返回广播后的共同 shape。",sideEffect:"不修改输入。",example:"where([True,False],[1,2],[10,20]) = [1,20]。"},
    indexing:{title:`${entry.leaf}：用索引张量读取或写入指定坐标`,what:"index 中的每个整数决定在目标 dim 上访问哪个位置；其他维坐标沿用当前输出坐标。",steps:["读取输出位置及对应 index 值","把 index 值代入输入的目标维坐标","读取、聚合或写入该坐标"],returns:"shape 由 index 与接口规则决定。",sideEffect:/scatter|put|index_add/.test(n)?"写入型版本会更新目标张量。":"读取型版本不修改输入。",example:"gather([10,20,30], index=[2,0]) = [30,10]。"},
    sequence:{title:`${entry.leaf}：按起点、终点与步长生成数列`,what:"第 i 项直接由序列参数计算，不读取已有张量元素。",steps:["读取 start、end、step 或 steps","计算第 i 项并判断是否到达终点","按索引顺序写入输出"],returns:"返回一维序列张量。",sideEffect:"不修改其他张量。",example:"arange(1,6,2) = [1,3,5]。"},
    creation:{title:`${entry.leaf}：按目标 shape 创建并填充张量`,what:"先分配指定 shape/dtype/device 的存储，再按零、一、常量、对角线或输入数据填充。",steps:["解析 shape、dtype 与 device","分配足够的元素存储","按当前创建规则写入每个坐标"],returns:"返回新张量。",sideEffect:"创建新存储，不修改现有输入。",example:"full((2,3),7) = [[7,7,7],[7,7,7]]。"},
    random:{title:`${entry.leaf}：从指定概率分布逐元素采样`,what:"随机数生成器状态与分布参数共同决定每个值；设置相同 generator/seed 可复现序列。",steps:["读取分布参数与随机数生成器状态","为每个位置生成随机比特并转换为分布样本","更新生成器状态并写出样本"],returns:"返回目标 shape 的随机张量。",sideEffect:"推进随机数生成器状态；原地版本覆盖目标张量。",example:"manual_seed 固定后重复运行 rand 可得到相同序列。"},
    fft:{title:`${entry.leaf}：把采样序列分解成复数频率分量`,what:"每个频率 k 把全部时间样本乘以对应复指数后求和；逆变换执行相反方向并按规范缩放。",steps:["选定变换维和长度 N","为频率 k 累加 N 个样本×复指数","写出复数频谱或逆变换样本"],returns:"返回复数或实数频域/时域张量，长度依接口而定。",sideEffect:"不修改输入。",example:"fft([1,0,0,0]) = [1+0j,1+0j,1+0j,1+0j]。"},
    linalg:{title:`${entry.leaf}：按线性代数定义处理矩阵或矩阵批次`,what:"最后两个维度作为矩阵；前置维度是可广播的批次。具体分解、求解或范数公式显示在下方。",steps:["验证矩阵维数与可解条件","对每个批次矩阵执行数值算法","按接口写出矩阵、向量或分解因子"],returns:"shape 由矩阵维度与返回的分解因子决定。",sideEffect:"通常不修改输入。",example:"det([[1,2],[3,4]]) = 1×4−2×3 = −2。"},
    optimizer:{title:`${entry.leaf}：根据梯度与优化器状态更新参数`,what:"读取参数梯度、学习率以及动量/矩估计等状态，算出本步更新量后写回参数。",steps:["读取每个参数的 grad","按当前优化算法更新内部状态并计算 Δθ","原地执行 θ←θ+Δθ"],returns:"step 通常返回 None 或 closure 的损失；state_dict 返回状态。",sideEffect:"修改模型参数及优化器内部状态。",example:"基础 SGD：θ=2、grad=3、lr=0.1 → θ=1.7。"},
    distribution:{title:`${entry.leaf}：构造分布、采样或计算概率量`,what:"分布参数定义 batch_shape 和事件规则；sample/rsample 生成样本，log_prob 把输入代入对数概率密度。",steps:["验证分布参数的取值域","按方法选择采样、统计量或概率计算","按 sample_shape+batch_shape+event_shape 组织结果"],returns:"返回分布对象、样本或概率张量。",sideEffect:"采样会推进随机状态；纯概率计算不修改输入。",example:"Normal(0,1).log_prob(0) = −0.5×log(2π) ≈ −0.91894。"},
    dataloader:{title:`${entry.leaf}：把样本索引、读取和批处理组织成数据流`,what:"采样器决定索引顺序，Dataset 按索引取样，collate_fn 把多个样本合成 batch。",steps:["由 sampler 产生一批索引","读取每个索引对应的样本","拼接/整理样本并交给调用方"],returns:"返回可迭代对象、样本或批次。",sideEffect:"可能推进迭代器、随机采样器和工作进程状态。",example:"4 个标量样本、batch_size=2 → 依次产生两个长度 2 的 batch。"},
    inspection:{title:`${entry.leaf}：读取张量的形状、布局或存储信息`,what:"不进行数值计算，只读取张量元数据或对当前状态做查询。",steps:["定位张量元数据字段","按接口解释维度、步幅、元素数或存储","返回 Python 值、元组或查询对象"],returns:"返回元数据，不返回变换后的张量。",sideEffect:"不修改张量。",example:"numel(shape=(2,3,4)) = 2×3×4 = 24。"},
    device:{title:`${entry.leaf}：转换张量的数据类型、设备或内存格式`,what:"目标属性与当前相同且 copy=False 时可直接返回原对象；否则分配目标存储并转换每个元素。",steps:["解析目标 dtype/device/layout","判断是否需要复制或转换","分配目标存储并逐元素转换"],returns:"返回目标属性的张量。",sideEffect:"通常不修改输入；显式原地转换接口除外。",example:"tensor([1.8]).to(torch.int32) → tensor([1], dtype=int32)。"},
    predicate:{title:`${entry.leaf}：检查能力、类型或当前状态是否满足条件`,what:"读取对象或后端状态，按明确条件产生布尔结论，不做张量数学变换。",steps:["读取待检查对象及环境状态","逐条应用当前接口的判定条件","返回 True/False 或布尔张量"],returns:"返回布尔值或布尔张量。",sideEffect:"不修改被检查对象。",example:`${entry.leaf}(...) 的结果直接表示条件是否成立。`},
    autograd:{title:`${entry.leaf}：沿计算图用链式法则传播梯度`,what:"从输出梯度开始，按计算图反向访问每个运算；把上游梯度乘以局部导数，来自多条路径的梯度相加。",steps:["确定目标输出及上游 grad_output","按反向拓扑顺序计算每个节点的局部梯度","把梯度累加到叶子张量的 .grad 或作为结果返回"],returns:"backward 通常返回 None 并累积 .grad；grad 返回所请求输入的梯度。",sideEffect:"backward 会累积叶子张量的 .grad；需显式清零后再做下一轮。",example:"y=x²，x=3，dy/dx=2x=6；若上游梯度为 2，则传给 x 的梯度为 12。"},
    cross_entropy:{title:`${entry.leaf}：对目标类别计算 LogSoftmax 后取负对数`,what:"先把每个样本的 logits 归一化为对数概率，再取 target 指定类别的负对数；最后按 reduction 汇总。",steps:["对每个样本计算 logsumexp(logits)","目标类损失为 −logit[target]+logsumexp(logits)","按 none、mean 或 sum 汇总样本损失"],returns:"reduction=none 返回逐样本损失；mean/sum 返回标量或较低维结果。",sideEffect:"不修改 logits 和 target。",example:"logits=[0,ln2]、target=1 时，概率=[1/3,2/3]，损失=−ln(2/3)≈0.40547。"},
    mse:{title:`${entry.leaf}：计算预测与目标之差的平方`,what:"prediction 与 target 先广播；每个位置计算 (prediction−target)²，再按 reduction 汇总。",steps:["广播 prediction 与 target","逐位置计算误差 eᵢ=pᵢ−tᵢ 及 eᵢ²","按 none、mean 或 sum 返回"],returns:"none 保留广播 shape；mean/sum 返回归约结果。",sideEffect:"不修改输入。",example:"prediction=[2,4]、target=[1,2]：平方误差=[1,4]，mean=2.5。"},
    distance_loss:{title:`${entry.leaf}：把预测与目标的距离转换成损失`,what:"先逐位置计算差值 d=prediction−target，再按 L1、SmoothL1 或 Huber 的分段规则惩罚误差。",steps:["广播预测与目标并计算差值 d","把 |d| 代入当前损失的绝对值或二次/线性分段公式","按 reduction 汇总"],returns:"none 返回逐位置损失；mean/sum 返回归约结果。",sideEffect:"不修改输入。",example:"L1Loss：prediction=[2,−1]、target=[1,1] → |d|=[1,2]，mean=1.5。"},
    bce:{title:`${entry.leaf}：计算二分类目标的负对数似然`,what:"对目标 y∈{0,1}，同时计算正类项 −y·log(p) 与负类项 −(1−y)·log(1−p)；logits 版本会用数值稳定等价式。",steps:["取得概率 p 或从 logits 得到稳定概率表达","逐位置计算正类项与负类项之和","应用 weight/pos_weight 后按 reduction 汇总"],returns:"none 返回逐位置损失；mean/sum 返回归约结果。",sideEffect:"不修改输入。",example:"p=0.8、y=1 时损失=−log(0.8)≈0.22314。"},
    copy_state:{title:`${entry.leaf}：复制张量、建立视图或改变梯度关联`,what:"根据接口决定是否共享底层存储、是否复制元素，以及结果是否继续连接原计算图。",steps:["读取输入的存储、stride 与梯度状态","按 clone/view/detach 等规则建立结果","保留或切断 autograd 关系"],returns:"返回新张量对象；它可能共享存储，也可能拥有独立存储。",sideEffect:n.endsWith("_")?"原地版本修改当前张量的元数据或状态。":"通常不修改输入；共享视图的后续写入可能反映到同一存储。",example:"clone 独立复制数据；view 共享数据；detach 共享数据但结果不再追踪该计算图。"},
    state:{title:`${entry.leaf}：读取或修改运行时状态`,what:"这类接口处理开关、配置、上下文或对象属性，不对张量元素套用数学公式。",steps:["读取当前状态与调用参数","验证状态切换或属性访问是否合法","返回状态，或把新状态写入指定作用域"],returns:"查询接口返回值；设置接口通常返回 None、旧值或上下文管理器。",sideEffect:/set|enable|disable|clear|reset/.test(n)?"会改变后续调用读取到的状态。":"只读查询不修改状态。",example:`${entry.leaf}(...) 的效果作用于对应模块、线程上下文或进程状态。`},
    grad_mode:{title:`${entry.leaf}：控制是否记录梯度`,what:"在指定代码作用域内开启或关闭 autograd 记录。它不改变已有 Tensor 的 requires_grad，但会影响作用域中新运算是否进入计算图。",steps:["保存进入作用域前的梯度模式","在当前线程切换梯度记录开关","离开作用域时恢复之前的模式"],returns:"通常返回上下文管理器或装饰器；被包裹函数仍返回自己的结果。",sideEffect:"临时改变当前线程的梯度记录模式；作用域结束后恢复。",example:"with torch.no_grad():\n    prediction = model(x)  # 推理时不保存反向图"},
    tensor_bridge:{title:`${entry.leaf}：把 Tensor 转成 Python / NumPy 表示`,what:"从 Tensor 取出 Python 标量、列表或 NumPy 数组。不同接口对元素个数、CPU、梯度状态和共享存储有不同限制。",steps:["检查 Tensor 的元素数、device 与梯度状态","按 item/tolist/numpy 的规则转换表示","返回 Python 或 NumPy 对象"],returns:n==="item"?"返回一个 Python 数字；输入必须恰好只有一个元素。":n==="tolist"?"返回嵌套 Python list；必要时会先移动到 CPU。":"返回 NumPy ndarray；默认情况下可能与 CPU Tensor 共享存储。",sideEffect:"转换本身不修改数值；numpy 共享存储时，后续写入可能同时改变 Tensor 与 ndarray。",example:n==="item"?"torch.tensor([3.5]).item()  # 3.5":"x = torch.tensor([1, 2]); a = x.numpy()  # CPU Tensor 与 a 可能共享内存"},
    module_state:{title:`${entry.leaf}：读取或恢复模块状态`,what:"state_dict 把参数和持久缓冲区整理成有名称的映射；load_state_dict 按名称与 shape 把这些值写回模块。",steps:["遍历模块树并生成完整参数名称","保存或匹配每个参数、缓冲区的 Tensor","返回映射，或报告 missing/unexpected keys"],returns:n.includes("load")?"返回缺失键与多余键等加载结果；strict=True 时不匹配会报错。":"返回按名称组织的浅拷贝字典，值引用模块参数和缓冲区。",sideEffect:n.includes("load")?"会把提供的状态写入模块参数和缓冲区。":"读取状态，不修改模块；返回字典中的 Tensor 仍关联当前状态。",example:"torch.save(model.state_dict(), 'model.pt')\nmodel.load_state_dict(torch.load('model.pt', weights_only=True))"},
    distributed:{title:`${entry.leaf}：让多个进程交换或汇总数据`,what:"参与同一 process group 的 rank 必须按兼容顺序调用集合通信；每个 rank 提供 Tensor，通信后按接口得到汇总、广播或收集结果。",steps:["确认 process group、rank 与 Tensor shape/dtype 兼容","所有参与进程进入同一次通信操作","等待同步完成，或通过 Work 句柄等待异步操作"],returns:"同步调用常返回 None；async_op=True 时返回 Work 句柄。",sideEffect:/all_reduce|broadcast|reduce|scatter/.test(n)?"通常原地修改传入 Tensor；还可能同步或阻塞当前进程。":"会改变通信状态或返回跨进程数据。",example:"dist.all_reduce(x, op=dist.ReduceOp.SUM)  # 每个 rank 的 x 都变为总和"},
    compile:{title:`${entry.leaf}：捕获并优化 Python / PyTorch 执行`,what:"编译器观察一次或多次调用，把可捕获的 Tensor 运算组成图并交给后端优化；遇到不支持的 Python 行为可能 graph break 或回退。",steps:["用示例调用捕获可编译区域","依据输入 shape、dtype 与 Python 条件建立 guards","后端生成优化代码并缓存给后续兼容输入"],returns:"返回优化后的可调用对象，或查询/配置编译状态。",sideEffect:"首次调用可能有明显编译开销；输入变化可能触发重新编译。",example:"optimized = torch.compile(model)\ny = optimized(x)  # 首次调用编译，后续兼容输入复用"},
    export:{title:`${entry.leaf}：把模块和示例输入转换为可验证计算图`,what:"根据示例输入捕获 Tensor 计算，并把输入约束、图签名和参数一起保存为 ExportedProgram，供部署或进一步转换。",steps:["提供模块与代表性 args/kwargs","捕获运算并记录静态或动态 shape 约束","生成可检查、可变换的导出程序"],returns:"通常返回 ExportedProgram 或相关图对象。",sideEffect:"不训练模型；导出会分析执行路径，示例输入必须能覆盖目标行为。",example:"ep = torch.export.export(model, (example_x,))\nprint(ep.graph_module.graph)"},
    sparse:{title:`${entry.leaf}：按稀疏存储规则执行运算`,what:"只保存非零值及其坐标/压缩索引。接口支持的稀疏布局、dtype、设备和梯度能力可能少于稠密 Tensor。",steps:["读取 COO/CSR 等稀疏索引与 values","按接口验证布局和维度兼容性","只遍历相关非零项并生成稀疏或稠密输出"],returns:"返回稀疏或稠密 Tensor，具体布局由接口决定。",sideEffect:n.endsWith("_")?"原地版本修改稀疏 Tensor 的 values 或结构。":"通常不修改输入。",example:"y = torch.sparse.mm(sparse_matrix, dense_matrix)"},
    quantization:{title:`${entry.leaf}：用 scale 和 zero_point 表示低精度数值`,what:"把浮点值映射为整数存储，并保存反量化所需的 scale 与 zero_point；量化误差来自取整和截断。",steps:["读取浮点输入和量化参数","计算 q = round(x / scale) + zero_point","截断到量化 dtype 范围并保存参数"],returns:"返回量化 Tensor、观察器、配置或转换后的模块。",sideEffect:"转换模块时可能替换子模块；量化 Tensor 的数值表示与普通浮点 Tensor 不同。",example:"q = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.quint8)"},
    object:{title:`${entry.leaf}：创建或管理一个 PyTorch 对象`,what:"构造函数保存配置并初始化成员；对象方法读取这些成员与调用输入，完成模块、容器、数据集或运行时对象的职责。",steps:["解析构造或方法参数","创建/定位对象内部成员与资源","执行对象行为并返回结果或句柄"],returns:"构造时返回对象实例；方法按职责返回张量、迭代器、状态或 None。",sideEffect:"构造会创建对象状态；训练模块调用可能更新缓冲区，管理方法可能修改成员。",example:`obj=${entry.name}(...)；随后通过 obj(...) 或对象方法使用已保存配置。`},
    api_behavior:{title:`${entry.leaf}：执行该工具接口定义的具体行为`,what:`它属于“${subcategoryOf(entry)}”，处理 Python 对象、资源或运行时行为，而不是对每个张量元素计算同一个公式。`,steps:["解析调用参数与当前上下文","定位操作需要的对象或后端资源",`执行 ${entry.leaf} 对应行为并生成返回结果`],returns:"在“调用与变量”中按具体签名显示返回对象、句柄、状态或 None。",sideEffect:n.endsWith("_")?"下划线版本通常修改当前对象。":"配置、注册和 I/O 接口可能改变外部状态；查询接口不会。",example:`${entry.name}(...) 的实验会把返回值和发生的状态变化分别列出。`},
  };
  const path=entry.name.toLowerCase();
  if(/all_reduce/.test(path))return {title:"把所有进程的张量归约，并让每个进程得到结果",what:"每个进程提供同 shape 张量；通信后按 SUM、MAX 等 op 逐位置合并。",steps:["每个 rank 提交本地张量","同一坐标的值跨 rank 归约","归约结果写回每个 rank 的张量"],returns:"同步接口完成后各进程持有相同归约结果。",sideEffect:"通常原地修改传入张量。",example:"rank0=[1,2]、rank1=[3,4]，SUM 后两边都为 [4,6]。"};
  if(/all_gather/.test(path))return {title:"收集所有进程的张量到每个进程",what:"每个 rank 提供一份输入，最终每个 rank 按 rank 顺序得到全部输入。",steps:["各 rank 提交本地张量","交换每个 rank 的数据","按 rank 0…world_size−1 写入输出列表"],returns:"每个进程得到长度为 world_size 的结果集合。",sideEffect:"写入预分配输出容器，不改变输入值。",example:"rank0 提交 [1]、rank1 提交 [2]，两边都得到 [[1],[2]]。"};
  if(/broadcast/.test(path))return {title:"把源进程的张量复制给其他进程",what:"src rank 的张量作为唯一数据源，其他 rank 的对应张量被相同数值覆盖。",steps:["读取 src rank 张量","发送到通信组内其他 rank","各 rank 写入相同结果"],returns:"同步版本在通信完成后返回。",sideEffect:"非源 rank 的目标张量会被修改。",example:"src=0 的 [1,2] 广播后，所有 rank 都持有 [1,2]。"};
  if(/reduce_scatter/.test(path))return {title:"先跨进程归约，再把结果分片发送给各进程",what:"它等价于对全部输入做归约后，按 rank 切分结果；每个 rank 只保留自己的分片。",steps:["收集各 rank 输入","逐位置执行 SUM 等归约","把归约结果切片并分发"],returns:"每个 rank 得到一个不同输出分片。",sideEffect:"写入输出张量，不修改输入。",example:"适合把梯度归约与参数分片通信合并。"};
  if(/send|isend|recv|irecv/.test(path))return {title:"在两个进程之间发送或接收张量",what:"send/isend 指定目标 rank，recv/irecv 指定来源；收发两端的元素数量与 dtype 必须匹配。",steps:["确定通信组、源 rank 与目标 rank","发送端传输张量缓冲区","接收端把数据写入目标张量"],returns:/^i|\.i/.test(n)?"异步版本返回可等待的 Work 句柄。":"同步版本在操作完成后返回。",sideEffect:"接收操作会覆盖接收缓冲区。",example:"rank0 send([1,2])，rank1 recv(buf) 后 buf=[1,2]。"};
  if(/compile|dynamo/.test(path))return {title:"捕获 Python/PyTorch 执行并生成优化后的可调用对象",what:"第一次运行根据示例输入追踪可编译区域、建立守卫并编译；后续匹配守卫的输入复用已编译代码。",steps:["执行函数并捕获张量运算图","为 shape、dtype 和 Python 状态建立守卫","交给后端生成代码并缓存"],returns:"返回语义等价的可调用对象或编译结果。",sideEffect:"可能产生编译缓存和首次运行开销，不应改变函数数学语义。",example:"compiled=torch.compile(model)；compiled(x) 首次编译，后续复用。"};
  if(/export|onnx/.test(path))return {title:"把模块与示例输入转换为可部署计算图",what:"根据示例输入记录张量运算、参数和输入约束，并生成目标图表示。",steps:["读取模块与示例输入","捕获前向计算图和参数","验证动态 shape 约束并序列化"],returns:"返回导出图对象，或写出目标模型文件。",sideEffect:"不修改模型参数；文件型接口会写入指定路径。",example:"export(model,(x,)) 得到可检查、可保存的图。"};
  if(/save|checkpoint/.test(path)&&!/saved_tensors/.test(path))return {title:"把对象状态序列化到存储",what:"遍历对象中的张量和元数据，将它们编码后写入文件或分布式存储。",steps:["收集 state_dict 或待保存对象","序列化张量存储与元数据","写入目标位置并完成提交"],returns:"通常返回 None 或保存元数据。",sideEffect:"会创建或覆盖目标存储内容。",example:"torch.save(state_dict, path) 后可用 torch.load(path) 恢复。"};
  if(/load/.test(path))return {title:"从存储中反序列化对象或状态",what:"读取保存的元数据与张量数据，根据 map_location 等参数恢复到目标设备。",steps:["读取文件或检查点元数据","重建张量存储和对象结构","映射设备并返回结果"],returns:"返回恢复后的对象、state_dict 或加载元数据。",sideEffect:"直接 load 通常不改模型；load_state_dict 会把值写入现有对象。",example:"state=torch.load(path,map_location='cpu')。"};
  if(/register/.test(path))return {title:"把函数、实现或钩子登记到注册表",what:"用名称、类型或调度键保存回调，之后对应流程会查找并调用它。",steps:["验证登记名称和适用条件","保存回调或实现到注册表","后续事件发生时按规则调用"],returns:"通常返回注册句柄、装饰器结果或 None。",sideEffect:"会改变进程内注册状态，可能持续到注销或进程结束。",example:`${entry.leaf}(callback) 后，匹配事件会调用 callback。`};
  if(/remove|unregister|destroy/.test(path))return {title:"移除已注册资源或销毁运行时对象",what:"根据句柄或标识找到目标，解除引用并释放关联状态。",steps:["定位目标资源","停止后续使用或通信","从注册表移除并释放状态"],returns:"通常返回 None 或是否成功。",sideEffect:"目标在调用后不可继续按原方式使用。",example:`调用 ${entry.leaf}(handle) 释放对应资源。`};
  if(/set_|enable|disable|config/.test(n))return {title:"读取参数并更新运行时配置",what:"把给定选项写入当前后端、模块或全局上下文，影响之后执行的相关操作。",steps:["读取配置名与新值","检查值是否受支持","写入配置并让后续调用读取它"],returns:"通常返回 None、旧值或上下文管理器。",sideEffect:"会改变后续执行行为；作用域可能是线程、上下文或整个进程。",example:`${entry.leaf}(...) 后，相关后续操作采用新设置。`};
  if(/pad|interpolate|upsample/.test(n))return {title:"按空间参数改变张量边界或采样尺寸",what:"pad 在边界增加指定值或复制边界；interpolate/upsample 根据目标 size 或 scale_factor 计算新采样坐标。",steps:["解析输入空间维和目标尺寸","为每个输出坐标找到输入坐标或边界规则","复制或插值得到输出值"],returns:"返回空间 shape 改变的新张量。",sideEffect:"不修改输入。",example:"interpolate 输入 H×W 到 2H×2W；不同 mode 决定邻近或线性插值。"};
  if(/normalize|norm/.test(n))return {title:"按指定维度计算范数并缩放输入",what:"先在 dim 上求 Lp 范数，再用每个元素除以该范数（并用 eps 避免除零）。",steps:["沿 dim 计算元素绝对值的 p 次幂和","开 p 次方得到范数","用输入除以 max(范数,eps)"],returns:"返回归一化后的张量或范数值，取决于具体函数。",sideEffect:n.endsWith("_")?"原地版本修改输入。":"默认不修改输入。",example:"L2 normalize([3,4]) = [0.6,0.8]。"};
  if(guides[family])return guides[family];
  return {title:`${entry.leaf}：读取参数并执行“${subcategoryOf(entry)}”操作`,what:`${entry.name} 不是逐元素数学算子。它读取调用参数和当前 ${entry.group} 状态，完成名称所指的 ${entry.leaf} 操作。`,steps:[`解析 ${entry.leaf} 所需的位置参数与命名参数`,`定位本次操作涉及的对象、状态或运行时资源`,`执行 ${entry.leaf} 并产生结果；是否写入状态见下方副作用`],returns:`返回类型由 ${entry.leaf} 的用途决定，并在“调用与变量”中显示具体签名。`,sideEffect:n.endsWith("_")?"名称以下划线结尾，通常会原地修改当前对象。":"若该操作涉及注册、配置或 I/O，会修改对应外部状态；纯查询不会修改。",example:`调用 ${entry.name}(...)；页面会把实际返回值和状态变化分开显示。`};
}

function formulaSpecOf(entry: ApiEntry): FormulaSpec {
  /* eslint-disable no-useless-escape -- LaTeX control sequences are intentionally stored in symbol strings. */
  const n = cleanLeaf(entry), family = familyOf(entry);
  if (family === "foreach") { const {base,inPlace,rule}=foreachOperation(entry);return {latex:rule?String.raw`Y^{(k)}_i=${rule.latex}`:String.raw`Y^{(k)}=\operatorname{torch.${base}}\!\left(X^{(k)}\right)`,spoken:rule?`对列表中第 k 个张量的第 i 个元素执行${rule.label}。`:`对列表中每个张量分别执行 torch.${base}。`,explanation:`公式里的 x 就是当前元素 X^(k)_i。k 是张量列表索引，i 是张量内部元素位置。${inPlace?"下划线版本把 Y 写回 X。":"非下划线版本创建新张量列表。"}`,symbols:[{symbol:"k",meaning:"输入列表中第几个 Tensor"},{symbol:"i",meaning:"当前 Tensor 内的元素位置"},{symbol:"x",meaning:"当前输入元素，也就是 X^(k)_i"},{symbol:String.raw`X^{(k)}_i`,meaning:"第 k 个输入张量位置 i 的元素"},{symbol:String.raw`Y^{(k)}_i`,meaning:inPlace?"写回原张量的结果元素":"第 k 个新输出张量的结果元素"}]}; }
  if (family === "matmul") return { latex: String.raw`C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}`, spoken: "输出矩阵第 i 行第 j 列，等于 A 的第 i 行与 B 的第 j 列做点积。", explanation: "固定输出位置 (i,j)，把左矩阵这一行和右矩阵这一列的对应元素相乘，再把 K 个乘积相加。", symbols: [{symbol:String.raw`A\in\mathbb{R}^{M\times K}`,meaning:"左输入矩阵，M 行 K 列"},{symbol:String.raw`B\in\mathbb{R}^{K\times N}`,meaning:"右输入矩阵，K 行 N 列"},{symbol:String.raw`C\in\mathbb{R}^{M\times N}`,meaning:"输出矩阵"},{symbol:"i,j,k",meaning:"输出行、输出列、点积累加索引"}] };
  if (family === "cross_entropy") return { latex: String.raw`\ell_n=-\log\!\left(\frac{\exp(x_{n,y_n})}{\sum_{c=1}^{C}\exp(x_{n,c})}\right),\qquad L=\frac{1}{N}\sum_{n=1}^{N}\ell_n`, spoken: "每个样本先把 logits 转成类别概率，取真实类别的负对数，再对批次求平均。", explanation: "这是目标为类别索引、reduction='mean' 时的官方定义；数值实现通常先减去每行最大值以避免指数溢出。", symbols: [{symbol:"x_{n,c}",meaning:"第 n 个样本对第 c 类的未归一化分数 logit"},{symbol:"y_n",meaning:"第 n 个样本的真实类别索引，范围为 0 到 C−1"},{symbol:"C",meaning:"类别总数"},{symbol:"N",meaning:"批次中的样本数"},{symbol:String.raw`\ell_n,\,L`,meaning:"单样本损失与批次平均损失"}] };
  if (family === "mse") return { latex: String.raw`\ell_n=(x_n-y_n)^2,\qquad L=\frac{1}{N}\sum_{n=1}^{N}\ell_n`, spoken: "逐元素计算预测与目标之差的平方，然后对所有元素取平均。", explanation: "对应 PyTorch MSELoss 在 reduction='mean' 时的定义；N 是输入张量的元素总数，不只表示 batch 大小。", symbols: [{symbol:"x_n",meaning:"第 n 个预测值"},{symbol:"y_n",meaning:"第 n 个目标值，与 x 形状相同"},{symbol:"N",meaning:"参与归约的元素总数"},{symbol:String.raw`\ell_n,\,L`,meaning:"逐元素平方误差与最终平均损失"}] };
  if (family === "distance_loss") return { latex: String.raw`\ell_n=|x_n-y_n|,\qquad L=\frac{1}{N}\sum_{n=1}^{N}\ell_n`, spoken: "逐元素取预测值与目标值之差的绝对值，再取平均。", explanation: "这是 L1Loss 的 mean 归约形式；SmoothL1Loss 和 HuberLoss 会在误差接近零时改用二次函数。", symbols: [{symbol:"x_n",meaning:"第 n 个预测值"},{symbol:"y_n",meaning:"第 n 个目标值"},{symbol:"N",meaning:"元素总数"},{symbol:"L",meaning:"平均绝对误差"}] };
  if (family === "bce") return { latex: String.raw`\ell_n=-\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right],\qquad L=\frac1N\sum_{n=1}^{N}\ell_n`, spoken: "正样本惩罚预测概率太小，负样本惩罚概率太大。", explanation: "p 必须是 0 到 1 之间的概率；若输入是 logits，应使用 BCEWithLogitsLoss 获得更稳定的计算。", symbols: [{symbol:"p_n",meaning:"第 n 个样本预测为正类的概率"},{symbol:String.raw`y_n\in\{0,1\}`,meaning:"二分类目标标签"},{symbol:"N",meaning:"样本或元素数量"},{symbol:"L",meaning:"平均二元交叉熵"}] };
  if (family === "softmax") return { latex: String.raw`\operatorname{Softmax}(x_i)=\frac{\exp(x_i)}{\sum_j\exp(x_j)}`, spoken: "当前元素的指数除以指定维度上所有元素指数之和。", explanation: "Softmax 沿 dim 的每个切片独立计算，输出值位于 [0,1]，并且该切片全部概率之和等于 1。", symbols: [{symbol:"x_i",meaning:"指定 dim 上第 i 个输入值"},{symbol:"j",meaning:"遍历同一切片全部元素的求和索引"},{symbol:String.raw`\exp`,meaning:"自然指数函数"},{symbol:String.raw`\operatorname{Softmax}(x_i)`,meaning:"第 i 个位置的归一化概率"}] };
  if (family === "convolution") {
    const is1d=/1d/.test(n), is3d=/3d/.test(n);
    const latex=is1d?String.raw`Y_{n,o,t}=b_o+\sum_{c=0}^{C_{\mathrm{in}}-1}\sum_{u=0}^{K-1}W_{o,c,u}\,X_{n,c,\,t s+u d-p}`:is3d?String.raw`Y_{n,o,i,j,k}=b_o+\sum_c\sum_u\sum_v\sum_w W_{o,c,u,v,w}\,X_{n,c,\,is+ud-p,\,js+vd-p,\,ks+wd-p}`:String.raw`Y_{n,o,i,j}=b_o+\sum_{c=0}^{C_{\mathrm{in}}-1}\sum_{u=0}^{K_H-1}\sum_{v=0}^{K_W-1}W_{o,c,u,v}\,X_{n,c,\,i s_H+u d_H-p_H,\,j s_W+v d_W-p_W}`;
    return { latex, spoken: "卷积输出的一个位置，是所有输入通道对应窗口与卷积核逐元素乘积的总和，再加偏置。", explanation: "与 PyTorch 官方定义一致：这里实际执行的是互相关，不会先翻转卷积核。滑动位置由 stride、padding 和 dilation 共同决定。", symbols: [{symbol:String.raw`X_{n,c,\dots}`,meaning:"第 n 个样本、第 c 个输入通道中的输入值"},{symbol:String.raw`W_{o,c,\dots}`,meaning:"连接输入通道 c 与输出通道 o 的卷积核权重"},{symbol:String.raw`Y_{n,o,\dots}`,meaning:"第 n 个样本、第 o 个输出通道的结果"},{symbol:"b_o",meaning:"输出通道 o 的偏置"},{symbol:String.raw`s,\,p,\,d`,meaning:"stride 步幅、padding 填充、dilation 膨胀间距"}] };
  }
  if (family === "pooling") return n.includes("avg") ? { latex:String.raw`Y_{n,c,i,j}=\frac{1}{|\Omega_{ij}|}\sum_{(u,v)\in\Omega_{ij}}X_{n,c,u,v}`,spoken:"对当前池化窗口中的全部值求平均。",explanation:"每个通道独立池化；Ωᵢⱼ 是由 kernel_size、stride、padding 决定的当前窗口。",symbols:[{symbol:String.raw`\Omega_{ij}`,meaning:"输出位置 (i,j) 对应的输入窗口"},{symbol:String.raw`|\Omega_{ij}|`,meaning:"窗口内元素数量"},{symbol:String.raw`X,\,Y`,meaning:"输入张量与池化输出"}] } : { latex:String.raw`Y_{n,c,i,j}=\max_{(u,v)\in\Omega_{ij}}X_{n,c,u,v}`,spoken:"从当前池化窗口中选出最大值。",explanation:"MaxPool 不混合通道，只在每个通道的局部窗口内取最大值。",symbols:[{symbol:String.raw`\Omega_{ij}`,meaning:"输出位置 (i,j) 对应的输入窗口"},{symbol:String.raw`n,\,c`,meaning:"样本索引与通道索引"},{symbol:String.raw`X,\,Y`,meaning:"输入张量与池化输出"}] };
  if (family === "linear") return { latex:String.raw`Y=XW^{\mathsf T}+b,\qquad y_j=b_j+\sum_{i=1}^{H_{\mathrm{in}}}x_iW_{j,i}`,spoken:"输入向量乘权重矩阵的转置，再加上偏置。",explanation:"这是 PyTorch Linear 的官方变换形式。最后一维必须等于 in_features，输出最后一维等于 out_features。",symbols:[{symbol:"X",meaning:"输入，最后一维大小为 Hᵢₙ"},{symbol:"W",meaning:"可学习权重，形状为 (Hₒᵤₜ,Hᵢₙ)"},{symbol:"b",meaning:"可选偏置，长度为 Hₒᵤₜ"},{symbol:"Y",meaning:"线性变换输出"}] };
  if (family === "activation") {
    const specs:Record<string,[string,string]>={relu:[String.raw`\operatorname{ReLU}(x)=\max(0,x)`,"负数变为 0，非负数保持不变。"],relu6:[String.raw`\operatorname{ReLU6}(x)=\min(\max(0,x),6)`,"先把负数截为 0，再把大于 6 的值截为 6。"],sigmoid:[String.raw`\sigma(x)=\frac{1}{1+\exp(-x)}`,"把任意实数压缩到 0 与 1 之间。"],tanh:[String.raw`\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}`,"把输入压缩到 −1 与 1 之间。"],gelu:[String.raw`\operatorname{GELU}(x)=x\,\Phi(x)`,"用标准正态分布累计概率平滑地门控输入。"],silu:[String.raw`\operatorname{SiLU}(x)=x\,\sigma(x)`,"输入乘以自身的 Sigmoid 门值。"],leaky_relu:[String.raw`\operatorname{LeakyReLU}(x)=\max(0,x)+\alpha\min(0,x)`,"负半轴保留斜率 α，避免梯度完全为零。"],elu:[String.raw`\operatorname{ELU}(x)=\begin{cases}x,&x>0\\ \alpha(\exp(x)-1),&x\le0\end{cases}`,"正半轴保持线性，负半轴指数饱和。"],selu:[String.raw`\operatorname{SELU}(x)=\lambda\begin{cases}x,&x>0\\ \alpha(\exp(x)-1),&x\le0\end{cases}`,"先做 ELU 形状变换，再乘固定缩放系数 λ。"],softplus:[String.raw`\operatorname{Softplus}(x)=\frac1\beta\log(1+\exp(\beta x))`,"ReLU 的平滑近似。"]};
    const [latex,spoken]=specs[n]??[String.raw`Y_i=\operatorname{${n.replace(/_/g,"\\_")} }(X_i)`,`对位置 i 的输入执行 ${entry.leaf} 明确命名的激活运算。`];return {latex,spoken,explanation:"激活函数逐元素计算，因此默认不改变张量形状。",symbols:[{symbol:"x,\,y",meaning:"单个输入元素与对应输出元素"},{symbol:"\alpha,\,\beta,\,\lambda",meaning:"部分激活函数使用的斜率、平滑或缩放参数"},{symbol:"\Phi",meaning:"标准正态分布的累积分布函数"}]};
  }
  if (family === "reduction") {
    if (/mean/.test(n)) return {latex:String.raw`\mu=\frac{1}{N}\sum_{i=1}^{N}x_i`,spoken:"把指定维度的 N 个元素相加后除以 N。",explanation:"未指定 dim 时归约全部元素；指定 dim 时对每个切片分别计算。",symbols:[{symbol:"x_i",meaning:"归约切片中的第 i 个元素"},{symbol:"N",meaning:"该切片的元素数量"},{symbol:"\mu",meaning:"算术平均值"}]};
    if (/prod/.test(n)) return {latex:String.raw`y=\prod_{i=1}^{N}x_i`,spoken:"把指定维度中的全部元素连续相乘。",explanation:"输出形状取决于 dim 与 keepdim。",symbols:[{symbol:"x_i",meaning:"第 i 个被乘元素"},{symbol:"N",meaning:"归约元素数"},{symbol:"y",meaning:"乘积结果"}]};
    if (/std|var/.test(n)) return {latex:n.includes("std")?String.raw`s=\sqrt{\frac{1}{N-\delta N}\sum_{i=1}^{N}(x_i-\bar{x})^2}`:String.raw`s^2=\frac{1}{N-\delta N}\sum_{i=1}^{N}(x_i-\bar{x})^2`,spoken:"计算每个元素与均值的偏差平方，再按校正后的自由度归一化。",explanation:"PyTorch 用 correction 控制自由度校正；默认 correction=1，即样本标准差/方差。",symbols:[{symbol:"\bar{x}",meaning:"归约切片的均值"},{symbol:"N",meaning:"元素数量"},{symbol:"\delta N",meaning:"correction 自由度校正值"},{symbol:"s,\,s^2",meaning:"标准差与方差"}]};
    const op=/max|amax|argmax/.test(n)?"\max":/min|amin|argmin/.test(n)?"\min":"\sum";return {latex:String.raw`y=${op}_{1\le i\le N}x_i`,spoken:op==="\\sum"?"把指定维度中的全部元素相加。":"在指定维度中寻找极值。",explanation:"argmax/argmin 返回极值位置索引；max/min 返回极值本身。",symbols:[{symbol:"x_i",meaning:"归约切片中的第 i 个元素"},{symbol:"N",meaning:"归约元素数量"},{symbol:"y",meaning:"归约后的值或索引"}]};
  }
  if (family === "binary") { const ops:Record<string,string>={add:"+",sub:"-",mul:"\\cdot",multiply:"\\cdot",div:"/",divide:"/",pow:"^{\,b}",remainder:"\\bmod"};const op=ops[n]??"+";const latex=n==="pow"?String.raw`Y_i=X_i^{\,b}`:String.raw`Y_i=X_i ${op} B_i`;return {latex,spoken:"按广播规则对两个输入逐元素计算。",explanation:"如果两个张量形状不同，PyTorch 会从最后一维开始按 broadcasting 规则对齐；标量会扩展到每个位置。",symbols:[{symbol:"X_i",meaning:"第一个输入在位置 i 的元素"},{symbol:"B_i\text{ 或 }b",meaning:"广播后的第二个输入元素或标量"},{symbol:"Y_i",meaning:"位置 i 的输出"}]}; }
  if (family === "unary") { const rule=foreachUnaryRules[n]??foreachUnaryRules[n.replace(/_$/,"")];const map:Record<string,string>={absolute:String.raw`|x|`,negative:String.raw`-x`,square:String.raw`x^2`};const expression=rule?.latex??map[n]??String.raw`\operatorname{${n.replace(/_/g,"\\_")}}(x)`;return {latex:String.raw`Y_i=${expression}`,spoken:rule?`对输入第 i 个元素执行${rule.label}。`:`对输入第 i 个元素执行 ${entry.leaf} 定义的运算。`,explanation:"公式里的 x 就是输入位置 i 的数值 X_i；输出通常与输入 shape 相同。",symbols:[{symbol:"i",meaning:"张量中的具体元素位置"},{symbol:"x",meaning:"当前输入元素，也就是 X_i"},{symbol:"X_i",meaning:"输入位置 i 的数值"},{symbol:"Y_i",meaning:"该位置计算后的输出数值"}]}; }
  if (family === "autograd") return {latex:String.raw`\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\,\frac{\partial y}{\partial x}`,spoken:"反向传播用链式法则，把上游梯度乘以当前运算的局部导数。",explanation:"对于非标量输出，backward 还需传入与输出同形状的 grad_output，用它计算向量—雅可比积。",symbols:[{symbol:"L",meaning:"最终标量目标或损失"},{symbol:"x,\,y",meaning:"当前计算图节点的输入与输出"},{symbol:"\partial L/\partial y",meaning:"从后续节点传来的上游梯度"},{symbol:"\partial y/\partial x",meaning:"当前运算的局部导数"}]};
  if (family === "reshape") return {latex:String.raw`\prod_{r=1}^{R}d_r=\prod_{s=1}^{S}d'_s,\qquad \operatorname{vec}(Y)=\operatorname{vec}(X)`,spoken:"变形前后元素总数相同，并保持元素的线性顺序。",explanation:"reshape 只重新解释维度分组；若内存布局允许会返回 view，否则可能复制数据。",symbols:[{symbol:"d_r",meaning:"输入 shape 的第 r 个维度"},{symbol:"d'_s",meaning:"目标 shape 的第 s 个维度"},{symbol:"\operatorname{vec}",meaning:"按存储顺序展开为一维序列"}]};
  if (family === "transpose") return {latex:String.raw`Y_{i,j}=X_{j,i}`,spoken:"交换两个维度的索引位置。",explanation:"二维转置把行变成列；高维 transpose 只交换指定的两个维度。",symbols:[{symbol:"X_{j,i}",meaning:"输入在交换后索引处的元素"},{symbol:"Y_{i,j}",meaning:"输出位置 (i,j) 的元素"}]};
  if (family === "fft") return {latex:String.raw`X_k=\sum_{n=0}^{N-1}x_n\exp\!\left(-\mathrm{i}\frac{2\pi kn}{N}\right)`,spoken:"把时域的 N 个采样分解成 N 个复数频率分量。",explanation:"这是离散傅里叶变换；FFT 是计算该公式的高效算法。",symbols:[{symbol:"x_n",meaning:"第 n 个时域采样"},{symbol:"X_k",meaning:"第 k 个频率分量"},{symbol:"N",meaning:"变换长度"},{symbol:"\mathrm{i}",meaning:"虚数单位，i²=−1"}]};
  if (family === "linalg") return {latex:/det/.test(n)?String.raw`\det\!\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc`:/inv|inverse/.test(n)?String.raw`A^{-1}=\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}`:String.raw`\lVert A\rVert_F=\sqrt{\sum_i\sum_j|a_{ij}|^2}`,spoken:"按照当前线性代数接口计算矩阵的行列式、逆矩阵或范数。",explanation:"逆矩阵只在 det(A)≠0 时存在；Frobenius 范数是所有矩阵元素平方和的平方根。",symbols:[{symbol:"A=(a_{ij})",meaning:"输入矩阵"},{symbol:"\det(A)",meaning:"行列式，反映矩阵是否可逆"},{symbol:"A^{-1}",meaning:"满足 AA⁻¹=I 的逆矩阵"},{symbol:"\lVert A\rVert_F",meaning:"Frobenius 范数"}]};
  if (family === "optimizer") return {latex:String.raw`\theta_{t+1}=\theta_t-\eta\,g_t,\qquad g_t=\nabla_{\theta}L(\theta_t)`,spoken:"沿损失梯度的反方向移动参数。",explanation:"这是基础 SGD 更新；带 momentum、Adam 等优化器会先根据历史梯度计算更新量。",symbols:[{symbol:"\theta_t",meaning:"第 t 步的模型参数"},{symbol:"\eta",meaning:"learning rate 学习率"},{symbol:"g_t",meaning:"损失对参数的当前梯度"},{symbol:"L",meaning:"需要最小化的损失函数"}]};
  if (family === "sequence") return n.includes("linspace")?{latex:String.raw`x_i=\mathrm{start}+i\frac{\mathrm{end}-\mathrm{start}}{S-1},\quad i=0,\ldots,S-1`,spoken:"在 start 与 end 之间生成包含两个端点的等间隔序列。",explanation:"steps=S 决定元素个数，因此相邻元素间隔为 (end−start)/(S−1)。",symbols:[{symbol:"S",meaning:"steps，输出元素个数"},{symbol:"i",meaning:"从 0 开始的元素索引"},{symbol:"x_i",meaning:"第 i 个生成值"}]}:{latex:String.raw`x_i=\mathrm{start}+i\cdot\mathrm{step},\quad x_i<\mathrm{end}`,spoken:"从 start 开始反复增加 step，在到达 end 前停止。",explanation:"正 step 时通常不包含 end；负 step 时不等式方向相反。",symbols:[{symbol:"i",meaning:"从 0 开始的索引"},{symbol:"\mathrm{step}",meaning:"相邻元素之差"},{symbol:"x_i",meaning:"第 i 个生成值"}]};
  if (family === "comparison") return {latex:String.raw`Y_i=\mathbf{1}\!\left[X_i\;\mathcal{R}\;B_i\right]`,spoken:"逐元素判断两个值是否满足指定比较关系。",explanation:"关系 R 可以是 =、≠、>、≥、< 或 ≤；结果是同形状的布尔张量。",symbols:[{symbol:"\mathcal R",meaning:"当前接口对应的比较关系"},{symbol:"\mathbf1[\cdot]",meaning:"条件成立返回 True，否则 False"},{symbol:"X_i,B_i",meaning:"广播后位置 i 的两个输入"}]};
  if (family === "selection") return {latex:String.raw`Y_i=\begin{cases}X_i,&C_i=\mathrm{True}\\B_i,&C_i=\mathrm{False}\end{cases}`,spoken:"条件为真选择 input，否则选择 other。",explanation:"condition、input 和 other 会先按 broadcasting 规则对齐。",symbols:[{symbol:"C_i",meaning:"位置 i 的布尔条件"},{symbol:"X_i,B_i",meaning:"两个候选输入"},{symbol:"Y_i",meaning:"选择后的输出"}]};
  if (["squeeze","combine","split","indexing","inspection"].includes(family)) return {latex:String.raw`Y_{\boldsymbol{i}}=X_{\phi(\boldsymbol{i})}`,spoken:"数值保持不变，由索引映射 φ 决定元素在输出中的位置。",explanation:"这类操作主要改变 shape、维度或索引组织；具体映射由 dim、shape、index 等参数决定。",symbols:[{symbol:"\boldsymbol{i}",meaning:"输出张量的多维索引"},{symbol:"\phi",meaning:"当前变形或索引操作定义的位置映射"},{symbol:"X,\,Y",meaning:"输入张量与输出张量"}]};
  if (family === "creation" && n.startsWith("empty")) return {latex:"",spoken:"empty 只分配存储，不定义任何元素值。",explanation:"输出 shape、dtype 和 device 是确定的，但内容未初始化；必须先完整写入再读取，不能把实验中的某组数值当作 PyTorch 保证。",symbols:[]};
  if (family === "creation") return {latex:String.raw`Y_{i_1,\ldots,i_D}=c,\qquad \operatorname{shape}(Y)=(d_1,\ldots,d_D)`,spoken:"按照目标 shape 创建张量，并按接口规则写入元素。",explanation:"zeros、ones、full 分别令 c 为 0、1 或 fill_value；eye 只把主对角线置为 1。",symbols:[{symbol:"D",meaning:"张量维数"},{symbol:"d_1,\ldots,d_D",meaning:"各维长度"},{symbol:"c",meaning:"填充值"}]};
  if (family === "random" || family === "distribution") return {latex:String.raw`X_i\overset{\mathrm{i.i.d.}}{\sim}\mathcal{D}(\boldsymbol{\theta})`,spoken:"每个元素按照指定分布及其参数独立采样。",explanation:"设置随机种子可以复现实验；分布 D 和参数 θ 由当前随机接口决定。",symbols:[{symbol:"X_i",meaning:"第 i 个随机样本"},{symbol:"\mathcal D",meaning:"均匀、正态、伯努利等概率分布"},{symbol:"\boldsymbol\theta",meaning:"分布参数，如均值、标准差或概率"},{symbol:"\mathrm{i.i.d.}",meaning:"独立同分布"}]};
  return {latex:"",spoken:"该接口不是可用一个标量等式说明的数学运算。",explanation:"此处直接展示具体处理步骤、返回值与副作用，不用空泛的函数符号充当数学公式。",symbols:[]};
}

function MathExpression({ latex, spoken, inline = false }: { latex: string; spoken: string; inline?: boolean }) {
  const html = katex.renderToString(latex, { displayMode: !inline, throwOnError: false, strict: false, output: "html" });
  return <span className={inline ? "math-expression math-expression--inline" : "math-expression"} role="math" aria-label={spoken} dangerouslySetInnerHTML={{ __html: html }} />;
}

function FormulaPanel({ entry, compact = false }: { entry: ApiEntry; compact?: boolean }) {
  const formula = formulaSpecOf(entry);
  if(!formula.latex)return <div className={`formula-panel formula-panel--process ${compact ? "formula-panel--compact" : ""}`}><b>此接口没有独立数学公式</b><p>{formula.explanation}</p>{!compact&&<a className="formula-source" href={entry.url} target="_blank" rel="noreferrer">查看官方行为定义 ↗</a>}</div>;
  return <div className={`formula-panel ${compact ? "formula-panel--compact" : ""}`}>
    <MathExpression latex={formula.latex} spoken={formula.spoken} />
    <p className="formula-reading"><b>怎么读：</b>{formula.spoken}</p>
    {!compact && <><p className="formula-note">{formula.explanation}</p><div className="formula-symbols"><span>公式变量</span><dl>{formula.symbols.map((item) => <div key={item.symbol}><dt><MathExpression latex={item.symbol} spoken={item.symbol} inline /></dt><dd>{item.meaning}</dd></div>)}</dl></div><a className="formula-source" href={entry.url} target="_blank" rel="noreferrer">依据 PyTorch 官方定义 · 查看原文 ↗</a></>}
  </div>;
}

function OperationGuidePanel({entry,compact=false}:{entry:ApiEntry;compact?:boolean}){
  const guide=operationGuideOf(entry);
  return <div className={`operation-guide ${compact?"operation-guide--compact":""}`}><h5>{guide.title}</h5><p>{guide.what}</p>{!compact&&<><ol>{guide.steps.map((step,index)=><li key={step}><span>{index+1}</span><p>{step}</p></li>)}</ol><div className="operation-contract"><div><b>返回值</b><p>{guide.returns}</p></div><div><b>副作用</b><p>{guide.sideEffect}</p></div></div><div className="operation-example"><b>看得懂的例子</b><code>{guide.example}</code></div></>}</div>;
}

function parameterInfoOf(name: string) {
  const key = name.toLowerCase();
  const rules: Array<[RegExp, string, string]> = [
    [/^(input|tensor|self|x|a)$/, "要被处理的输入 Tensor 或当前对象", "torch.tensor([[1, 2], [3, 4]])"],
    [/^(other|b)$/, "第二个输入；是否允许广播取决于当前接口", "2 或另一个 Tensor"],
    [/^(k)$/, "每条 dim 切片需要保留、选择或比较的元素个数", "3"],
    [/^(dim|axis|dim0|dim1)$/, "指定操作发生在哪个维度；负数从最后一维倒数", "-1"],
    [/^(keepdim)$/, "归约后是否把该维保留为长度 1", "False"],
    [/^(largest)$/, "True 取最大 k 项；False 取最小 k 项", "True"],
    [/^(sorted)$/, "是否保证选出的元素按数值顺序返回；False 时顺序不受保证", "True"],
    [/^(stable)$/, "相等元素是否保持它们在输入中的先后顺序", "False"],
    [/^(descending)$/, "True 使用降序，False 使用升序", "False"],
    [/^(dtype)$/, "输出元素的数据类型", "torch.float32"],
    [/^(device)$/, "数据或计算所在设备", "'cpu' 或 'cuda'"],
    [/^(out)$/, "可选的预分配输出 Tensor 或 Tensor 元组；结构、shape 和 dtype 必须兼容", "None"],
    [/^(requires_grad)$/, "是否让浮点或复数结果被 autograd 追踪", "False"],
    [/^(size|shape|normalized_shape|output_size)$/, "目标尺寸或 shape；各维含义由当前接口决定", "(2, 3)"],
    [/^(target)$/, "监督学习目标；dtype 与 shape 必须符合损失函数约定", "torch.tensor([1, 0])"],
    [/^(weight)$/, "权重 Tensor 或样本/类别权重", "None"],
    [/^(bias)$/, "是否使用偏置，或要加到输出上的偏置 Tensor", "True"],
    [/^(kernel_size)$/, "滑动窗口或卷积核的空间大小", "3 或 (3, 3)"],
    [/^(stride)$/, "窗口每次移动的步长", "1"],
    [/^(padding)$/, "输入边缘补齐的元素数或策略", "0"],
    [/^(dilation)$/, "卷积核采样点之间的间隔", "1"],
    [/^(groups)$/, "输入与输出通道的分组方式，必须整除相应通道数", "1"],
    [/^(reduction)$/, "如何汇总逐元素损失：none、mean 或 sum", "'mean'"],
    [/^(correction)$/, "方差/标准差的自由度校正值", "1"],
    [/^(generator)$/, "可选随机数生成器，用于控制可复现性", "None"],
    [/^(non_blocking)$/, "条件允许时是否尝试异步设备传输", "False"],
    [/^(copy)$/, "即使属性相同，是否仍强制创建副本", "False"],
    [/^(memory_format)$/, "输出 Tensor 的内存格式", "torch.preserve_format"],
    [/^(index|indices)$/, "整数索引 Tensor；通常要求 dtype=torch.int64", "torch.tensor([2, 0])"],
    [/^(condition|mask)$/, "决定选择或写入位置的布尔条件 Tensor", "torch.tensor([True, False])"],
    [/^(batch_size)$/, "每次迭代组合多少个样本", "32"],
    [/^(shuffle)$/, "每轮是否打乱样本顺序", "True"],
    [/^(lr|learning_rate)$/, "每次参数更新的基础步长", "0.001"],
  ];
  const hit = rules.find(([pattern]) => pattern.test(key));
  return hit ? { meaning: hit[1], sample: hit[2] } : { meaning: `${name} 的具体作用见上方签名与官方说明`, sample: "按任务填写" };
}

function variablesOf(entry: ApiEntry, remote?: RemoteDoc | null): Variable[] {
  const params = (remote?.parameters ?? []).map((raw) => raw.trim()).filter((raw) => raw && !/^(\*|\/)$/.test(raw)).slice(0, 12);
  if (params.length) return params.map((raw) => {
    const name = raw.split(/[=:]/)[0].trim().replace(/^\*+/, "") || "参数";
    const info = parameterInfoOf(name);
    const defaultValue = raw.includes("=") ? raw.split("=").slice(1).join("=").trim() : "";
    return { name, meaning: info.meaning, sample: defaultValue || info.sample, required: !raw.includes("=") && !raw.startsWith("*"), raw };
  });
  const guide = operationGuideOf(entry);
  return [
    { name: entry.type === "method" ? "self" : "input", meaning: "当前接口处理的主要输入；官方签名加载后会显示精确参数", sample: "见 Example 与输入", required: true },
    { name: "result", meaning: guide.returns, sample: "由输入与参数共同决定", required: false },
  ];
}

function cleanSignature(signature: string) {
  return signature
    .replace(/\s*\.\s*/g, ".")
    .replace(/\s*=\s*/g, "=")
    .replace(/\s*\(\s*/g, "(")
    .replace(/\s*,\s*/g, ", ")
    .replace(/\s*\)\s*/g, ")")
    .replace(/\s*#\s*$/, "")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function simulationTierOf(entry: ApiEntry): SimulationTier {
  const family = familyOf(entry), n = cleanLeaf(entry);
  const numericFamilies = new Set(["autograd", "cross_entropy", "mse", "distance_loss", "bce", "softmax", "activation", "unary", "binary", "comparison", "reduction", "reshape", "squeeze", "transpose", "combine", "split", "matmul", "linear", "convolution", "pooling", "sorting", "counting", "selection", "indexing", "sequence", "random", "fft", "linalg", "foreach"]);
  if (family === "creation" && n.startsWith("empty")) return { label: "规则示意", note: "只演示分配规则；真实内容未初始化，本页不会伪造一组确定数值。", numeric: false };
  if (numericFamilies.has(family) || family === "creation") return { label: "数值教学模拟", note: "由网页按页面给出的简化规则计算，用来理解步骤；边界行为仍以真实 PyTorch 为准。", numeric: true };
  if (["copy_state", "device", "predicate", "inspection", "tensor_bridge", "sparse", "quantization"].includes(family)) return { label: "规则示意", note: "展示 shape、存储或类型规则，不会在浏览器里创建真实 Tensor。", numeric: false };
  return { label: "流程 / 状态示意", note: "展示调用顺序与状态变化；没有启动真实 Python、GPU、多进程或编译后端。", numeric: false };
}

function readabilityOf(entry: ApiEntry): ReadabilityContract {
  const family = familyOf(entry), n = cleanLeaf(entry), guide = operationGuideOf(entry);
  const availableQuery = family === "predicate" && n.includes("available");
  const advanced = new Set(["autograd", "convolution", "fft", "linalg", "optimizer", "distribution", "distributed", "compile", "export", "sparse", "quantization"]);
  const beginner = new Set(["unary", "binary", "creation", "reshape", "squeeze", "transpose", "reduction", "inspection", "predicate"]);
  const call = availableQuery
    ? `直接调用 ${entry.name}()，不需要传入 Tensor 或设备名称。`
    : family === "copy_state" && n.includes("detach")
      ? "在 Tensor 上调用 tensor.detach()，这个方法不接收位置参数。"
      : entry.type === "method"
    ? `先有一个对象，再调用 obj.${entry.leaf}(...)；不要写成独立函数。`
    : entry.type === "class"
      ? `先用 ${entry.name}(...) 构造对象，再用对象的方法或 obj(input)。`
      : `直接调用 ${entry.name}(...)；主要数据通常放在第一个参数。`;
  const shapes: Record<string, string> = {
    unary: "通常与 input 完全同 shape。", binary: "等于两个输入广播后的共同 shape。", comparison: "等于广播后的 shape，元素是 bool。",
    reduction: "默认删除 dim；keepdim=True 时该维保留为 1。", sorting: n.includes("topk") ? "指定 dim 的长度变成 k；values 与 indices 同 shape。" : "通常与 input 同 shape。",
    reshape: "变成目标 shape，但元素总数必须不变。", transpose: "只交换维度顺序，元素总数不变。", squeeze: "只增加或删除长度为 1 的维度。",
    creation: "由 size / shape 参数直接决定。", convolution: "batch 与输出通道保留，空间维按 kernel/stride/padding/dilation 公式变化。",
    pooling: "通道数通常不变，空间维按窗口、步长与 padding 缩小。", linear: "最后一维从 in_features 变为 out_features。",
    matmul: "二维时 (M,K) × (K,N) → (M,N)；批量维可能广播。", selection: "等于 condition、input、other 广播后的共同 shape。",
    indexing: "由 index 的 shape 与被索引维共同决定。", inspection: "多数返回 Python 标量、Size 或属性，不一定返回 Tensor。",
    copy_state: "数值 shape 通常不变，但存储和计算图关系可能改变。", predicate: "返回一个 Python bool 或 bool Tensor，取决于具体签名。",
  };
  let autograd = "浮点/复数 Tensor 参与可微运算时通常会被 autograd 追踪；整数、布尔与对象结果通常不可求导。";
  if (family === "copy_state" && n.includes("detach")) autograd = "输出从当前计算图分离，不会把后续梯度传回原图；它仍可能与输入共享存储。";
  else if (family === "grad_mode") autograd = "它直接控制作用域内是否记录新运算；不会回头修改已有 Tensor 的 requires_grad。";
  else if (availableQuery) autograd = "只返回运行时能力状态，不创建 Tensor，也不参与 autograd。";
  else if (["predicate", "inspection", "state", "distributed", "compile", "export", "object", "api_behavior"].includes(family)) autograd = "这个接口本身主要返回状态、对象或流程结果；是否可求导要看它内部调用的 Tensor 运算。";
  const pitfalls: Record<string, string> = {
    sorting: n.includes("topk") ? "k 不能超过 dim 的长度；largest 决定取大还是取小，sorted 只决定返回的 k 项是否排序。" : "dim=-1 表示最后一维；稳定排序与降序是不同概念。",
    creation: n.startsWith("empty") ? "empty 不是 zeros：读取前必须完整写入，任何一次显示出的数值都不受保证。" : "dtype 和 device 不写时会受默认设置与输入数据影响。",
    copy_state: n.includes("detach") ? "detach 不等于复制：修改共享存储可能同时影响原 Tensor；需要独立数据时用 detach().clone()。" : "先确认结果是否共享底层存储，再决定能否安全原地修改。",
    device: "设备移动可能产生复制与同步；non_blocking 只有满足固定内存等条件时才可能异步。",
    predicate: n.includes("available") ? "它只说明当前运行时是否可用，不保证某次分配、编译或算子一定成功。" : "先确认返回的是 Python bool 还是逐元素 bool Tensor。",
    reduction: "dim 可以是负数；空张量、NaN、dtype 与 correction 会改变结果或合法性。",
    reshape: "reshape 可能返回 view，也可能复制；需要保证共享存储时不要只凭 shape 判断。",
    distributed: "所有 rank 必须以兼容顺序进入集合通信，否则可能永久等待。",
    compile: "首次调用有编译开销；输入 shape、dtype 或 Python 分支变化可能触发重新编译或 graph break。",
    tensor_bridge: n === "item" ? "只能用于单元素 Tensor；在 GPU 上调用还可能触发同步。" : "转成 NumPy 前注意 CPU、梯度状态，以及两边是否共享存储。",
  };
  return {
    level: advanced.has(family) ? "进阶" : beginner.has(family) ? "入门" : "常用",
    call,
    input: availableQuery ? "没有数据输入；它读取当前 PyTorch 构建、驱动和硬件环境。" : guide.what,
    output: availableQuery ? "返回一个 Python bool：当前运行时可用为 True，否则为 False。" : guide.returns,
    shape: availableQuery ? "返回 Python bool，没有 Tensor shape。" : shapes[family] ?? "不能只由接口名判断；先看签名和返回值说明，再用小 shape 验证。",
    autograd,
    pitfall: pitfalls[family] ?? (n.endsWith("_") ? "名称以下划线结尾通常表示原地修改；先确认是否会破坏 autograd 所需的中间值。" : "先核对 dtype、device、shape、返回类型和是否修改输入，不要只看函数名猜行为。"),
  };
}

function exampleSpecOf(entry: ApiEntry): ExampleSpec {
  const family = familyOf(entry), n = cleanLeaf(entry);
  if (n === "topk") return { title: "可直接运行：同时看值和原索引", code: "import torch\n\nx = torch.tensor([4, 1, 7, 3, 9, 2])\nvalues, indices = torch.topk(x, k=3, largest=True, sorted=True)\nprint(values)\nprint(indices)", output: "tensor([9, 7, 4])\ntensor([4, 2, 0])", runnable: true };
  if (family === "creation" && n.startsWith("empty")) return { title: "可直接运行：只验证 shape，不断言内容", code: "import torch\n\nx = torch.empty((2, 3))\nprint(x.shape)\nx.fill_(0)          # 完整写入后再读取\nprint(x)", output: "torch.Size([2, 3])\ntensor([[0., 0., 0.],\n        [0., 0., 0.]])", runnable: true };
  if (family === "copy_state" && n.includes("detach")) return { title: "可直接运行：验证梯度与共享存储", code: "import torch\n\nx = torch.tensor([1.0, 2.0], requires_grad=True)\ny = x.detach()\nprint(y.requires_grad)\ny[0] = 9.0\nprint(x)             # detach 默认共享底层存储", output: "False\ntensor([9., 2.], requires_grad=True)", runnable: true };
  if (entry.name === "torch.cuda.is_available") return { title: "可直接运行：查询当前运行时", code: "import torch\n\nprint(torch.cuda.is_available())", output: "True 或 False（取决于机器、驱动和当前 PyTorch 构建）", runnable: true };
  if (entry.name === "torch.compile") return { title: "可直接运行：编译一个小函数", code: "import torch\n\ndef fn(x):\n    return torch.sin(x) + x\n\noptimized_fn = torch.compile(fn)\nprint(optimized_fn(torch.tensor([0.0, 1.0])))", output: "tensor([0.0000, 1.8415])（首次调用可能较慢）", runnable: true };
  if (family === "binary") return { title: "可直接运行：观察 broadcasting", code: `import torch\n\nx = torch.tensor([[1], [2]])\nother = torch.tensor([10, 20])\nprint(torch.${n}(x, other))`, output: "输出 shape 为 (2, 2)；每个位置按该接口的二元规则计算。", runnable: ["add", "sub", "mul", "div", "pow"].includes(n) };
  if (family === "reduction" && ["sum", "mean", "max", "min"].includes(n)) return { title: "可直接运行：明确指定 dim", code: `import torch\n\nx = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nprint(torch.${n}(x, dim=0))`, output: n === "sum" ? "tensor([4., 6.])" : "输出沿 dim=0 汇总；max/min 还可能返回 indices。", runnable: true };
  if (family === "tensor_bridge" && n === "item") return { title: "可直接运行：Tensor 变 Python 标量", code: "import torch\n\nx = torch.tensor([3.5])\nvalue = x.item()\nprint(value, type(value))", output: "3.5 <class 'float'>", runnable: true };
  const call = entry.type === "method" ? `result = tensor.${entry.leaf}(...)` : entry.type === "class" ? `obj = ${entry.name}(...)\nresult = obj(input)` : `result = ${entry.name}(...)`;
  return { title: "调用骨架：请按上方签名补齐参数", code: `import torch\n\n${call}\nprint(result)`, output: "返回类型、shape 与副作用见本页“30 秒读懂”和官方签名。", runnable: false };
}

function searchRank(entry: ApiEntry, keyword: string) {
  const name = entry.name.toLowerCase(), leaf = entry.leaf.toLowerCase();
  if (name === keyword) return 0;
  if (leaf === keyword) return 1;
  if (name.startsWith(keyword)) return 2;
  if (leaf.startsWith(keyword)) return 3;
  return 4;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function conceptOf(entry: ApiEntry) {
  const family=familyOf(entry), n=cleanLeaf(entry);
  if(family==="foreach"){const {base,inPlace,rule}=foreachOperation(entry);return `接收一个 Tensor 列表，对列表中的每个 Tensor ${rule?`逐元素${rule.label}`:`执行 torch.${base}`}${inPlace?"，并把结果原地写回输入张量":"，返回同样长度的新 Tensor 列表"}。每个张量独立处理，不会把不同张量广播成一个共同 shape。`;}
  const familyConcepts:Record<string,string>={
    unary:"对输入的每个元素独立执行名称对应的数学运算，输出位置与输入位置一一对应，shape 通常不变。",binary:"先按广播规则对齐两个输入，再在每个位置执行名称对应的二元运算。",reduction:"沿 dim 取出每个切片，并把切片内多个元素合成求和、均值、极值或索引。",convolution:"让卷积核在输入空间轴上滑动；每个输出位置等于当前窗口与核的逐项乘积之和再加偏置。",pooling:"在每个局部窗口内取最大值或平均值，不混合不同通道。",reshape:"保持元素总数与线性顺序不变，只改变这些元素如何组成维度。",transpose:"交换指定维度的索引含义，使元素移动到交换坐标后的输出位置。",matmul:"把左输入的行与右输入的列做点积，并将每个点积写入相应输出坐标。",sequence:"从起点开始按固定间隔生成数列，直到元素数量或终点条件满足。",creation:"按指定 shape 分配张量，并按 zeros、ones、full、eye 等规则填充每个位置。",random:"按指定概率分布与参数生成每个输出元素。",autograd:"沿计算图反向应用链式法则，把上游梯度传到输入或参数。",sorting:"比较指定维度上的元素并重新排列，同时按接口决定是否返回原位置索引。",selection:"根据布尔条件或索引，从输入中选择对应元素组成输出。",indexing:"按 index 指定的坐标读取或写入元素。",fft:"把采样序列分解为复数频率分量。",optimizer:"用当前梯度和优化器状态计算参数更新量并修改参数。"};
  return familyConcepts[family]??`${entry.leaf} 属于“${subcategoryOf(entry)}”。它${n.endsWith("_")?"可能原地修改输入；":""}按官方签名接收参数并执行对应的 ${entry.group} 操作；下方处理流程会分别说明输入、步骤、返回值与副作用。`;
}

function scenarioOf(entry: ApiEntry) {
  if(familyOf(entry)==="foreach")return foreachOperation(entry).inPlace?"在优化器或批量参数更新中，一次原地处理多个 Tensor，减少逐张量发起运算的开销":"在优化器或批量张量处理中，一次对多个 Tensor 执行相同逐元素运算并获得新列表";
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
  if (/^_foreach_/.test(n)) return "foreach";
  if (/no_grad|enable_grad|inference_mode|set_grad_enabled/.test(n)) return "grad_mode";
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
  if (/clone|copy|detach|contiguous|requires_grad/.test(n)) return "copy_state";
  if (/^(item|tolist|numpy)$/.test(n)) return "tensor_bridge";
  if (/state_dict/.test(n) && entry.group === "神经网络模块") return "module_state";
  if (/^(det|determinant|inverse|inv|solve|matrix_rank|norm)$/.test(n) || entry.group === "线性代数") return "linalg";
  if (entry.group === "优化器") return "optimizer";
  if (entry.group === "概率分布") return "distribution";
  if (entry.group === "数据加载") return "dataloader";
  if (/shape|^size$|^dim$|ndim|numel|element_size/.test(n)) return "inspection";
  if (/^is_|^has_|^can_|enabled|available/.test(n)) return "predicate";
  if (/^(to|cpu|cuda|xpu|float|double|half|long|bool)$/.test(n) || entry.group === "设备与加速") return "device";
  if (entry.group === "分布式训练") return "distributed";
  if (entry.group === "模型导出" || entry.name.startsWith("torch.export")) return "export";
  if (n === "compile" || entry.group === "编译与导出" || /^torch\.(compiler|_dynamo)/.test(entry.name)) return "compile";
  if (entry.group === "稀疏张量") return "sparse";
  if (entry.group === "量化") return "quantization";
  if (/^set_|enable|disable|config/.test(n) || entry.type === "data" || entry.type === "attribute" || entry.type === "property") return "state";
  if (entry.type === "class" || entry.type === "module") return "object";
  return "api_behavior";
}

const familyLabels: Record<string, string> = {
  autograd:"反向传播与计算图",cross_entropy:"多分类损失",mse:"回归损失",distance_loss:"距离与鲁棒损失",bce:"二分类损失",softmax:"概率归一化",
  activation:"非线性激活",convolution:"卷积与转置卷积",pooling:"局部与自适应池化",linear:"全连接与双线性层",matmul:"向量与矩阵乘法",reshape:"重塑与展平",
  squeeze:"增删单维",transpose:"维度交换",combine:"拼接与堆叠",split:"拆分与解绑",reduction:"统计与归约",sorting:"排序与 Top-K",counting:"去重、计数与直方图",
  selection:"条件选择",indexing:"索引、Gather 与 Scatter",binary:"二元逐元素运算",comparison:"比较与近似判断",unary:"一元数学函数",sequence:"等差与等距序列",creation:"张量创建与填充",
  random:"随机采样",fft:"傅里叶变换",linalg:"线性代数",optimizer:"参数优化与调度",distribution:"概率分布与变换",dataloader:"数据集、采样与批处理",inspection:"形状与存储检查",
  copy_state:"复制、视图与梯度状态",tensor_bridge:"Python 与 NumPy 转换",module_state:"模块参数与状态",grad_mode:"梯度记录模式",device:"设备与数据类型",predicate:"能力与状态判断",state:"配置、属性与开关",distributed:"分布式通信",compile:"编译与图捕获",export:"Export 图与动态形状",sparse:"稀疏张量运算",quantization:"量化与低精度",foreach:"Foreach 批量张量运算",object:"类、容器与对象",api_behavior:"其他接口与工具",
};

function subcategoryOf(entry: ApiEntry) {
  const family=familyOf(entry), path=entry.name.toLowerCase(), leaf=cleanLeaf(entry);
  if (!["api_behavior","object","state","predicate"].includes(family)) return familyLabels[family]??"其他接口与工具";
  if (entry.group === "神经网络模块" || entry.group === "神经网络函数") {
    if (/batchnorm|layernorm|groupnorm|instancenorm|rmsnorm|normalize|normalization/.test(path)) return "归一化层";
    if (/dropout|droppath|stochasticdepth/.test(path)) return "随机失活与正则化";
    if (/attention|transformer|multihead/.test(path)) return "注意力与 Transformer";
    if (/\brnn|lstm|gru|recurrent/.test(path)) return "循环神经网络";
    if (/embedding/.test(path)) return "嵌入与查表";
    if (/loss|criterion/.test(path)) return "其他损失函数";
    if (/sequential|modulelist|moduledict|parameterlist|parameterdict|container/.test(path)) return "模块容器";
    if (/padding|pad/.test(path)) return "填充与边界";
    if (/upsampl|interpol|pixelshuffle|fold|unfold/.test(path)) return "采样与空间重排";
    if (/init|weight_norm|spectral_norm|parametr/.test(path)) return "参数初始化与参数化";
    return entry.type === "class" ? "网络层与模块基类" : "其他神经网络函数";
  }
  if (entry.group === "稀疏张量") {
    if (/coo/.test(path)) return "COO 稀疏格式"; if (/csr/.test(path)) return "CSR 稀疏格式"; if (/csc/.test(path)) return "CSC 稀疏格式"; if (/bsr/.test(path)) return "BSR 块稀疏格式"; if (/bsc/.test(path)) return "BSC 块稀疏格式";
    if (/to_sparse|to_dense|convert|compressed/.test(path)) return "稀疏格式转换"; if (/mm|matmul|addmm|sampled_addmm|solve/.test(path)) return "稀疏线性代数"; if (/softmax|sum|log_softmax/.test(path)) return "稀疏归约与概率";
    return "稀疏存储与索引";
  }
  if (entry.group === "分布式训练") {
    if (/all_reduce|all_gather|all_to_all|broadcast|reduce_scatter|gather|scatter/.test(path)) return "集合通信"; if (/send|recv|isend|irecv/.test(path)) return "点对点通信"; if (/processgroup|new_group|init_process_group|destroy_process_group/.test(path)) return "进程组管理";
    if (/fsdp|fully_sharded/.test(path)) return "FSDP 参数分片"; if (/distributed\.checkpoint|dcp/.test(path)) return "分布式检查点"; if (/rpc|rref/.test(path)) return "RPC 与远程引用"; if (/elastic|rendezvous/.test(path)) return "弹性训练与会合"; if (/dtensor|device_mesh/.test(path)) return "DTensor 与设备网格";
    return "分布式状态与工具";
  }
  if (entry.group === "量化") {
    if (/observer/.test(path)) return "量化观察器"; if (/fake_quant/.test(path)) return "伪量化"; if (/quantize_fx|prepare_fx|convert_fx|fx/.test(path)) return "FX 图模式量化"; if (/qat/.test(path)) return "量化感知训练"; if (/backend|config|qconfig/.test(path)) return "量化配置与后端"; if (/quantized/.test(path)) return "量化算子与模块";
    return "量化流程与工具";
  }
  if (entry.group === "优化器") {
    if (/lr_scheduler|scheduler|warmup|anneal|cyclic|plateau/.test(path)) return "学习率调度器"; if (/adam|adagrad|adadelta|rmsprop|radam|nadam/.test(path)) return "自适应优化器"; if (/sgd|asgd|lbfgs|rprop/.test(path)) return "基础与二阶优化器"; return "优化器状态与钩子";
  }
  if (entry.group === "概率分布") {
    if (/constraint/.test(path)) return "参数约束"; if (/transform/.test(path)) return "随机变量变换"; if (/kl_divergence|register_kl/.test(path)) return "KL 散度"; if (/multivariate|wishart|lkj|lowrank/.test(path)) return "多元分布"; if (/categorical|bernoulli|binomial|poisson|geometric/.test(path)) return "离散分布"; return "连续分布与分布基类";
  }
  if (entry.group === "设备与加速") {
    if (/cuda/.test(path)) return "CUDA 设备与流"; if (/xpu/.test(path)) return "XPU 设备"; if (/mps/.test(path)) return "Apple MPS"; if (/backends/.test(path)) return "计算后端配置"; if (/memory/.test(path)) return "显存与内存管理"; return "设备能力与运行时";
  }
  if (["模型导出","编译与导出"].includes(entry.group)) { if (/onnx/.test(path)) return "ONNX 导出"; if (/export/.test(path)) return "Export 图与动态形状"; if (/compile|compiler|dynamo/.test(path)) return "编译与图捕获"; if (/jit|script|trace/.test(path)) return "TorchScript"; return "部署格式与转换"; }
  if (entry.group === "数据加载") { if (/sampler/.test(path)) return "采样器"; if (/dataset|datapipe/.test(path)) return "数据集与 DataPipe"; if (/collate/.test(path)) return "样本合批"; return "DataLoader 与工作进程"; }
  if (entry.group === "自动微分") { if (/forward_ad|dual/.test(path)) return "前向模式自动微分"; if (/functional|jacobian|hessian|jvp|vjp/.test(path)) return "雅可比与高阶导数"; if (/profiler|anomaly/.test(path)) return "梯度诊断"; return "反向传播与计算图"; }
  if (entry.group === "性能分析") return /memory/.test(path)?"内存分析":"CPU 与 GPU 性能分析";
  if (entry.group === "混合精度") return /gradscaler|grad_scaler/.test(path)?"梯度缩放":"自动混合精度";
  if (entry.group === "特殊函数") return /bessel/.test(leaf)?"贝塞尔函数":/gamma|digamma|polygamma/.test(leaf)?"Gamma 函数族":"特殊数学函数";
  if (entry.group === "工具组件") return /benchmark/.test(path)?"基准测试":/cpp_extension/.test(path)?"C++ 与 CUDA 扩展":/hub/.test(path)?"模型 Hub":"实用工具与环境";
  if (entry.group === "核心运算" || entry.group === "Tensor 方法") return familyLabels[family]??(entry.type === "method"?"Tensor 高级方法":"核心工具与元数据");
  return familyLabels[family]??(entry.type === "class"?"类与对象":entry.type === "function"?"函数与运算":"属性、常量与状态");
}

function comparisonOf(entry: ApiEntry): ComparisonSpec | null {
  const family=familyOf(entry);
  if (family === "convolution") return {title:"Conv1d、Conv2d、Conv3d 到底差在哪里？",intro:"可以先理解为：核心区别是卷积核拥有 1 / 2 / 3 个空间维度，并在输入对应的 1 / 2 / 3 个空间轴上滑动；但权重 shape 还始终带有输出通道和输入通道两维。数字 1/2/3 不是输入张量的总维数。",columns:["算法","允许的输入 shape","空间核与滑动轴","完整权重 shape","常用输出遍历顺序","参数写法","输出空间 shape"],rows:[
    {name:"Conv1d",api:"torch.nn.Conv1d",cells:["无 batch：(C_in,L)；有 batch：(N,C_in,L)","1D 核 K_L；只沿 L 滑动","(C_out,C_in/groups,K_L)","L：左 → 右","K、stride、padding、dilation：int 或长度1 tuple","(L_out)"]},
    {name:"Conv2d",api:"torch.nn.Conv2d",cells:["无 batch：(C_in,H,W)；有 batch：(N,C_in,H,W)","2D 核 (K_H,K_W)；沿 H、W 滑动","(C_out,C_in/groups,K_H,K_W)","W：左 → 右（快）→ H：上 → 下（慢）","int 或 (H,W) 二元 tuple","(H_out,W_out)"]},
    {name:"Conv3d",api:"torch.nn.Conv3d",cells:["无 batch：(C_in,D,H,W)；有 batch：(N,C_in,D,H,W)","3D 核 (K_D,K_H,K_W)；沿 D、H、W 滑动","(C_out,C_in/groups,K_D,K_H,K_W)","W：左 → 右（最快）→ H：上 → 下 → D：前 → 后（最慢）","int 或 (D,H,W) 三元 tuple","(D_out,H_out,W_out)"]},
  ],note:"因此不只是‘核的 shape 不同’：输入/输出的空间维数、滑动轴、stride/padding/dilation 的参数维数也一起变化。遍历先后只描述实现或演示如何枚举输出位置；每个输出格独立由对应窗口计算，改变合法的枚举顺序不会改变最终结果。"};
  if (family === "pooling") return {title:"池化算法区别",intro:"池化不混合通道，只在每个通道的局部窗口内归约。",columns:["算法","输入/输出 shape","窗口规则","同一输入 [[1,2],[3,4]]","适合场景"],rows:[
    {name:"MaxPool2d",api:"torch.nn.MaxPool2d",cells:["(N,C,H,W) → (N,C,H_out,W_out)","窗口内取最大值","kernel=2 → [[4]]","保留最显著响应"]},
    {name:"AvgPool2d",api:"torch.nn.AvgPool2d",cells:["形状规则与 MaxPool2d 相同","窗口内求和再除以元素数","kernel=2 → [[2.5]]","平滑与整体统计"]},
    {name:"AdaptiveAvgPool2d",api:"torch.nn.AdaptiveAvgPool2d",cells:["直接指定输出 (H_out,W_out)","自动反推每个输出格覆盖范围","output_size=(1,1) → [[2.5]]","接不同尺寸输入并得到固定尺寸"]},
  ],note:"MaxPool 的 padding 概念上使用负无穷；AvgPool 通常使用零填充，并由 count_include_pad 决定填充值是否进入除数。"};
  if (family === "activation") return {title:"常用激活函数区别",intro:"它们都逐元素计算且不改变 shape，但输出范围、零点附近梯度和饱和特性不同。",columns:["算法","公式特征","输出范围","同一输入 [-1,0,1] 的输出","常见用途"],rows:[
    {name:"ReLU",api:"torch.nn.ReLU",cells:["max(0,x)","[0,+∞)","[0,0,1]","CNN/MLP 默认首选，便宜但负区梯度为0"]},
    {name:"Sigmoid",api:"torch.nn.Sigmoid",cells:["1/(1+e^-x)","(0,1)","[0.269,0.5,0.731]","二分类概率或门控"]},
    {name:"Tanh",api:"torch.nn.Tanh",cells:["双曲正切","(-1,1)","[-0.762,0,0.762]","零中心门控、循环网络"]},
    {name:"GELU",api:"torch.nn.GELU",cells:["x·Φ(x)","约 (-0.17,+∞)","[-0.159,0,0.841]","Transformer 中常用，过渡平滑"]},
  ],note:"同样输入下输出不同并不代表谁绝对更好；要结合网络结构、梯度传播、计算成本与输出语义选择。"};
  if (["mse","distance_loss","cross_entropy","bce"].includes(family)) return {title:"损失函数区别",intro:"回归损失比较连续数值，分类损失比较概率或 logits；输入语义不同，不能只看最终数字大小。",columns:["算法","输入要求","逐项计算","同一回归输入 x=[2,0], y=[1,1]","什么时候用"],rows:[
    {name:"L1Loss",api:"torch.nn.L1Loss",cells:["prediction 与 target 同 shape","|x−y|","mean=1","回归；对异常值相对稳健"]},
    {name:"MSELoss",api:"torch.nn.MSELoss",cells:["prediction 与 target 同 shape","(x−y)²","mean=1","回归；大误差惩罚更强"]},
    {name:"HuberLoss",api:"torch.nn.HuberLoss",cells:["prediction 与 target 同 shape","小误差平方，大误差线性","delta=1 → mean=0.5","兼顾平滑梯度与鲁棒性"]},
    {name:"CrossEntropyLoss",api:"torch.nn.CrossEntropyLoss",cells:["logits: (N,C,...)；target: 类别索引 (N,...)","LogSoftmax + NLLLoss","不适用此回归输入","互斥的 C 类分类"]},
    {name:"BCEWithLogitsLoss",api:"torch.nn.BCEWithLogitsLoss",cells:["logits 与 0/1 target 同 shape","Sigmoid + BCE 的稳定实现","不适用此回归输入","二分类或多标签分类"]},
  ],note:"CrossEntropyLoss 的输入应是未归一化 logits，不需要先做 Softmax；BCEWithLogitsLoss 也不需要先做 Sigmoid。"};
  if (family === "matmul") return {title:"dot、mv、mm、matmul、bmm 区别",intro:"它们都基于乘加，但接受的维数、批处理和广播规则不同。",columns:["算法","输入 shape","输出 shape","同值示例","关键区别"],rows:[
    {name:"dot",api:"torch.dot",cells:["(K) · (K)","标量","[1,2]·[3,4]=11","只接两个1D向量"]},
    {name:"mv",api:"torch.mv",cells:["(M,K) · (K)","(M)","[[1,2],[3,4]]·[1,1]=[3,7]","矩阵乘向量"]},
    {name:"mm",api:"torch.mm",cells:["(M,K) · (K,N)","(M,N)","A·I=A","只接两个2D矩阵，不广播"]},
    {name:"bmm",api:"torch.bmm",cells:["(B,M,K) · (B,K,N)","(B,M,N)","每个 batch 独立相乘","只接3D批矩阵，不广播"]},
    {name:"matmul",api:"torch.matmul",cells:["1D/2D/ND 多种组合","依规则决定","2D时与 mm 相同","通用版本；批维支持广播"]},
  ],note:"核心约束始终是左输入最后一维 K 与右输入倒数第二维 K 相等。"};
  if (family === "reduction") return {title:"归约算法区别",intro:"归约沿指定 dim 把多个元素合成更少元素；keepdim 决定归约维是否保留为长度1。",columns:["算法","对 x=[1,3,2] 的结果","返回内容","是否可导","常见用途"],rows:[
    {name:"sum",api:"torch.sum",cells:["6","元素总和","是","累计量、损失求和"]},{name:"mean",api:"torch.mean",cells:["2","算术平均","是","归一化统计"]},{name:"max",api:"torch.max",cells:["3（指定dim时还可返回index=1）","最大值，可带索引","极值处可导","选择最强响应"]},{name:"argmax",api:"torch.argmax",cells:["1","最大值位置索引","否","分类预测标签"]},{name:"prod",api:"torch.prod",cells:["6","元素乘积","是","联合缩放、维度乘积"]},
  ],note:"不指定 dim 时通常归约全部元素；指定 dim 后，输出 shape 等于输入 shape 删除该维（keepdim=False）。"};
  if (["reshape","squeeze","transpose","combine","split"].includes(family)) return {title:"形状与维度操作区别",intro:"这些接口大多不改变元素值，但改变 shape、元素位置解释或张量数量。",columns:["算法","元素总数","顺序/位置","是否常为 view","例子"],rows:[
    {name:"reshape",api:"torch.reshape",cells:["必须不变","线性顺序不变，重新分组","能则 view，不能则复制","[1,2,3,4] → 2×2"]},{name:"view",api:"torch.Tensor.view",cells:["必须不变","线性顺序不变","要求 stride/内存布局兼容","连续张量 4 → 2×2"]},{name:"flatten",api:"torch.flatten",cells:["不变","合并一段连续维","可能是 view","2×2 → 4"]},{name:"transpose",api:"torch.transpose",cells:["不变","交换两个维的索引","通常是 view","2×3 → 3×2"]},{name:"stack",api:"torch.stack",cells:["各输入相加","新增一个维度","新张量","两个(2,3) → (2,2,3)"]},{name:"cat",api:"torch.cat",cells:["各输入相加","沿已有维连接","新张量","两个(2,3)沿dim0 → (4,3)"]},
  ],note:"reshape/view/flatten 关注同一张量的维度解释；transpose 改变索引映射；stack/cat 组合多个张量。"};
  if (family === "binary") return {title:"二元逐元素运算区别",intro:"都遵循 broadcasting，区别在每个对应位置使用的数学运算。",columns:["算法","x=[2,4], other=2","结果","风险/注意"],rows:[{name:"add",api:"torch.add",cells:["x+2","[4,6]","整数和浮点均常用"]},{name:"sub",api:"torch.sub",cells:["x−2","[0,2]","注意操作数顺序"]},{name:"mul",api:"torch.mul",cells:["x×2","[4,8]","逐元素乘，不是矩阵乘"]},{name:"div",api:"torch.div",cells:["x÷2","[1,2]","整数输入时关注 rounding_mode"]},{name:"pow",api:"torch.pow",cells:["x²","[4,16]","负底数与非整数指数可能产生 NaN"]}],note:"torch.mul 是逐元素乘法；矩阵乘法应使用 matmul/mm/bmm。"};
  return null;
}

const comparisonCatalog: ComparisonCatalogItem[] = [
  {id:"convolution",title:"Conv1d / Conv2d / Conv3d",methods:"Conv1d、Conv2d、Conv3d",difference:"空间核维数、合法输入 shape、滑动轴与 W→H→D 遍历顺序",families:["convolution"]},
  {id:"pooling",title:"最大池化 / 平均池化 / 自适应池化",methods:"MaxPool2d、AvgPool2d、AdaptiveAvgPool2d",difference:"窗口归约规则、padding 语义以及是否直接指定输出尺寸",families:["pooling"]},
  {id:"activation",title:"常用激活函数",methods:"ReLU、Sigmoid、Tanh、GELU",difference:"公式、输出范围、饱和区间、梯度和常用网络场景",families:["activation","softmax"]},
  {id:"loss",title:"回归与分类损失",methods:"L1Loss、MSELoss、HuberLoss、CrossEntropyLoss、BCEWithLogitsLoss",difference:"输入语义、逐项损失、异常值敏感度以及 logits/target 要求",families:["mse","distance_loss","cross_entropy","bce"]},
  {id:"matmul",title:"向量与矩阵乘法",methods:"dot、mv、mm、bmm、matmul",difference:"接受的维数、输出 shape、批处理与广播规则",families:["matmul"]},
  {id:"reduction",title:"统计与归约",methods:"sum、mean、max、argmax、prod",difference:"归约结果、是否返回索引、可导性与 keepdim 影响",families:["reduction"]},
  {id:"shape",title:"形状与维度操作",methods:"reshape、view、flatten、transpose、stack、cat",difference:"元素总数、索引映射、view/复制以及新增维和已有维连接",families:["reshape","squeeze","transpose","combine","split"]},
  {id:"binary",title:"二元逐元素运算",methods:"add、sub、mul、div、pow",difference:"对应位置的运算、broadcasting 与数值风险",families:["binary"]},
];

function comparisonCatalogOf(entry: ApiEntry) {
  const family=familyOf(entry);
  return comparisonCatalog.find(item=>item.families.includes(family))??null;
}

function ComparisonDirectory({ selected, onOpen, onClose }: { selected: ApiEntry; onOpen: (entry: ApiEntry, tab: DetailTab) => void; onClose: () => void }) {
  const current=comparisonCatalogOf(selected);
  return <section className="comparison-directory" id="comparison-directory">
    <header><div><small>二级分类 · 独立资料</small><h3>相似方法对比索引</h3><p>先找“哪些方法相似”，再进入区别表或对应函数模拟，不必先打开某一个函数页。</p></div><button type="button" onClick={onClose}>返回接口目录</button></header>
    <div className="comparison-directory__table"><table><thead><tr><th>相似方法组</th><th>包含哪些方法</th><th>介绍 / 主要区分</th><th>区别表 / 模拟表格在哪</th></tr></thead><tbody>{comparisonCatalog.map(item=>{const target=entries.find(entry=>item.families.includes(familyOf(entry))&&comparisonOf(entry));return <tr className={current?.id===item.id?"is-current":""} key={item.id}><th>{item.title}</th><td>{item.methods}</td><td>{item.difference}</td><td>{target?<div className="comparison-directory__links"><button type="button" onClick={()=>onOpen(target,"compare")}>打开区别表</button><button type="button" onClick={()=>onOpen(target,"result")}>进入对应模拟</button></div>:"整理中"}</td></tr>;})}</tbody></table></div>
  </section>;
}

function AlgorithmComparison({ entry, onSelect }: { entry: ApiEntry; onSelect: (entry: ApiEntry) => void }) {
  const comparison=comparisonOf(entry); if(!comparison)return null;
  return <section className="deep-card deep-card--wide comparison-card"><small>扩展 · 同类算法区别</small><header><div><h4>{comparison.title}</h4><p>{comparison.intro}</p></div><span>{subcategoryOf(entry)}</span></header><div className="comparison-scroll"><table><thead><tr>{comparison.columns.map(column=><th key={column}>{column}</th>)}</tr></thead><tbody>{comparison.rows.map(row=>{const target=row.api?entries.find(item=>item.name===row.api):undefined;return <tr className={cleanLeaf(entry)===row.name.toLowerCase()?"is-current":""} key={row.name}><th><b>{row.name}</b>{target&&<button type="button" onClick={()=>onSelect(target)}>进入实验 →</button>}</th>{row.cells.map((cell,index)=><td key={`${row.name}-${index}`}>{cell}</td>)}</tr>;})}</tbody></table></div>{familyOf(entry)==="convolution"&&<ConvolutionDimensionLab key={entry.name} entry={entry}/>}<p className="comparison-note"><b>选择要点：</b>{comparison.note}</p></section>;
}

const convDimensionDemos: ConvDimensionDemo[] = [
  {dimension:1,name:"Conv1d",axisNames:["L"],inputShape:[7],input:[2,1,3,0,4,2,5],kernelShape:[3],kernel:[1,-1,2],stride:[2],padding:[1],dilation:[1],description:"长度轴 L 上从左向右。这里 stride=2，所以窗口左边界每次跨 2 格；padding=1 让首尾窗口包含虚拟 0。"},
  {dimension:2,name:"Conv2d",axisNames:["H","W"],inputShape:[4,5],input:[1,2,0,1,3,4,1,2,5,0,2,3,1,0,4,1,0,2,3,5],kernelShape:[2,3],kernel:[1,0,-1,2,0,-2],stride:[1,2],padding:[0,0],dilation:[1,1],description:"先沿 W 从左到右扫完一行，再沿 H 向下换行。这里 stride=(1,2)，所以横向每步跨 2 列、纵向每步跨 1 行。"},
  {dimension:3,name:"Conv3d",axisNames:["D","H","W"],inputShape:[3,3,4],input:[1,0,2,1,2,1,0,3,1,2,1,0,0,2,1,1,1,0,2,2,3,1,0,1,2,1,0,2,0,3,1,0,1,0,2,3],kernelShape:[2,2,2],kernel:[1,0,0,-1,0,1,-1,0],stride:[1,1,2],padding:[0,0,0],dilation:[1,1,1],description:"W 是最快变化轴：先向右；一行扫完后 H 向下；一层扫完后 D 向后切到下一深度层。stride=(1,1,2)。"},
];

function coordinatesOf(shape:number[]):number[][] { if(!shape.length)return [[]];return Array.from({length:shape[0]},(_,i)=>coordinatesOf(shape.slice(1)).map(rest=>[i,...rest])).flat(); }
function coordFlatIndex(shape:number[],coord:number[]){let index=0;for(let i=0;i<shape.length;i++)index=index*shape[i]+coord[i];return index;}
function coordLabel(coord:number[],axes:string[]){return coord.map((value,index)=>`${axes[index].toLowerCase()}=${value}`).join(",");}
function calculateConvDemo(demo:ConvDimensionDemo){
  const outputShape=demo.inputShape.map((size,axis)=>Math.floor((size+2*demo.padding[axis]-demo.dilation[axis]*(demo.kernelShape[axis]-1)-1)/demo.stride[axis]+1));
  const positions:ConvDemoPosition[]=coordinatesOf(outputShape).map(outputCoord=>{const origin=outputCoord.map((value,axis)=>value*demo.stride[axis]-demo.padding[axis]);const terms=coordinatesOf(demo.kernelShape).map(kernelCoord=>{const inputCoord=kernelCoord.map((value,axis)=>origin[axis]+value*demo.dilation[axis]);const inside=inputCoord.every((value,axis)=>value>=0&&value<demo.inputShape[axis]);const inputValue=inside?demo.input[coordFlatIndex(demo.inputShape,inputCoord)]:0;const kernelValue=demo.kernel[coordFlatIndex(demo.kernelShape,kernelCoord)];return {inputCoord,kernelCoord,inputValue,kernelValue,product:inputValue*kernelValue,inside};});return {outputCoord,origin,terms,value:terms.reduce((sum,term)=>sum+term.product,0)};});
  return {outputShape,positions};
}

function ConvGrid({shape,values,axes,activeCoords=[],outputCoord,includePadding=false,padding=[]}:{shape:number[];values:number[];axes:string[];activeCoords?:number[][];outputCoord?:number[];includePadding?:boolean;padding?:number[]}){
  const isActive=(coord:number[])=>activeCoords.some(candidate=>candidate.length===coord.length&&candidate.every((value,index)=>value===coord[index]));
  if(shape.length===1){const start=includePadding?-padding[0]:0,end=shape[0]+(includePadding?padding[0]:0);return <div className="conv-vector">{Array.from({length:end-start},(_,offset)=>start+offset).map(index=>{const outside=index<0||index>=shape[0],active=isActive([index]),value=outside?0:values[index];return <div className={`${active?"is-active":""} ${outside?"is-padding":""}`} key={index}><small>{axes[0].toLowerCase()}={index}</small><b>{value}</b></div>;})}</div>}
  const depth=shape.length===3?shape[0]:1,rows=shape.length===3?shape[1]:shape[0],cols=shape.length===3?shape[2]:shape[1];
  return <div className="conv-planes">{Array.from({length:depth},(_,layer)=><section key={layer}><small>{shape.length===3?`${axes[0]}=${layer}`:"单个空间平面"}</small><div className="conv-matrix" style={{gridTemplateColumns:`repeat(${cols}, minmax(34px,1fr))`}}>{Array.from({length:rows*cols},(_,flatIndex)=>{const row=Math.floor(flatIndex/cols),col=flatIndex%cols,coord=shape.length===3?[layer,row,col]:[row,col],active=isActive(coord),selected=outputCoord&&outputCoord.length===coord.length&&outputCoord.every((value,index)=>value===coord[index]);return <div className={`${active?"is-active":""} ${selected?"is-output":""}`} key={coord.join("-")}><small>{coord.join(",")}</small><b>{values[coordFlatIndex(shape,coord)]}</b></div>;})}</div></section>)}</div>;
}

function ConvolutionDimensionLab({entry}:{entry:ApiEntry}){
  const requested=cleanLeaf(entry).includes("3d")?3:cleanLeaf(entry).includes("2d")?2:1;
  const [dimension,setDimension]=useState<1|2|3>(requested);const [step,setStep]=useState(0);
  const demo=convDimensionDemos.find(item=>item.dimension===dimension)??convDimensionDemos[0],calculation=calculateConvDemo(demo),current=calculation.positions[Math.min(step,calculation.positions.length-1)],next=calculation.positions[Math.min(step+1,calculation.positions.length-1)];
  const outputValues=calculation.positions.map(position=>position.value),activeInput=current.terms.filter(term=>term.inside).map(term=>term.inputCoord),activeKernel=current.terms.map(term=>term.kernelCoord);
  const axisCalculation=demo.axisNames.map((axis,index)=>`${axis}_out = floor((${demo.inputShape[index]} + 2×${demo.padding[index]} − ${demo.dilation[index]}×(${demo.kernelShape[index]}−1) − 1) / ${demo.stride[index]} + 1) = ${calculation.outputShape[index]}`);
  return <div className="conv-dimension-lab"><header><div><small>交互式滑窗演示</small><h5>窗口到底沿哪个方向移动？</h5></div><div className="conv-dimension-tabs" role="group" aria-label="选择卷积维度">{convDimensionDemos.map(item=><button className={dimension===item.dimension?"active":""} aria-pressed={dimension===item.dimension} key={item.name} onClick={()=>{setDimension(item.dimension);setStep(0);}}>{item.name}</button>)}</div></header>
    <div className="conv-core-difference"><strong>一句话结论</strong><p><b>是，但要说完整：</b>Conv{dimension}d 的核有 {dimension} 个空间维度，并沿输入的 {demo.axisNames.join("、")} 共 {dimension} 个空间轴滑动。完整权重还包含 <code>C_out</code> 与 <code>C_in/groups</code>，所以它不是单纯的 {dimension} 维张量。</p><p><b>遍历顺序：</b>{dimension===1?"L 从左到右。":dimension===2?"W 先从左到右，扫完一行后 H 再向下，因此是 W → H。":"W 先从左到右，一行结束后 H 向下，一层结束后 D 向后，因此是 W → H → D。"} 这只是枚举输出格的常用顺序，不参与卷积公式，也不会影响最终输出值。</p></div>
    <div className="conv-axis-order"><b>实际扫描顺序</b><span>{dimension===1?"L：左 → 右":dimension===2?"W：左 → 右（内层循环） → H：上 → 下（换行）":"W：左 → 右（最快） → H：上 → 下 → D：前 → 后（最慢）"}</span><p>{demo.description}</p></div>
    <div className="conv-shape-summary"><div><small>示例输入</small><b>(N=1, C_in=1, {demo.axisNames.map((axis,index)=>`${axis}=${demo.inputShape[index]}`).join(", ")})</b></div><div><small>卷积核</small><b>(C_out=1, C_in=1, {demo.kernelShape.join("×")})</b></div><div><small>参数</small><b>stride={JSON.stringify(demo.stride)} · padding={JSON.stringify(demo.padding)} · dilation={JSON.stringify(demo.dilation)}</b></div><div><small>输出</small><b>(1, 1, {calculation.outputShape.join(", ")}) · 共 {calculation.positions.length} 个位置</b></div></div>
    <div className="conv-step-picker"><button disabled={step===0} onClick={()=>setStep(value=>Math.max(0,value-1))}>← 上一位置</button><div>{calculation.positions.map((position,index)=><button className={step===index?"active":""} aria-label={`查看输出位置 ${position.outputCoord.join(",")}`} key={position.outputCoord.join("-")} onClick={()=>setStep(index)}>Y[{position.outputCoord.join(",")}]</button>)}</div><button disabled={step===calculation.positions.length-1} onClick={()=>setStep(value=>Math.min(calculation.positions.length-1,value+1))}>下一位置 →</button></div>
    <div className="conv-current-move"><b>第 {step+1}/{calculation.positions.length} 步：计算 Y[{current.outputCoord.join(",")}]</b><span>窗口左上/前起点：({coordLabel(current.origin,demo.axisNames)})</span>{step<calculation.positions.length-1&&<span>下一步：Y[{next.outputCoord.join(",")}]，起点变为 ({coordLabel(next.origin,demo.axisNames)})</span>}</div>
    <div className="conv-demo-flow"><section><h6>输入 X · 橙色是当前窗口</h6><ConvGrid shape={demo.inputShape} values={demo.input} axes={demo.axisNames} activeCoords={activeInput} includePadding={dimension===1} padding={demo.padding}/></section><div className="conv-demo-arrow"><b>×</b><span>逐项乘加</span></div><section><h6>卷积核 W · 对应权重</h6><ConvGrid shape={demo.kernelShape} values={demo.kernel} axes={demo.axisNames.map(axis=>`K${axis}`)} activeCoords={activeKernel}/></section><div className="conv-demo-arrow"><b>=</b><span>写入当前位置</span></div><section><h6>输出 Y · 绿色是已计算位置</h6><ConvGrid shape={calculation.outputShape} values={outputValues} axes={demo.axisNames} outputCoord={current.outputCoord}/></section></div>
    <div className="conv-equation"><small>当前位置完整算式（坐标明确对应上方格子）</small><div>{current.terms.map((term,index)=><span className={!term.inside?"is-padding":""} key={`${term.inputCoord.join("-")}-${index}`}><b>X[{term.inputCoord.join(",")}]={term.inputValue}</b> × W[{term.kernelCoord.join(",")}]={term.kernelValue} = {term.product}{!term.inside&&<i>padding</i>}</span>)}</div><strong>{current.terms.map(term=>term.product).join(" + ")} = {current.value}</strong></div>
    <div className="conv-output-formula"><small>输出尺寸逐轴代入</small>{axisCalculation.map(line=><code key={line}>{line}</code>)}</div>
    <div className="conv-input-limits"><h5>输入 shape 是否有限制？有，但空间大小不是固定值</h5><div><section><b>① 张量维数必须正确</b><p>Conv1d 接 2D/3D，Conv2d 接 3D/4D，Conv3d 接 4D/5D；少一维表示省略 batch，不能省略 channel。</p></section><section><b>② 通道必须对上</b><p>input.shape 的 C 必须等于 in_channels；in_channels 和 out_channels 都必须能被 groups 整除。</p></section><section><b>③ 每个空间轴必须放得下有效卷积核</b><p>S_in + 2p ≥ d×(K−1)+1。右侧是有效核尺寸；不满足时该轴无法产生合法输出。</p></section><section><b>④ 输出尺寸由参数决定</b><p>S_out=floor((S_in+2p−d×(K−1)−1)/s+1)，每个 L / H / W / D 轴分别计算。</p></section><section><b>⑤ int 与 tuple 的含义不同</b><p>Conv2d 中 kernel_size=3 表示 (3,3)，而 (3,5) 表示高3、宽5；Conv3d 的 tuple 顺序是 (D,H,W)。</p></section><section><b>⑥ same 不是任意 stride</b><p>padding=&quot;same&quot; 会保持空间 shape，但官方接口要求 stride=1；padding=&quot;valid&quot; 等价于不填充。</p></section></div></div>
  </div>;
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

function TensorMatrix({ values, columns, highlights = [], colorHighlights = {} }: { values: Array<number | boolean>; columns: number; highlights?: number[]; colorHighlights?: Record<number, number> }) {
  const visibleColumns = Math.max(1, Math.min(columns, 8));
  return <div className="tensor-matrix" style={{ gridTemplateColumns: `repeat(${visibleColumns}, minmax(31px, 1fr))` }}>
    {values.slice(0, 64).map((value, index) => <span className={`${highlights.includes(index) ? "is-active" : ""} ${colorHighlights[index] !== undefined ? `term-color-${colorHighlights[index] % 8}` : ""}`} key={`${index}-${String(value)}`}>{cellText(value)}</span>)}
    {values.length > 64 && <span className="tensor-more">+{values.length - 64}</span>}
  </div>;
}

function TensorVisual({ item, highlights = [], colorHighlights = {} }: { item: TensorItem; highlights?: number[]; colorHighlights?: Record<number, number> }) {
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
  const selectedColorHighlights = Object.fromEntries(Object.entries(colorHighlights).filter(([index]) => Number(index) >= selectedStart && Number(index) < selectedStart + planeSize).map(([index, color]) => [Number(index) - selectedStart, color]));
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
        <div className="tensor-plane tensor-plane--selected"><TensorMatrix values={values.slice(selectedStart, selectedStart + planeSize)} columns={columns} highlights={selectedHighlights} colorHighlights={selectedColorHighlights} /></div>
      </div>
      {planeCount > visiblePlaneCount && <span className="tensor-depth-more">当前展示前 {visiblePlaneCount} 层，共 {planeCount} 层</span>}
    </div> : <TensorMatrix values={values} columns={columns} highlights={highlights} colorHighlights={colorHighlights} />}
  </article>;
}

function positionDetail(entry: ApiEntry, spec: Record<string, unknown>, sim: Simulation, step: number): PositionDetail | null {
  const family=familyOf(entry), n=cleanLeaf(entry);
  if (family === "convolution" && n.includes("1d")) {
    const xRaw=spec.input as number[][]|number[], kRaw=spec.kernel as number[][]|number[], x=Array.isArray(xRaw?.[0])?xRaw as number[][]:[xRaw as number[]], k=Array.isArray(kRaw?.[0])?kRaw as number[][]:[kRaw as number[]];
    const length=x[0]?.length??0,kernelSize=k[0]?.length??0,stride=Number(spec.stride??1),padding=Number(spec.padding??0),dilation=Number(spec.dilation??1),t=step;
    const terms:PositionTerm[]=[];x.forEach((channel,c)=>k[c].forEach((weight,u)=>{const position=t*stride+u*dilation-padding,inputValue=position>=0&&position<length?Number(channel[position]):0;terms.push({inputKey:"input",inputIndex:c*length+Math.max(0,position),inputCoord:`X[n=0, c=${c}, t=${position}]`,inputValue,parameterKey:"kernel",parameterIndex:c*kernelSize+u,parameterCoord:`W[o=0, c=${c}, u=${u}]`,parameterValue:Number(weight),result:Number(weight)*inputValue,operator:"×"});}));
    return {title:`Conv1d 输出位置 t=${t}`,outputCoord:`Y[n=0, o=0, t=${t}]`,outputValue:Number((sim.value as number[])?.[t]),terms,bias:Number(spec.bias??0),rule:`窗口起点 = ${t}×stride(${stride}) − padding(${padding})；核内位置还要乘 dilation(${dilation})`,aggregation:"sum"};
  }
  if (family === "convolution") {
    const rawX=spec.input,rawK=spec.kernel,x=shapeOf(rawX).length===3?rawX as number[][][]:[rawX as number[][]],k=shapeOf(rawK).length===3?rawK as number[][][]:[rawK as number[][]],height=x[0].length,width=x[0][0].length,kH=k[0].length,kW=k[0][0].length,stride=Number(spec.stride??1),out=sim.value as number[][],outCols=out[0]?.length??1,i=Math.floor(step/outCols),j=step%outCols;
    const terms:PositionTerm[]=[];k.forEach((plane,c)=>plane.forEach((row,u)=>row.forEach((weight,v)=>{const inputRow=i*stride+u,inputCol=j*stride+v,inputValue=Number(x[c][inputRow][inputCol]);terms.push({inputKey:"input",inputIndex:c*height*width+inputRow*width+inputCol,inputCoord:`X[n=0, c=${c}, h=${inputRow}, w=${inputCol}]`,inputValue,parameterKey:"kernel",parameterIndex:c*kH*kW+u*kW+v,parameterCoord:`W[o=0, c=${c}, u=${u}, v=${v}]`,parameterValue:Number(weight),result:Number(weight)*inputValue,operator:"×"});})));
    return {title:`Conv2d 输出位置 (h=${i}, w=${j})`,outputCoord:`Y[n=0, o=0, h=${i}, w=${j}]`,outputValue:Number(out[i]?.[j]),terms,bias:Number(spec.bias??0),rule:`输入窗口左上角 = (${i}×stride, ${j}×stride) = (${i*stride}, ${j*stride})`,aggregation:"sum"};
  }
  if (family === "pooling") {
    const x=spec.input as number[][],size=Number(spec.kernel_size??2),stride=Number(spec.stride??size),out=sim.value as number[][],outCols=out[0]?.length??1,i=Math.floor(step/outCols),j=step%outCols;
    const terms:PositionTerm[]=[];for(let u=0;u<size;u++)for(let v=0;v<size;v++){const row=i*stride+u,col=j*stride+v,value=Number(x[row][col]);terms.push({inputKey:"input",inputIndex:row*x[0].length+col,inputCoord:`X[n=0, c=0, h=${row}, w=${col}]`,inputValue:value,result:value,operator:"读取"});}
    const aggregation=cleanLeaf(entry).includes("avg")?"mean":"max";return {title:`${aggregation==="mean"?"平均":"最大"}池化输出位置 (h=${i}, w=${j})`,outputCoord:`Y[n=0, c=0, h=${i}, w=${j}]`,outputValue:Number(out[i]?.[j]),terms,rule:`窗口左上角 = (${i}×${stride}, ${j}×${stride})，窗口大小 = ${size}×${size}`,aggregation};
  }
  if (family === "matmul") {
    const a=spec.input as number[][],b=spec.other as number[][],out=sim.value as number[][],cols=b[0].length,i=Math.floor(step/cols),j=step%cols;
    const terms=a[i].map((value,k)=>({inputKey:"input",inputIndex:i*a[0].length+k,inputCoord:`A[i=${i}, k=${k}]`,inputValue:Number(value),parameterKey:"other",parameterIndex:k*cols+j,parameterCoord:`B[k=${k}, j=${j}]`,parameterValue:Number(b[k][j]),result:Number(value)*Number(b[k][j]),operator:"×" as const}));
    return {title:`矩阵乘法输出位置 (${i}, ${j})`,outputCoord:`C[i=${i}, j=${j}]`,outputValue:Number(out[i][j]),terms,rule:`固定 A 的第 ${i} 行和 B 的第 ${j} 列，按 k 对齐后逐项相乘并求和`,aggregation:"sum"};
  }
  if (family === "linear") {
    const input=(spec.input as number[]).map(Number),weight=spec.weight as number[][],bias=(spec.bias as number[]).map(Number),output=sim.value as number[],j=step%weight.length;
    const terms=input.map((value,i)=>({inputKey:"input",inputIndex:i,inputCoord:`x[i=${i}]`,inputValue:value,parameterKey:"weight",parameterIndex:j*input.length+i,parameterCoord:`W[j=${j}, i=${i}]`,parameterValue:Number(weight[j][i]),result:value*Number(weight[j][i]),operator:"×" as const}));
    return {title:`Linear 输出神经元 j=${j}`,outputCoord:`y[j=${j}]`,outputValue:Number(output[j]),terms,bias:bias[j],rule:`取权重矩阵第 ${j} 行，与输入向量逐项相乘并求和`,aggregation:"sum"};
  }
  if (["activation","unary","binary","comparison","selection"].includes(family)) {
    const inputValues=visualLeaves(spec.input),otherValues=visualLeaves(spec.other),outputValues=visualLeaves(sim.value),index=step%Math.max(1,inputValues.length),other=otherValues.length===1?otherValues[0]:otherValues[index];
    const operators:Record<string,string>={add:"+",sub:"−",mul:"×",multiply:"×",div:"÷",divide:"÷",pow:"幂",remainder:"取余",eq:"=",ne:"≠",gt:">",ge:"≥",lt:"<",le:"≤"};
    return {title:`逐元素输出位置 i=${index}`,outputCoord:`Y[i=${index}]`,outputValue:outputValues[index],terms:[{inputKey:"input",inputIndex:index,inputCoord:`X[i=${index}]`,inputValue:Number(inputValues[index]),parameterKey:other!==undefined?"other":undefined,parameterIndex:otherValues.length>1?index:0,parameterCoord:other!==undefined?`B[i=${otherValues.length>1?index:"broadcast"}]`:undefined,parameterValue:other!==undefined?Number(other):undefined,result:Number(outputValues[index]),operator:operators[n]??entry.leaf}],rule:"逐元素算子只读取相同位置的输入；标量 other 会广播到全部位置",aggregation:"direct"};
  }
  return null;
}

function termColorMap(detail: PositionDetail | null, key: string) {
  if (!detail) return {};
  const map:Record<number,number>={};detail.terms.forEach((term,index)=>{if(term.inputKey===key)map[term.inputIndex]=index;if(term.parameterKey===key&&term.parameterIndex!==undefined)map[term.parameterIndex]=index;});return map;
}

function PositionCalculationDialog({ detail, onClose }: { detail: PositionDetail; onClose: () => void }) {
  const sum=detail.terms.reduce((total,term)=>total+Number(term.result),0),aggregate=detail.aggregation==="max"?Math.max(...detail.terms.map(term=>term.result)):detail.aggregation==="mean"?sum/detail.terms.length:detail.aggregation==="direct"?detail.terms[0]?.result:sum,result=typeof detail.outputValue==="number"?cellText(detail.outputValue):String(detail.outputValue);
  useEffect(()=>{const close=(event:KeyboardEvent)=>{if(event.key==="Escape")onClose();};window.addEventListener("keydown",close);return()=>window.removeEventListener("keydown",close);},[onClose]);
  return <div className="position-dialog-backdrop" role="presentation" onMouseDown={(event)=>{if(event.target===event.currentTarget)onClose();}}><section className="position-dialog" role="dialog" aria-modal="true" aria-label="当前输出位置计算明细">
    <header><div><span>POSITION CALCULATION</span><h4>{detail.title}</h4></div><button type="button" onClick={onClose} aria-label="关闭计算明细">×</button></header>
    <div className="position-rule"><b>坐标怎么来的</b><p>{detail.rule}</p></div>
    <div className="position-terms">{detail.terms.map((term,index)=><article className={`term-color-${index%8}`} key={`${term.inputCoord}-${index}`}><span>第 {index+1} 项 · 同色格子相互对应</span><div><code>{term.inputCoord}</code><strong>{cellText(term.inputValue)}</strong></div>{term.parameterCoord&&<div><code>{term.parameterCoord}</code><strong>{cellText(term.parameterValue??0)}</strong></div>}<p>{term.parameterValue!==undefined?<>{cellText(term.inputValue)} {term.operator} {cellText(term.parameterValue)} = <b>{cellText(term.result)}</b></>:<>读取该位置 = <b>{cellText(term.result)}</b></>}</p></article>)}</div>
    <div className="position-total"><code>{detail.outputCoord}</code><div><span>{detail.aggregation==="max"?"窗口取最大值":detail.aggregation==="mean"?"窗口求和后除以元素数":detail.aggregation==="direct"?"应用当前逐元素函数":"各项乘积求和"}</span><b>{detail.aggregation==="max"?`max(${detail.terms.map(term=>cellText(term.result)).join(", ")}) = ${cellText(aggregate)}`:detail.aggregation==="mean"?`(${detail.terms.map(term=>cellText(term.result)).join(" + ")}) ÷ ${detail.terms.length} = ${cellText(aggregate)}`:detail.aggregation==="direct"?cellText(aggregate):`${detail.terms.map((term)=>cellText(term.result)).join(" + ")} = ${cellText(aggregate)}`}</b></div>{detail.bias!==undefined&&<div><span>加偏置</span><b>{cellText(aggregate)} + {cellText(detail.bias)} = {result}</b></div>}<strong>最终输出 = {result}</strong></div>
  </section></div>;
}

function highlightCells(entry: ApiEntry, spec: Record<string, unknown>, sim: Simulation, step: number, item: TensorItem, side: "input" | "output") {
  const family = familyOf(entry), count = visualLeaves(item.value).length;
  if (!count) return [];
  if(family==="foreach"){const tensorIndex=Number(item.key.split(".").at(-1));return step===tensorIndex+1?Array.from({length:count},(_,index)=>index):[];}
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
    if (family === "convolution" && cleanLeaf(entry).includes("1d")) {
      if (side === "output") return [step % count];
      if (item.key === "kernel") return Array.from({length:count},(_,index)=>index);
      if (item.key === "input") { const length=sourceShape.at(-1)??1,kernelSize=shapeOf(spec.kernel).at(-1)??1,stride=Number(spec.stride??1),padding=Number(spec.padding??0),dilation=Number(spec.dilation??1),channels=sourceShape.at(-2)??1;return Array.from({length:channels},(_,c)=>Array.from({length:kernelSize},(__,u)=>c*length+step*stride+u*dilation-padding)).flat().filter(index=>index>=0&&index<count); }
    }
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
  const foreach=familyOf(entry)==="foreach",foreachInputs=Array.isArray(parsed.tensors)?parsed.tensors:[],foreachResult=inPlaceResult(entry,sim.value);
  const inputs = foreach?foreachInputs.map((value,index)=>({key:`tensors.${index}`,label:`输入 Tensor ${index}`,value})):inputTensors(parsed);
  const outputs = foreach&&Array.isArray(foreachResult)?foreachResult.map((value,index)=>({key:`tensors.${index}`,label:`输出 Tensor ${index}`,value})):outputTensors(sim.value);
  const [step, setStep] = useState(0), [playing, setPlaying] = useState(false), [detailOpen, setDetailOpen] = useState(false);
  const total = Math.max(1, sim.trace.length), safeStep = Math.min(step, total - 1);
  const detail=positionDetail(entry,parsed,sim,safeStep);
  useEffect(() => {
    if (!playing || total < 2) return;
    const timer = window.setInterval(() => setStep((current) => current >= total - 1 ? 0 : current + 1), 1100);
    return () => window.clearInterval(timer);
  }, [playing, total]);
  if (!inputs.length && !outputs.length) return null;
  return <section className="calculation-viz" aria-label="张量计算过程可视化">
    <div className="calculation-viz__head"><div><span>INTERACTIVE TENSOR FLOW</span><h4>输入 → 逐步计算 → 输出</h4></div><p>橙色单元格是第 {safeStep + 1} 步正在参与计算的位置</p></div>
    <div className="tensor-flow">
      <div className="tensor-side"><small>输入张量</small>{inputs.map((item) => <TensorVisual key={item.key} item={item} highlights={highlightCells(entry, parsed, sim, safeStep, item, "input")} colorHighlights={termColorMap(detail,item.key)} />)}</div>
      <div className="operator-node"><span>{entry.leaf}</span><MathExpression latex={formulaSpecOf(entry).latex} spoken={formulaSpecOf(entry).spoken} /><i>→</i></div>
      <div className="tensor-side"><small>输出张量</small>{outputs.map((item, index) => <TensorVisual key={`${item.key}-${index}`} item={item} highlights={highlightCells(entry, parsed, sim, safeStep, item, "output")} />)}</div>
    </div>
    <div className="process-stepper">
      <div className="stepper-controls"><button type="button" onClick={() => setStep(Math.max(0, safeStep - 1))} disabled={safeStep === 0}>← 上一步</button><button type="button" className={playing ? "is-playing" : ""} onClick={() => setPlaying((value) => !value)} disabled={total < 2}>{playing ? "暂停" : "自动播放"}</button><button type="button" onClick={() => setStep(Math.min(total - 1, safeStep + 1))} disabled={safeStep === total - 1}>下一步 →</button></div>
      <div className="stepper-copy"><span>{safeStep + 1} / {total}</span><p>{sim.trace[safeStep] ?? "输入经过当前算子后直接得到右侧输出。"}</p></div>
      {detail&&<button type="button" className="open-position-detail" onClick={()=>{setPlaying(false);setDetailOpen(true);}}>⊞ 打开第 {safeStep+1} 步的位置计算小窗口</button>}
      {total > 1 && <input aria-label="选择计算步骤" type="range" min="0" max={total - 1} value={safeStep} onChange={(event) => { setStep(Number(event.target.value)); setPlaying(false); }} />}
    </div>
    {detailOpen&&detail&&<PositionCalculationDialog detail={detail} onClose={()=>setDetailOpen(false)} />}
  </section>;
}

function inPlaceResult(entry:ApiEntry,value:unknown){if(familyOf(entry)==="foreach"&&foreachOperation(entry).inPlace&&value&&typeof value==="object")return (value as Record<string,unknown>).mutated_inputs;return value;}

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
    convolution: n.includes("1d") ? { input: [[1, 2, 0, 7, 1, 0, 2], [1, 0, 2, 1, 3, 1, 0]], kernel: [[1, 0, 0, -1], [0, 1, 1, 0]], stride: 1, padding: 0, dilation: 1, bias: 0 } : { input: [[[v[0], 2, 0], [0, v[1], 3], [2, 1, v[2]]], [[1, 0, 2], [2, 1, 0], [0, 3, 1]]], kernel: [[[1, 0], [0, -1]], [[0, 1], [1, 0]]], stride: 1, bias: 0 },
    pooling: { input: [[1, 5, 2, 4], [3, 2, 7, 1], [0, 6, 3, 8], [4, 1, 2, 5]], kernel_size: 2, stride: 2 },
    linear: { input: [2, -1, 3], weight: [[0.5, 1, -1], [2, 0, 0.5]], bias: [0.1, -0.2] },
    matmul: { input: [[v[0], 2, v[1]], [4, v[2], 6]], other: [[1, 2], [0, v[0]], [2, 0]] },
    reshape: { input: [v[0], 2, v[1], 4, v[2], 6, 7, 8], shape: [2, 2, 2] },
    squeeze: { input: [[[1, 2, 3]]], dim: 0 },
    transpose: { input: [[1, 2, 3], [4, 5, 6]], dim0: 0, dim1: 1 },
    combine: { tensors: [[1, 2], [3, 4]], other: [[5, 6], [7, 8]], dim: 0 },
    split: { input: [10, 20, 30, 40, 50, 60], sections: 3, dim: 0 },
    reduction: { input: [[v[0], 7, v[1]], [8, v[2], 5]], dim: -1 },
    sorting: n.includes("topk") ? { input: [4, 1, 7, 3, 9, 2], dim: -1, k: 3, largest: true, sorted: true } : { input: [4, 1, 7, 3, 9, 2], dim: -1, descending: false, stable: false },
    counting: { input: [3, 1, 3, 2, 1, 3, 4, 2] },
    selection: { condition: [true, false, true, false], input: [10, 20, 30, 40], other: [-1, -1, -1, -1] },
    indexing: { input: [[10, 20, 30], [40, 50, 60]], index: [2, 0], dim: 1 },
    binary: { input: [v[0], v[1], v[2]], other: n === "pow" ? 2 : v[0] },
    comparison: { input: [1, 4, 2, 7], other: 3 },
    unary: { input: n.includes("log") || n === "sqrt" ? [0.25, 1, 4, 9] : [-2.7, -0.5, 0, 1.2, 3.8] },
    sequence: n.includes("linspace") ? { start: -1, end: 1, steps: 5 } : { start: 1, end: 10, step: 2 },
    creation: n.startsWith("empty") ? { shape: [2, 2], dtype: "float32" } : { data: [[v[0], v[1]], [v[2], v[0] + v[1]]], shape: [2, 2], fill_value: v[0], dtype: "float32" },
    random: { shape: [2, 3], seed: 42, distribution: n.includes("normal") ? "normal(0,1)" : "uniform(0,1)" },
    fft: { input: [1, 0, -1, 0] },
    linalg: { matrix: [[4, 7], [2, 6]], vector: [1, 0] },
    optimizer: { parameters: [1.5, -0.5], gradients: [0.2, -0.4], learning_rate: 0.1, momentum: 0.9 },
    distribution: { distribution: entry.leaf, parameters: { loc: 0, scale: 1 }, evaluate_at: [-1, 0, 1] },
    dataloader: { samples: [{ x: [1, 2], y: 0 }, { x: [3, 4], y: 1 }, { x: [5, 6], y: 0 }, { x: [7, 8], y: 1 }], batch_size: 2, shuffle: false },
    inspection: { input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] },
    copy_state: { input: [1, 2, 3], requires_grad: true, contiguous: true },
    device: { input: [v[0], v[1], v[2]], from_device: "cpu", to_device: n || "cuda", from_dtype: "float32", to_dtype: n.includes("long") ? "int64" : "float32" },
    predicate: { input: [1, 2, 3], device: "cpu", dtype: "float32", condition: entry.leaf, simulated_available: false },
    state: { setting: entry.name, before: false, requested: true },
    grad_mode: { enabled_before: true, enabled_inside: !n.includes("no_grad") && !n.includes("inference") },
    tensor_bridge: { input: [3.5], device: "cpu", requires_grad: false, shares_storage_when_possible: n === "numpy" },
    module_state: { module: "model", strict: true, keys: ["layer.weight", "layer.bias"] },
    distributed: { process_group: "world", ranks: [0, 1], tensor_before: [1, 2], operation: entry.leaf, async_op: false },
    compile: { callable: "model_or_function", example_input_shape: [2, 3], backend: "inductor", first_call_compiles: true },
    export: { module: "model", example_input_shape: [2, 3], dynamic_shapes: null },
    sparse: { layout: "sparse_coo", indices: [[0, 1], [1, 0]], values: [3, 4], size: [2, 2] },
    quantization: { input: [-1, 0, 1], scale: 0.1, zero_point: 10, dtype: "quint8" },
    object: { constructor: entry.name, arguments: { input_features: v[0] + 2, output_features: v[1] + 1 }, sample_input_shape: [2, v[0] + 2] },
    foreach: { tensors: [[1.2,-1.8,2.0],[[2.3,-0.2],[4.0,4.8]]] },
    api_behavior: { api: entry.name, example_values: v, operation: operationGuideOf(entry).title, parameter_hint: entry.type === "method" ? "在对应对象上调用" : entry.type === "function" ? "把示例值替换进官方签名" : "按构造参数创建对象" },
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
  else if (family === "convolution" && n.includes("1d")) { const rawX=spec.input,rawK=spec.kernel,x=Array.isArray(rawX[0])?rawX:[rawX],k=Array.isArray(rawK[0])?rawK:[rawK],stride=Number(spec.stride??1),padding=Number(spec.padding??0),dilation=Number(spec.dilation??1),length=x[0].length,kernelSize=k[0].length,outLength=Math.floor((length+2*padding-dilation*(kernelSize-1)-1)/stride)+1;value=Array.from({length:outLength},(_,t)=>k.reduce((sum:number,weights:number[],c:number)=>sum+weights.reduce((channel:number,weight:number,u:number)=>{const position=t*stride+u*dilation-padding;return channel+weight*(position>=0&&position<length?x[c][position]:0);},0),Number(spec.bias??0)));trace=(value as number[]).map((result,t)=>{const terms=k.flatMap((weights:number[],c:number)=>weights.map((weight:number,u:number)=>{const position=t*stride+u*dilation-padding;return `${weight}×X[c=${c},t=${position}]=${position>=0&&position<length?x[c][position]:0}`;}));return `输出 Y[t=${t}]：${terms.join(" + ")} + bias(${spec.bias??0}) = ${result}`;});}
  else if (family === "convolution") { const rawX=spec.input,rawK=spec.kernel,x=shapeOf(rawX).length===3?rawX:[rawX],k=shapeOf(rawK).length===3?rawK:[rawK],stride=Number(spec.stride??1),oh=Math.floor((x[0].length-k[0].length)/stride)+1,ow=Math.floor((x[0][0].length-k[0][0].length)/stride)+1;value=Array.from({length:oh},(_,i)=>Array.from({length:ow},(_,j)=>k.reduce((channelSum:number,plane:number[][],c:number)=>channelSum+plane.reduce((s:number,row:number[],u:number)=>s+row.reduce((q,z,v)=>q+z*x[c][i*stride+u][j*stride+v],0),0),Number(spec.bias))));trace=[];for(let i=0;i<oh;i++)for(let j=0;j<ow;j++){const terms=k.flatMap((plane:number[][],c:number)=>plane.flatMap((row:number[],u:number)=>row.map((z:number,v:number)=>`${z}×X[c=${c},h=${i*stride+u},w=${j*stride+v}]=${x[c][i*stride+u][j*stride+v]}`)));trace.push(`输出 Y[h=${i},w=${j}]：${terms.join(" + ")} + bias(${spec.bias??0}) = ${value[i][j]}`);}}
  else if (family === "pooling") { const x=spec.input,size=Number(spec.kernel_size),stride=Number(spec.stride??size),out=[] as number[][],outH=Math.floor((x.length-size)/stride)+1,outW=Math.floor((x[0].length-size)/stride)+1;for(let oi=0;oi<outH;oi++){const row=[];for(let oj=0;oj<outW;oj++){const i=oi*stride,j=oj*stride,window=x.slice(i,i+size).flatMap((r:number[])=>r.slice(j,j+size));row.push(n.includes("avg")?window.reduce((a:number,b:number)=>a+b,0)/window.length:Math.max(...window));trace.push(`输出[${oi},${oj}] 的完整窗口 [${window}] → ${n.includes("avg")?"平均":"最大"}值 ${row.at(-1)}`);}out.push(row);}value=out;}
  else if (family === "sorting") { const pairs=(spec.input as number[]).map((item,index)=>({item,index}));if(n.includes("topk")){const largest=spec.largest!==false,ranked=[...pairs].sort((a,b)=>largest?b.item-a.item:a.item-b.item),picked=ranked.slice(0,Number(spec.k));if(spec.sorted===false)picked.sort((a,b)=>a.index-b.index);value={values:picked.map(x=>x.item),indices:picked.map(x=>x.index)};trace=[`原序列 [${spec.input}]`,`${largest?"从大到小":"从小到大"}比较候选项`, `取 ${spec.k} 项；values 与 indices 一一对应${spec.sorted===false?"，返回顺序在真实 PyTorch 中不受保证":"，并按数值排序"}`];}else{const descending=Boolean(spec.descending),ranked=[...pairs].sort((a,b)=>descending?b.item-a.item:a.item-b.item);value=n.includes("argsort")?ranked.map(x=>x.index):{values:ranked.map(x=>x.item),indices:ranked.map(x=>x.index)};trace=[`原序列 [${spec.input}]`,`${descending?"降序":"升序"}比较并保留原索引`,n.includes("argsort")?"只返回能重排原输入的 indices":"返回 values 与 indices 两个字段"];} }
  else if (family === "counting") { const counts=spec.input.reduce((a:Record<string,number>,x:number)=>(a[x]=(a[x]??0)+1,a),{});value=n.includes("unique")?Object.keys(counts).map(Number):counts;trace=Object.entries(counts).map(([x,c])=>`数值 ${x} 出现 ${c} 次`);}
  else if (family === "selection") { value=spec.condition.map((c:boolean,i:number)=>c?spec.input[i]:spec.other[i]);trace=spec.condition.map((c:boolean,i:number)=>`位置 ${i}：condition=${c} → 选择 ${value[i]}`);}
  else if (family === "indexing") { const a=spec.input as number[][],indices=(spec.index as number[]).map(Number),dim=Number(spec.dim??0);value=dim===1?a.map((row)=>indices.map((index)=>row[index])):indices.map((index)=>a[index]);trace=dim===1?a.flatMap((row,i)=>indices.map((index,j)=>`输出[${i},${j}] ← 输入[${i},${index}] = ${row[index]}`)):indices.map((index,i)=>`输出第 ${i} 行 ← 输入第 ${index} 行`);mode="张量变换";}
  else if (family === "sequence") { const vals=[];if(n.includes("linspace")){for(let i=0;i<spec.steps;i++)vals.push(spec.start+i*(spec.end-spec.start)/(spec.steps-1));trace=[`间隔 = (${spec.end}−${spec.start})/(${spec.steps}−1)`,`从 start 起累计间隔，共生成 ${spec.steps} 项`];}else{for(let x=spec.start;x<spec.end;x+=spec.step)vals.push(x);trace=[`从 ${spec.start} 开始`,`每次增加 ${spec.step}`,`到 ${spec.end} 前停止`];}value=vals;}
  else if (family === "creation") { const dims=(spec.shape as number[]).map(Number),count=dims.reduce((a:number,b:number)=>a*b,1),source=visualLeaves(spec.data);if(n.startsWith("empty")){value={shape:dims,dtype:spec.dtype,contents:"未初始化：真实 PyTorch 不保证任何元素值"};mode="对象与状态";title="只展示内存分配规则";trace=[`确定 shape=[${dims}]、dtype=${spec.dtype}，共分配 ${count} 个元素的位置`,"不执行填零或其他初始化操作","读取前必须由后续运算完整写入；网页故意不展示伪造数值"];}else{let values:number[];if(n.includes("zeros"))values=Array(count).fill(0);else if(n.includes("ones"))values=Array(count).fill(1);else if(n.includes("eye")){const rows=dims[0]??2,cols=dims[1]??rows;values=Array.from({length:rows*cols},(_,i)=>Math.floor(i/cols)===i%cols?1:0);}else if(n.includes("full"))values=Array(count).fill(Number(spec.fill_value??0));else values=source.map(Number);value=rebuild(values,dims.length?dims:shapeOf(spec.data));trace=[`确定输出 shape=[${dims.length?dims:shapeOf(spec.data)}]，需要 ${values.length} 个元素`,n.includes("eye")?"主对角线填 1，其余位置填 0":`按 ${entry.leaf} 的填充值规则生成元素`,`按行写入输出张量`];}}
  else if (family === "random") { const dims=(spec.shape as number[]).map(Number),count=dims.reduce((a:number,b:number)=>a*b,1);let state=Number(spec.seed??42)>>>0;const uniform=()=>{state=(1664525*state+1013904223)>>>0;return state/4294967296;};const values=Array.from({length:count},()=>n.includes("normal")?Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12)))*Math.cos(2*Math.PI*uniform()):uniform());value=rebuild(values,dims);trace=[`固定 seed=${spec.seed??42}，因此每次实验可复现`,n.includes("normal")?"用 Box–Muller 把均匀随机数变换为正态随机数":"线性同余生成 [0,1) 均匀随机数",`生成 ${count} 个数并重组为 shape=[${dims}]`];}
  else if (family === "fft") { const x=spec.input.map(Number),N=x.length;const spectrum=Array.from({length:N},(_,k)=>{let re=0,im=0;for(let t=0;t<N;t++){re+=x[t]*Math.cos(-2*Math.PI*k*t/N);im+=x[t]*Math.sin(-2*Math.PI*k*t/N);}return {real:Number(re.toFixed(5)),imag:Number(im.toFixed(5))};});value=spectrum;trace=spectrum.map((z:{real:number;imag:number},k:number)=>`频点 k=${k}：Σ x[n]·e^(−i2π·${k}n/${N}) = ${z.real}${z.imag<0?"":"+"}${z.imag}i`);}
  else if (family === "linalg") { const [[a,b],[c,d]]=spec.matrix,det=a*d-b*c;if(n.includes("det"))value=det;else if(n.includes("inv")||n.includes("inverse"))value=[[d/det,-b/det],[-c/det,a/det]];else value={matrix:spec.matrix,determinant:det,frobenius_norm:Math.sqrt(a*a+b*b+c*c+d*d)};trace=[`det = ${a}×${d} − ${b}×${c} = ${det}`,n.includes("inv")?`A⁻¹ = (1/${det})·[[${d},${-b}],[${-c},${a}]]`:"根据接口继续做对应线性代数运算"]}
  else if (family === "optimizer") { const p=spec.parameters.map(Number),g=spec.gradients.map(Number),delta=g.map((z:number)=>spec.learning_rate*z),next=p.map((z:number,i:number)=>z-delta[i]);value={before:p,gradient:g,update:delta,after:next};mode="对象与状态";title="参数已按梯度更新";trace=p.map((z:number,i:number)=>`θ${i}: ${z} − ${spec.learning_rate}×${g[i]} = ${next[i]}`);}
  else if (family === "dataloader") { const samples=spec.samples as unknown[],batches:unknown[][]=[];for(let i=0;i<samples.length;i+=spec.batch_size)batches.push(samples.slice(i,i+spec.batch_size));value={batch_count:batches.length,batches};mode="对象与状态";title="样本已组成批次";trace=batches.map((b:unknown[],i:number)=>`batch ${i}: 样本索引 ${i*spec.batch_size}…${i*spec.batch_size+b.length-1}`);}
  else if (family === "inspection") { const sh=shapeOf(input);value=n.includes("numel")?flat(input).length:n==="dim"||n==="ndim"?sh.length:n==="size"||n==="shape"?sh:{shape:sh,numel:flat(input).length};mode="张量变换";trace=[`逐层读取嵌套长度 → shape=${JSON.stringify(sh)}`,`维度数=${sh.length}`,`元素数=${sh.join("×")}=${flat(input).length}`];}
  else if (family === "copy_state") { value={values:input,requires_grad:n.includes("detach")?false:spec.requires_grad,contiguous:spec.contiguous,shares_storage:n.includes("clone")?false:!n.includes("copy")};mode="对象与状态";title="张量值与元数据已更新";trace=[`复制数值 [${flat(input)}]`,n.includes("detach")?"从计算图分离，requires_grad=False":"保留梯度设置",n.includes("clone")?"分配独立存储":"按接口规则处理存储关系"];}
  else if (family === "device") { value={values:input,before:{device:spec.from_device,dtype:spec.from_dtype},after:{device:spec.to_device,dtype:spec.to_dtype},numeric_values_changed:false};mode="对象与状态";title="设备或 dtype 转换结果";trace=[`原张量：device=${spec.from_device}, dtype=${spec.from_dtype}`,`请求转换到 device=${spec.to_device}, dtype=${spec.to_dtype}`,"数值保持不变；真实内存迁移需要对应 PyTorch 硬件后端"];}
  else if (family === "state") { value={setting:spec.setting,before:spec.before,after:spec.requested,changed:spec.before!==spec.requested};mode="对象与状态";title="配置值变化";trace=[`读取 ${spec.setting} = ${spec.before}`,`写入请求值 ${spec.requested}`,`最终值 = ${spec.requested}`];}
  else if (family === "predicate") { const result=n.includes("available")?Boolean(spec.simulated_available):n.includes("float")?spec.dtype.includes("float"):Array.isArray(spec.input);value={condition:entry.leaf,result,runtime_note:n.includes("available")?"示意值由 simulated_available 提供；真实结果必须在目标机器运行 PyTorch 查询":"按页面给定属性判断"};mode="对象与状态";title="条件判断规则示意";trace=[n.includes("available")?"读取当前 PyTorch 构建、驱动与硬件状态（网页不实际探测）":`读取对象属性 device=${spec.device}, dtype=${spec.dtype}`,`应用判断 ${entry.leaf}`,`返回布尔值 ${result}`];}
  else if (["grad_mode","tensor_bridge","module_state","distributed","compile","export","sparse","quantization"].includes(family)) { const guide=operationGuideOf(entry);value={api:entry.name,simulation_only:true,input_contract:spec,returns:guide.returns,side_effect:guide.sideEffect};mode="对象与状态";title=guide.title;trace=[...guide.steps,guide.returns,guide.sideEffect];}
  else if (family === "object") { value={type:entry.name,constructor_arguments:spec.arguments,created:true,sample_input_shape:spec.sample_input_shape};mode="对象与状态";title="对象构造结果";trace=[`解析 ${entry.leaf} 构造参数`,`创建 ${entry.name} 实例`,`记录输入约束 shape=${JSON.stringify(spec.sample_input_shape)}`];}
  else if (family === "foreach") { const {base,inPlace,rule}=foreachOperation(entry),tensors=spec.tensors as unknown[];const calculate=(x:number)=>rule?rule.example(x):x;const outputs=tensors.map(tensor=>mapDeep(tensor,calculate));value=inPlace?{returned:null,mutated_inputs:outputs}:outputs;mode="数值计算";title=`${entry.leaf} 已处理 ${tensors.length} 个张量`;trace=tensors.map((tensor,k)=>{const before=flat(tensor),after=flat(outputs[k]);return `Tensor ${k}：${before.map((x,i)=>`${x} → ${after[i]}`).join("，")}`;});trace.unshift(`列表长度 ${tensors.length}；逐个张量执行 torch.${base}，张量之间不广播`);trace.push(inPlace?"名称以下划线结尾：结果写回输入张量，返回 None":"创建并返回同样长度的新 Tensor 列表，原输入不变");}
  else { const vals=spec.example_values??seededValues(entry),guide=operationGuideOf(entry);value={api:entry.name,example_call:entry.name.startsWith("torch.Tensor.")?`tensor.${entry.leaf}(${vals.join(", ")})`:`${entry.name}(${vals.join(", ")})`,input_values:vals,operation:guide.title,returns:guide.returns,side_effect:guide.sideEffect};mode="对象与状态";title=guide.title;trace=[...guide.steps,guide.returns,guide.sideEffect];}
  return { mode, title, value: roundDeep(value), trace };
}

export default function FullApiBrowser() {
  const [query, setQuery] = useState(""); const [group, setGroup] = useState("全部模块"); const [subcategory, setSubcategory] = useState("全部细分类"); const [kind, setKind] = useState<ApiKind>("函数"); const [page, setPage] = useState(0);
  const [detailTab,setDetailTab]=useState<DetailTab>("overview"); const [showComparisonDirectory,setShowComparisonDirectory]=useState(false);
  const [selectedName, setSelectedName] = useState("torch.add"); const [remote, setRemote] = useState<RemoteDoc | null>(null); const [docLoading, setDocLoading] = useState(true);
  const initial = entries.find((x) => x.name === "torch.add") ?? entries.find((x)=>x.type==="function") ?? entries[0];
  const [spec, setSpec] = useState(() => defaultSpec(initial)); const [sim, setSim] = useState<Simulation>(() => simulate(initial, defaultSpec(initial))); const [simError, setSimError] = useState("");
  const [runState, setRunState] = useState<"idle" | "running" | "success" | "error">("idle"); const [runCount, setRunCount] = useState(0); const [lastRunAt, setLastRunAt] = useState("");
  const [exampleCopied,setExampleCopied]=useState(false);
  const outputRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const pageSize = 12;
  const kindCounts=useMemo(()=>Object.fromEntries((["函数","类","方法","其他"] as ApiKind[]).map(name=>[name,entries.filter(entry=>kindOf(entry)===name).length])) as Record<ApiKind,number>,[]);
  const groupCountsByKind=useMemo(()=>Object.entries(entries.filter(entry=>kindOf(entry)===kind).reduce<Record<string,number>>((all,entry)=>(all[entry.group]=(all[entry.group]??0)+1,all),{})).sort((a,b)=>b[1]-a[1]),[kind]);
  const subcategoryCounts=useMemo(()=>{const scope=entries.filter(entry=>kindOf(entry)===kind&&(group==="全部模块"||entry.group===group));const counts=new Map<string,number>();scope.forEach(entry=>{const name=subcategoryOf(entry);counts.set(name,(counts.get(name)??0)+1);});return [...counts.entries()].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0],"zh-CN"));},[group,kind]);
  const filtered = useMemo(() => { const keyword=query.trim().toLowerCase(); const matches=entries.filter((entry)=>kindOf(entry)===kind&&(group==="全部模块"||entry.group===group)&&(subcategory==="全部细分类"||subcategoryOf(entry)===subcategory)&&(!keyword||`${entry.name} ${entry.summary} ${entry.typeLabel} ${entry.group} ${subcategoryOf(entry)}`.toLowerCase().includes(keyword))); return keyword?[...matches].sort((a,b)=>searchRank(a,keyword)-searchRank(b,keyword)||a.name.localeCompare(b.name)):matches; }, [query,group,subcategory,kind]);
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize)), safePage=Math.min(page,pages-1), visible=filtered.slice(safePage*pageSize,(safePage+1)*pageSize);
  const selected=entries.find((item)=>item.name===selectedName)??visible[0]??entries[0];
  const contract=readabilityOf(selected), tier=simulationTierOf(selected), example=exampleSpecOf(selected);

  useEffect(() => { let active=true; fetch(`/api/docs?name=${encodeURIComponent(selected.name)}&url=${encodeURIComponent(selected.url)}`).then((r)=>r.json()).then((data)=>{if(active){setRemote(data);setDocLoading(false);}}).catch(()=>{if(active){setRemote({error:"官方详情暂时无法读取"});setDocLoading(false);}}); return()=>{active=false;}; }, [selected.name,selected.url]);
  useEffect(()=>{const timer=window.setTimeout(()=>{const requested=new URLSearchParams(window.location.search).get("api"),entry=entries.find(item=>item.name===requested);if(entry){const next=defaultSpec(entry);setSelectedName(entry.name);setKind(kindOf(entry));setGroup(entry.group);setSubcategory(subcategoryOf(entry));setSpec(next);setSim(simulate(entry,next));}},0);return()=>window.clearTimeout(timer);},[]);
  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isEditing = target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName ?? "");
      if (event.key === "/" && !isEditing) {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);

  function applyKind(nextKind:ApiKind){setKind(nextKind);setGroup("全部模块");setSubcategory("全部细分类");setPage(0);setShowComparisonDirectory(false);const first=entries.find(entry=>kindOf(entry)===nextKind);if(first)choose(first);}
  function applyGroup(nextGroup:string){setGroup(nextGroup);setSubcategory("全部细分类");setPage(0);setShowComparisonDirectory(false);}
  function choose(entry:ApiEntry,syncFilters=false,nextTab:DetailTab="overview"){const next=defaultSpec(entry);if(syncFilters){setKind(kindOf(entry));setGroup(entry.group);setSubcategory(subcategoryOf(entry));setPage(0);}setSelectedName(entry.name);setRemote(null);setDocLoading(true);setSpec(next);setSimError("");setRunState("idle");setRunCount(0);setLastRunAt("");setExampleCopied(false);setSim(simulate(entry,next));setDetailTab(nextTab);setShowComparisonDirectory(false);const url=new URL(window.location.href);url.searchParams.set("api",entry.name);window.history.replaceState({},"",url);window.setTimeout(()=>document.getElementById("deep-lab")?.scrollIntoView({behavior:"smooth",block:"start"}),0);}
  function resetInput(){const next=defaultSpec(selected);setSpec(next);setSim(simulate(selected,next));setSimError("");setRunState("idle");setRunCount(0);setLastRunAt("");}
  async function copyExample(){try{await navigator.clipboard.writeText(example.code);setExampleCopied(true);window.setTimeout(()=>setExampleCopied(false),1600);}catch{setExampleCopied(false);}}
  function run(){
    setRunState("running"); setSimError("");
    window.setTimeout(()=>{
      try {
        setSim(simulate(selected,spec)); setRunCount((count)=>count+1); setLastRunAt(new Date().toLocaleTimeString("zh-CN",{hour12:false})); setRunState("success"); setDetailTab("result");
        window.requestAnimationFrame(()=>outputRef.current?.scrollIntoView({behavior:"smooth",block:"center"}));
      } catch(error) {
        setSimError(error instanceof Error?error.message:"请输入有效 JSON"); setRunState("error");
      }
    },180);
  }

  return <section className="docs-atlas" id="all-apis">
    <div className="docs-atlas__intro"><div><p className="eyebrow">PYTORCH 2.13 · API LOOKUP &amp; VISUAL LAB</p><h2>9,066 个接口，按需查询与实验</h2><p>每个条目都提供中文索引、官方链接与学习卡片；高频数值算子展示教学模拟计算，对象、硬件、分布式与编译接口展示处理步骤和状态示意。精确行为请以真实 PyTorch 运行时和官方文档为准。</p></div><div className="docs-atlas__stats"><div><strong>{entries.length.toLocaleString("zh-CN")}</strong><span>官方索引</span></div><div><strong>{groupCounts.length}</strong><span>学习模块</span></div><div><strong>15</strong><span>重点实验</span></div></div></div>
    <div className="docs-kind-tabs" role="tablist" aria-label="按接口种类浏览">{(["函数","类","方法","其他"] as ApiKind[]).map(name=><button role="tab" aria-selected={kind===name} className={kind===name?"active":""} key={name} onClick={()=>applyKind(name)}><span>{name}</span><b>{kindCounts[name].toLocaleString("zh-CN")}</b></button>)}</div>
    <div className="docs-toolbar"><label className="docs-search"><span>⌕</span><input ref={searchRef} value={query} onChange={(e)=>{setQuery(e.target.value);setPage(0);}} placeholder={`在${kind}中搜索 Conv2d、backward…`} aria-label={`搜索 PyTorch ${kind}`} /><kbd>/</kbd></label><span className="docs-toolbar__path">{kind} › {group} › {subcategory}</span></div>
    <nav className="docs-level-one" aria-label="一级模块"><strong>一级模块</strong><div><button className={group==="全部模块"?"active":""} onClick={()=>applyGroup("全部模块")}>全部 <b>{kindCounts[kind]}</b></button>{groupCountsByKind.map(([name,count])=><button key={name} className={group===name?"active":""} onClick={()=>applyGroup(name)}>{name} <b>{count}</b></button>)}</div></nav>
    <nav className="docs-level-two" aria-label="二级细分类"><div className="docs-level-two__label"><strong>二级分类</strong><span>仅显示当前一级模块的分类</span></div><div className="docs-level-two__tabs" role="tablist"><button role="tab" aria-selected={subcategory==="全部细分类"&&!showComparisonDirectory} className={subcategory==="全部细分类"&&!showComparisonDirectory?"active":""} onClick={()=>{setSubcategory("全部细分类");setShowComparisonDirectory(false);setPage(0);}}>全部 <b>{subcategoryCounts.reduce((sum,item)=>sum+item[1],0)}</b></button>{subcategoryCounts.map(([name,count])=><button role="tab" aria-selected={subcategory===name&&!showComparisonDirectory} key={name} className={subcategory===name&&!showComparisonDirectory?"active":""} onClick={()=>{setSubcategory(name);setShowComparisonDirectory(false);setPage(0);}}>{name} <b>{count}</b></button>)}<button role="tab" aria-selected={showComparisonDirectory} className={`comparison-index-link ${showComparisonDirectory?"active":""}`} onClick={()=>setShowComparisonDirectory(true)}>相似方法对比表 <b>{comparisonCatalog.length}</b></button></div></nav>
    {showComparisonDirectory?<ComparisonDirectory selected={selected} onClose={()=>setShowComparisonDirectory(false)} onOpen={(entry,tab)=>choose(entry,true,tab)}/>:<div className="docs-layout">
      <div className="docs-results"><div className="docs-results__head"><p>找到 <b>{filtered.length.toLocaleString("zh-CN")}</b> 个{kind} <em>{group} › {subcategory}</em></p><span>第 {safePage+1} / {pages} 页</span></div><div className="api-table" role="table"><div className="api-table__header"><span>接口 / 细分类</span><span>类型</span><span>具体做什么</span></div>{visible.map((entry)=><button key={entry.name} className={selected.name===entry.name?"selected":""} onClick={()=>choose(entry)}><div><code>{entry.name}</code><small>{subcategoryOf(entry)}</small></div><span>{entry.typeLabel}</span><p>{operationGuideOf(entry).title}</p></button>)}{!visible.length&&<div className="docs-empty">没有匹配的接口，换个关键词试试。</div>}</div><div className="pagination"><button disabled={safePage===0} onClick={()=>setPage(Math.max(0,safePage-1))}>← 上一页</button><span>{filtered.length?safePage*pageSize+1:0}–{Math.min((safePage+1)*pageSize,filtered.length)} / {filtered.length}</span><button disabled={safePage>=pages-1} onClick={()=>setPage(Math.min(pages-1,safePage+1))}>下一页 →</button></div></div>
      <aside className="api-inspector"><div className="api-inspector__top"><span>{selected.group}<em> › {subcategoryOf(selected)}</em></span><i>{selected.typeLabel}</i></div><h3>{selected.leaf}</h3><code className="api-inspector__path">{selected.name}</code><section><small>它具体做什么</small><OperationGuidePanel entry={selected} compact /></section><section><small>数学定义 / 处理规则</small><FormulaPanel entry={selected} compact /></section><section><small>应用场景</small><p>{scenarioOf(selected)}</p></section><button className="official-link" type="button" onClick={()=>choose(selected,false,"overview")}>进入本接口标签页 ↓</button></aside>
    </div>}

    <article className="deep-lab" id="deep-lab">
      <header><div><p className="eyebrow">API QUICK READ &amp; TEACHING LAB</p><h3>{selected.name}</h3></div><span>{tier.label}</span></header>
      <section className="api-quick-read" aria-labelledby="api-quick-read-title">
        <div className="api-quick-read__head"><div><small>先看这里，不必先啃完整文档</small><h4 id="api-quick-read-title">30 秒读懂这个接口</h4></div><div className="api-quick-read__badges"><span>{selected.typeLabel}</span><span>{familyLabels[familyOf(selected)]}</span><span>{contract.level}</span></div></div>
        <div className="api-quick-read__grid">
          <div><b>怎么调用</b><p>{contract.call}</p></div>
          <div><b>输入是什么</b><p>{contract.input}</p></div>
          <div><b>返回什么</b><p>{contract.output}</p></div>
          <div><b>shape 怎么变</b><p>{contract.shape}</p></div>
          <div><b>梯度关系</b><p>{contract.autograd}</p></div>
          <div><b>数据 / 状态副作用</b><p>{operationGuideOf(selected).sideEffect}</p></div>
        </div>
        <div className="api-quick-read__pitfall"><b>⚠ 最容易踩的坑</b><p>{contract.pitfall}</p></div>
        <div className="api-quick-read__check"><b>读完自测：</b><span>我能说出输入吗？</span><span>我能预测输出类型和 shape 吗？</span><span>我知道它会不会改原数据或计算图吗？</span></div>
      </section>
      <div className="deep-lab-tabs" role="tablist" aria-label="接口详情标签页">{([{id:"overview",label:"介绍与公式"},{id:"usage",label:"调用与变量"},{id:"example",label:"Example 与输入"},{id:"result",label:"计算过程与输出"},...(comparisonOf(selected)?[{id:"compare",label:"相似方法区别"}]:[])] as Array<{id:DetailTab;label:string}>).map(tab=><button role="tab" aria-selected={detailTab===tab.id} className={detailTab===tab.id?"active":""} key={tab.id} onClick={()=>setDetailTab(tab.id)}>{tab.label}</button>)}</div>
      <div className="deep-lab__grid">
        {detailTab==="overview"&&<section className="deep-card deep-card--wide"><small>① 它做什么、怎么算、返回什么</small><OperationGuidePanel entry={selected}/><div className="deep-formula"><span>数学定义 / 明确处理规则</span><FormulaPanel entry={selected} /></div><div className="deep-overview-scenario"><b>什么时候使用</b><p>{scenarioOf(selected)}</p></div></section>}
        {detailTab==="usage"&&<section className="deep-card deep-card--wide"><small>② 官方调用方法</small>{docLoading?<p className="loading-line">正在读取官方签名；你可以先看上方“30 秒读懂”…</p>:<><pre><code>{cleanSignature(remote?.signature||`${selected.name}（官方签名暂未加载）`)}</code></pre>{remote?.summary&&<p className="official-summary"><b>官方摘要（英文）</b>{remote.summary}</p>}{remote?.error&&<p className="sim-error" role="status">{remote.error}；请用下方官方链接核对。</p>}</>}<a href={selected.url} target="_blank" rel="noreferrer">核对官方原文 ↗</a></section>}
        {detailTab==="usage"&&<section className="deep-card deep-card--wide"><small>③ 参数与变量地图</small><div className="deep-vars">{variablesOf(selected,remote).map((v)=><div key={`${v.name}-${v.raw}`}><div className="deep-vars__name"><code>{v.name}</code><em className={v.required?"required":"optional"}>{v.name==="result"?"返回说明":v.required?"必填":"可选"}</em></div><p>{v.meaning}</p>{v.raw&&<small>签名片段：<code>{v.raw}</code></small>}<span>默认 / 例：{v.sample}</span></div>)}</div></section>}
        {detailTab==="compare"&&<AlgorithmComparison entry={selected} onSelect={(entry)=>choose(entry,true,"compare")} />}
        {detailTab==="example"&&<section className="deep-card"><small>④ {example.title}</small><p>{scenarioOf(selected)}</p><div className="example-meta"><span className={example.runnable?"runnable":"skeleton"}>{example.runnable?"可复制到 Python 运行":"调用骨架，不能直接运行"}</span><button type="button" onClick={copyExample}>{exampleCopied?"✓ 已复制":"复制代码"}</button></div><pre><code>{example.code}</code></pre><div className="example-output"><b>预期输出 / 观察重点</b><pre><code>{example.output}</code></pre></div></section>}
        {detailTab==="example"&&<section className="deep-card simulator-card"><small>⑤ 网页教学模拟输入</small><p className="simulator-contract"><b>{tier.label}</b> · {tier.note}</p><label><span>页面模拟器的 JSON（不是 Python 函数签名）</span><textarea value={spec} onChange={(e)=>{setSpec(e.target.value);setRunState("idle");}} spellCheck={false}/></label><div className="simulator-actions"><button type="button" className="secondary" onClick={resetInput}>恢复本接口默认输入</button><button type="button" onClick={run} disabled={runState==="running"}>{runState==="running"?"⏳ 正在模拟…":runState==="success"?"✓ 已模拟 · 查看结果":"▶ 运行教学模拟并打开输出"}</button></div>{runState==="idle"&&<p className="run-hint">修改 JSON 后运行，将自动切换到“计算过程与输出”。</p>}{simError&&<p className="sim-error" role="alert">{simError}</p>}</section>}
        {detailTab==="result"&&<section ref={outputRef} className={`deep-card deep-card--wide output-card output-card--${runState}`} aria-live="polite"><small>⑥ 教学模拟过程与输出</small><div className="run-receipt"><b>{runState==="running"?"正在模拟…":runState==="success"?`模拟完成 · 第 ${runCount} 次`:runState==="error"?"模拟失败":"示例结果预览"}</b><span>{lastRunAt?`完成时间 ${lastRunAt}`:"可在 Example 与输入标签修改 JSON"}</span></div><div className="sim-mode"><b>{sim.title}</b><span>{tier.label} · {tier.note}</span></div><CalculationVisualizer key={`${selected.name}-${runCount}`} entry={selected} source={spec} sim={sim} /><div className="trace-row trace-row--compact">{sim.trace.map((step,i)=><div key={`${i}-${step}`}><span>{i+1}</span><p>{step}</p></div>)}</div><div className="result-label">{tier.numeric?"网页按简化规则算出的结果":"规则 / 流程示意结果"}</div><pre><code>{JSON.stringify(sim.value,null,2)}</code></pre></section>}
      </div>
    </article>
  </section>;
}
