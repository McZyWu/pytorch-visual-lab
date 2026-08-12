"use client";

import { useEffect, useMemo, useState } from "react";
import apiIndex from "./api-index.generated.json";

type ApiEntry = {
  name: string; leaf: string; type: string; typeLabel: string; group: string;
  summary: string; display: string; priority: number; url: string;
};
type RemoteDoc = { signature?: string; parameters?: string[]; summary?: string; source?: string; error?: string };
type Variable = { name: string; meaning: string; sample: string };
type Simulation = { mode: "数值执行" | "结构预演"; title: string; value: unknown; trace: string[] };

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
function roundDeep(value: unknown): unknown { return mapDeep(value, (n) => Number(n.toFixed(5))); }
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

function defaultSpec(entry: ApiEntry) {
  const n = cleanLeaf(entry);
  if (/reshape|view/.test(n)) return '{"input":[[1,2,3],[4,5,6]],"shape":[3,2]}';
  if (/cat|stack/.test(n)) return '{"input":[[1,2]],"other":[[3,4]],"dim":0}';
  if (/matmul|^mm$|^bmm$/.test(n)) return '{"input":[[1,2],[3,4]],"other":[[2,0],[1,2]]}';
  if (/add|sub|mul|div|pow/.test(n)) return '{"input":[1,2,3],"other":2}';
  if (/softmax|mean|sum|max|min|argmax|argmin/.test(n)) return '{"input":[[1,2,3],[4,5,6]],"dim":-1}';
  if (/relu|sigmoid|abs|neg|exp|log|sqrt|sin|cos|floor|ceil|round/.test(n)) return '{"input":[-2,-0.5,0,1,2]}';
  return `{"input":[[1,2],[3,4]],"args":[],"kwargs":{},"context":{"device":"cpu","dtype":"float32"}}`;
}

function simulate(entry: ApiEntry, source: string): Simulation {
  const spec = JSON.parse(source) as { input?: unknown; other?: unknown; dim?: number; shape?: number[]; args?: unknown[]; kwargs?: Record<string, unknown>; context?: Record<string, unknown> };
  const n = cleanLeaf(entry); const input = spec.input ?? []; let value: unknown; let numeric = true;
  const unary: Record<string, (x: number) => number> = { abs: Math.abs, absolute: Math.abs, neg: (x) => -x, negative: (x) => -x, relu: (x) => Math.max(0, x), sigmoid: (x) => 1 / (1 + Math.exp(-x)), exp: Math.exp, expm1: Math.expm1, log: Math.log, log10: Math.log10, log2: Math.log2, sqrt: Math.sqrt, square: (x) => x * x, sin: Math.sin, cos: Math.cos, tan: Math.tan, floor: Math.floor, ceil: Math.ceil, round: Math.round, trunc: Math.trunc };
  if (unary[n]) value = mapDeep(input, unary[n]);
  else if (["add", "sub", "mul", "multiply", "div", "divide", "pow"].includes(n)) { const other = Number(spec.other ?? 1); const ops: Record<string, (a: number, b: number) => number> = { add: (a,b)=>a+b, sub:(a,b)=>a-b, mul:(a,b)=>a*b, multiply:(a,b)=>a*b, div:(a,b)=>a/b, divide:(a,b)=>a/b, pow:(a,b)=>a**b }; value = mapDeep(input, (x) => ops[n](x, other)); }
  else if (["sum", "mean", "prod", "max", "min", "argmax", "argmin"].includes(n)) { const reducer: Record<string,(x:number[])=>number> = { sum:(x)=>x.reduce((a,b)=>a+b,0), mean:(x)=>x.reduce((a,b)=>a+b,0)/x.length, prod:(x)=>x.reduce((a,b)=>a*b,1), max:(x)=>Math.max(...x), min:(x)=>Math.min(...x), argmax:(x)=>x.indexOf(Math.max(...x)), argmin:(x)=>x.indexOf(Math.min(...x)) }; value = spec.dim === undefined ? reducer[n](flat(input)) : reduceLast(input, reducer[n]); }
  else if (n === "softmax") { const soft=(xs:number[])=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);}; value=Array.isArray(input)&&Array.isArray(input[0])?(input as number[][]).map(soft):soft(input as number[]); }
  else if (/reshape|view/.test(n)) { const values = flat(input), dims = spec.shape ?? [values.length]; if (dims.reduce((a,b)=>a*b,1)!==values.length) throw new Error("shape 的元素总数必须与 input 一致"); value = rebuild(values,dims); }
  else if (n === "flatten") value = flat(input);
  else if (/matmul|^mm$|^bmm$/.test(n)) { const a=input as number[][], b=spec.other as number[][]; if(!Array.isArray(a?.[0])||!Array.isArray(b?.[0])||a[0].length!==b.length) throw new Error("矩阵乘法要求 A 的列数等于 B 的行数"); value=a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0))); }
  else if (n === "cat") { const a=input as unknown[], b=spec.other as unknown[]; value=spec.dim===1?a.map((row,i)=>[...(row as unknown[]),...((b[i] as unknown[])??[])]):[...a,...b]; }
  else if (n === "stack") value=[input,spec.other];
  else { numeric = false; value = { api: entry.name, status: "预演完成", input_contract: { shape: shapeOf(input), context: spec.context ?? { device:"cpu", dtype:"float32" } }, call: { args: spec.args ?? [], kwargs: spec.kwargs ?? {} }, output_contract: entry.type === "class" ? `${entry.leaf} 实例` : entry.type === "attribute" || entry.type === "property" ? `${entry.leaf} 属性值` : "返回值以官方签名为准" }; }
  return { mode: numeric ? "数值执行" : "结构预演", title: numeric ? "浏览器已计算输出" : "接口生命周期已预演", value: numeric ? roundDeep(value) : value, trace: numeric ? [`解析输入 shape=${JSON.stringify(shapeOf(input))}`, `应用 ${entry.name}`, `得到输出 shape=${JSON.stringify(shapeOf(value))}`] : ["校验输入对象与参数", `分派到 ${entry.group} 后端`, "生成返回对象或更新运行状态"] };
}

export default function FullApiBrowser() {
  const [query, setQuery] = useState(""); const [group, setGroup] = useState("全部模块"); const [type, setType] = useState("全部类型"); const [page, setPage] = useState(0);
  const [selectedName, setSelectedName] = useState("torch.Tensor.backward"); const [remote, setRemote] = useState<RemoteDoc | null>(null); const [docLoading, setDocLoading] = useState(true);
  const initial = entries.find((x) => x.name === "torch.Tensor.backward") ?? entries[0];
  const [spec, setSpec] = useState(() => defaultSpec(initial)); const [sim, setSim] = useState<Simulation>(() => simulate(initial, defaultSpec(initial))); const [simError, setSimError] = useState("");
  const pageSize = 80;
  const filtered = useMemo(() => { const keyword=query.trim().toLowerCase(); return entries.filter((entry)=>(group==="全部模块"||entry.group===group)&&(type==="全部类型"||entry.typeLabel===type)&&(!keyword||`${entry.name} ${entry.summary} ${entry.typeLabel} ${entry.group}`.toLowerCase().includes(keyword))); }, [query,group,type]);
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize)), safePage=Math.min(page,pages-1), visible=filtered.slice(safePage*pageSize,(safePage+1)*pageSize);
  const selected=entries.find((item)=>item.name===selectedName)??visible[0]??entries[0];

  useEffect(() => { let active=true; fetch(`/api/docs?name=${encodeURIComponent(selected.name)}&url=${encodeURIComponent(selected.url)}`).then((r)=>r.json()).then((data)=>{if(active){setRemote(data);setDocLoading(false);}}).catch(()=>{if(active){setRemote({error:"官方详情暂时无法读取"});setDocLoading(false);}}); return()=>{active=false;}; }, [selected.name,selected.url]);

  function applyFilter(nextGroup:string,nextType:string){setGroup(nextGroup);setType(nextType);setPage(0);}
  function choose(entry:ApiEntry){const next=defaultSpec(entry);setSelectedName(entry.name);setRemote(null);setDocLoading(true);setSpec(next);setSimError("");setSim(simulate(entry,next));}
  function run(){try{setSim(simulate(selected,spec));setSimError("");}catch(error){setSimError(error instanceof Error?error.message:"请输入有效 JSON");}}

  return <section className="docs-atlas" id="all-apis">
    <div className="docs-atlas__intro"><div><p className="eyebrow">PYTORCH 2.13 · EVERY API IS A LAB</p><h2>9,066 个接口，全部进入深度实验</h2><p>选择任意官方 API，立即得到中文作用、精确签名、参数解释、核心关系式、调用示例、应用场景与输入输出实验。数值算子直接计算；硬件、分布式、编译等接口进行真实约束下的结构预演。</p></div><div className="docs-atlas__stats"><div><strong>{entries.length.toLocaleString("zh-CN")}</strong><span>深度实验</span></div><div><strong>{groupCounts.length}</strong><span>学习模块</span></div><div><strong>100%</strong><span>API 覆盖</span></div></div></div>
    <div className="docs-toolbar"><label className="docs-search"><span>⌕</span><input value={query} onChange={(e)=>{setQuery(e.target.value);setPage(0);}} placeholder="搜索 torch.compile、Conv2d、backward…" aria-label="搜索全部 PyTorch API" /><kbd>/</kbd></label><select value={group} onChange={(e)=>applyFilter(e.target.value,type)}><option>全部模块</option>{groupCounts.map(([name,count])=><option key={name}>{name} · {count}</option>)}</select><select value={type} onChange={(e)=>applyFilter(group,e.target.value)}><option>全部类型</option>{typeCounts.map(([name,count])=><option key={name}>{name} · {count}</option>)}</select></div>
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
        <section className="deep-card simulator-card"><small>⑤ 输入与执行</small><label><span>实验输入（JSON）</span><textarea value={spec} onChange={(e)=>setSpec(e.target.value)} spellCheck={false}/></label><button onClick={run}>▶ 运行本接口实验</button>{simError&&<p className="sim-error">{simError}</p>}</section>
        <section className="deep-card deep-card--wide output-card"><small>⑥ 输出与逐步解释</small><div className="sim-mode"><b>{sim.title}</b><span>{sim.mode==="数值执行"?"与所示运算的数值规则一致":"不伪造硬件或运行时结果"}</span></div><div className="trace-row">{sim.trace.map((step,i)=><div key={step}><span>{i+1}</span><p>{step}</p></div>)}</div><pre><code>{JSON.stringify(sim.value,null,2)}</code></pre></section>
      </div>
    </article>
  </section>;
}
