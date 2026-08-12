"use client";

import { useMemo, useState } from "react";
import apiIndex from "./api-index.generated.json";

type ApiEntry = {
  name: string;
  leaf: string;
  type: string;
  typeLabel: string;
  group: string;
  summary: string;
  display: string;
  priority: number;
  url: string;
};

const entries = apiIndex as ApiEntry[];
const groupCounts = Object.entries(entries.reduce<Record<string, number>>((all, item) => {
  all[item.group] = (all[item.group] ?? 0) + 1;
  return all;
}, {})).sort((a, b) => b[1] - a[1]);

const typeCounts = Object.entries(entries.reduce<Record<string, number>>((all, item) => {
  all[item.typeLabel] = (all[item.typeLabel] ?? 0) + 1;
  return all;
}, {})).sort((a, b) => b[1] - a[1]);

function signatureOf(entry: ApiEntry) {
  if (entry.type === "class") return `${entry.name}(*args, **kwargs)`;
  if (entry.type === "function" || entry.type === "method") return `${entry.name}(*args, **kwargs)`;
  return entry.name;
}

function usageOf(entry: ApiEntry) {
  const namespace = entry.name.split(".").slice(0, -1).join(".");
  if (entry.type === "class") return `obj = ${entry.name}(...)\n# 按官方签名传入构造参数`;
  if (entry.name.startsWith("torch.Tensor.")) return `result = tensor.${entry.leaf}(...)\n# tensor 为当前张量，... 为该方法参数`;
  if (entry.type === "function") return `result = ${entry.name}(...)\n# ... 按官方签名替换为输入和关键字参数`;
  if (entry.type === "attribute" || entry.type === "property") return `value = ${entry.name}\n# 读取 ${namespace} 上的 ${entry.leaf} 属性`;
  return `from ${namespace} import ${entry.leaf}\n# 查阅官方文档了解完整约束`;
}

export default function FullApiBrowser() {
  const [query, setQuery] = useState("");
  const [group, setGroup] = useState("全部模块");
  const [type, setType] = useState("全部类型");
  const [page, setPage] = useState(0);
  const [selectedName, setSelectedName] = useState("torch.Tensor.backward");
  const pageSize = 80;

  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return entries.filter((entry) =>
      (group === "全部模块" || entry.group === group) &&
      (type === "全部类型" || entry.typeLabel === type) &&
      (!keyword || `${entry.name} ${entry.summary} ${entry.typeLabel} ${entry.group}`.toLowerCase().includes(keyword))
    );
  }, [query, group, type]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, pages - 1);
  const visible = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize);
  const selected = entries.find((item) => item.name === selectedName) ?? visible[0] ?? entries[0];

  function applyFilter(nextGroup: string, nextType: string) {
    setGroup(nextGroup); setType(nextType); setPage(0);
  }

  return (
    <section className="docs-atlas" id="all-apis">
      <div className="docs-atlas__intro">
        <div>
          <p className="eyebrow">PYTORCH 2.13 · OFFICIAL DOCS INVENTORY</p>
          <h2>官方文档全量接口图谱</h2>
          <p>索引由 PyTorch stable 官方文档清单自动生成。不是手写的几十个函数，而是覆盖函数、类、Tensor 方法、属性与各子模块的完整可检索目录。</p>
        </div>
        <div className="docs-atlas__stats">
          <div><strong>{entries.length.toLocaleString("zh-CN")}</strong><span>官方 API 条目</span></div>
          <div><strong>{groupCounts.length}</strong><span>学习模块</span></div>
          <div><strong>2.13</strong><span>文档版本</span></div>
        </div>
      </div>

      <div className="docs-toolbar">
        <label className="docs-search"><span>⌕</span><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="搜索 torch.compile、Conv2d、backward…" aria-label="搜索全部 PyTorch API" /><kbd>/</kbd></label>
        <select value={group} onChange={(event) => applyFilter(event.target.value, type)} aria-label="按模块筛选"><option>全部模块</option>{groupCounts.map(([name, count]) => <option key={name} value={name}>{name} · {count}</option>)}</select>
        <select value={type} onChange={(event) => applyFilter(group, event.target.value)} aria-label="按接口类型筛选"><option>全部类型</option>{typeCounts.map(([name, count]) => <option key={name} value={name}>{name} · {count}</option>)}</select>
      </div>

      <div className="docs-layout">
        <aside className="docs-groups">
          <button className={group === "全部模块" ? "active" : ""} onClick={() => applyFilter("全部模块", type)}><span>全部模块</span><b>{entries.length}</b></button>
          {groupCounts.map(([name, count]) => <button key={name} className={group === name ? "active" : ""} onClick={() => applyFilter(name, type)}><span>{name}</span><b>{count}</b></button>)}
        </aside>

        <div className="docs-results">
          <div className="docs-results__head"><p>找到 <b>{filtered.length.toLocaleString("zh-CN")}</b> 个接口</p><span>第 {safePage + 1} / {pages} 页</span></div>
          <div className="api-table" role="table" aria-label="PyTorch API 结果">
            <div className="api-table__header" role="row"><span>接口</span><span>类型</span><span>中文用途</span></div>
            {visible.map((entry) => <button role="row" key={entry.name} className={selected?.name === entry.name ? "selected" : ""} onClick={() => setSelectedName(entry.name)}>
              <code role="cell">{entry.name}</code><span role="cell">{entry.typeLabel}</span><p role="cell">{entry.summary}</p>
            </button>)}
            {!visible.length && <div className="docs-empty">没有匹配的接口。试试英文函数名或选择其他模块。</div>}
          </div>
          <div className="pagination"><button disabled={safePage === 0} onClick={() => setPage(Math.max(0, safePage - 1))}>← 上一页</button><span>{safePage * pageSize + 1}–{Math.min((safePage + 1) * pageSize, filtered.length)} / {filtered.length}</span><button disabled={safePage >= pages - 1} onClick={() => setPage(Math.min(pages - 1, safePage + 1))}>下一页 →</button></div>
        </div>

        {selected && <aside className="api-inspector">
          <div className="api-inspector__top"><span>{selected.group}</span><i>{selected.typeLabel}</i></div>
          <h3>{selected.leaf}</h3><code className="api-inspector__path">{selected.name}</code>
          <section><small>它做什么</small><p>{selected.summary}。该条目直接对应 PyTorch 2.13 官方文档，可从下方打开完整英文定义、注意事项与版本信息。</p></section>
          <section><small>调用形式</small><pre><code>{signatureOf(selected)}</code></pre></section>
          <section><small>使用骨架</small><pre><code>{usageOf(selected)}</code></pre></section>
          <section className="variable-legend"><small>通用变量说明</small><dl><div><dt><code>input / tensor</code></dt><dd>要被处理的输入张量或对象</dd></div><div><dt><code>*args</code></dt><dd>按位置传入的参数</dd></div><div><dt><code>**kwargs</code></dt><dd>带名称的可选参数</dd></div><div><dt><code>result</code></dt><dd>接口返回的结果</dd></div></dl></section>
          <a className="official-link" href={selected.url} target="_blank" rel="noreferrer">打开官方文档 ↗</a>
          <p className="api-inspector__note">重点函数仍可在页面上方的深度实验区修改 Tensor 并执行；其余接口提供全量检索与官方细节入口。</p>
        </aside>}
      </div>
    </section>
  );
}
