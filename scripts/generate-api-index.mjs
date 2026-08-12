import { readFile, writeFile } from "node:fs/promises";
import { inflateSync } from "node:zlib";

const inventoryPath = new URL("../pytorch-objects.inv", import.meta.url);
const outputPath = new URL("../app/api-index.generated.json", import.meta.url);
const buffer = await readFile(inventoryPath);
let offset = 0;
for (let i = 0; i < 4; i += 1) offset = buffer.indexOf(10, offset) + 1;
const rows = inflateSync(buffer.subarray(offset)).toString("utf8").trim().split(/\r?\n/);

const typeLabels = {
  function: "函数", class: "类", method: "方法", attribute: "属性",
  property: "属性", exception: "异常", data: "数据", module: "模块",
};

function groupOf(name) {
  if (name.startsWith("torch.Tensor")) return "Tensor 方法";
  if (name.startsWith("torch.nn.functional")) return "神经网络函数";
  if (name.startsWith("torch.nn")) return "神经网络模块";
  if (name.startsWith("torch.optim")) return "优化器";
  if (name.startsWith("torch.autograd")) return "自动微分";
  if (name.startsWith("torch.linalg")) return "线性代数";
  if (name.startsWith("torch.fft")) return "傅里叶变换";
  if (name.startsWith("torch.special")) return "特殊函数";
  if (name.startsWith("torch.distributions")) return "概率分布";
  if (name.startsWith("torch.utils.data")) return "数据加载";
  if (name.startsWith("torch.utils")) return "工具组件";
  if (name.startsWith("torch.cuda") || name.startsWith("torch.xpu") || name.startsWith("torch.mps")) return "设备与加速";
  if (name.startsWith("torch.distributed")) return "分布式训练";
  if (name.startsWith("torch.amp")) return "混合精度";
  if (name.startsWith("torch.jit") || name.startsWith("torch.compiler") || name.startsWith("torch._dynamo")) return "编译与导出";
  if (name.startsWith("torch.onnx") || name.startsWith("torch.export")) return "模型导出";
  if (name.startsWith("torch.profiler")) return "性能分析";
  if (name.startsWith("torch.sparse")) return "稀疏张量";
  if (name.startsWith("torch.quantization") || name.startsWith("torch.ao")) return "量化";
  if (name.startsWith("torch.random")) return "随机数";
  if (name.startsWith("torch")) return "核心运算";
  return "其他";
}

function actionOf(name, type) {
  const leaf = name.split(".").at(-1) ?? name;
  const map = [
    [/^(is_|has_|can_)/, "检查或判断"], [/^(get_|read_|load)/, "读取或获取"],
    [/^(set_|write_|save)/, "设置或保存"], [/^(add|sub|mul|div|pow)/, "执行逐元素数学运算"],
    [/^(mean|sum|prod|max|min|arg)/, "沿指定维度执行归约计算"], [/^(reshape|view|flatten|squeeze|unsqueeze)/, "调整张量形状或维度"],
    [/^(cat|stack|split|chunk)/, "组合或拆分张量"], [/^(rand|normal|bernoulli|multinomial)/, "生成或采样随机数据"],
    [/^(to|cpu|cuda|xpu|float|double|half|long|int|bool)/, "转换设备或数据类型"],
    [/^(backward|grad)/, "计算或管理梯度"], [/^(register|remove)/, "注册或移除组件"],
  ];
  const hit = map.find(([pattern]) => pattern.test(leaf));
  if (hit) return hit[1];
  if (type === "class") return `创建和管理 ${leaf} 对象`;
  if (type === "method") return `调用 ${leaf} 方法处理当前对象`;
  if (type === "attribute" || type === "property") return `读取 ${leaf} 属性值`;
  if (type === "exception") return `表示 ${leaf} 异常`;
  return `使用 PyTorch 的 ${leaf} 接口`;
}

const seen = new Set();
const entries = [];
for (const row of rows) {
  const match = row.match(/^(\S+)\s+py:(\S+)\s+(-?\d+)\s+(\S+)\s+(.*)$/);
  if (!match) continue;
  const [, name, type, priority, uri, display] = match;
  if (!name.startsWith("torch") || !typeLabels[type] || seen.has(name)) continue;
  seen.add(name);
  const finalUri = uri.replace(/\$$/, name);
  entries.push({
    name,
    leaf: name.split(".").at(-1),
    type,
    typeLabel: typeLabels[type],
    group: groupOf(name),
    summary: actionOf(name, type),
    display: display === "-" ? name : display,
    priority: Number(priority),
    url: `https://docs.pytorch.org/docs/2.13/${finalUri}`,
  });
}

entries.sort((a, b) => a.group.localeCompare(b.group) || a.name.localeCompare(b.name));
await writeFile(outputPath, `${JSON.stringify(entries)}\n`, "utf8");
console.log(`Generated ${entries.length} documented PyTorch APIs → ${outputPath.pathname}`);
