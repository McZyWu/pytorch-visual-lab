export type GuideLevel = "入门" | "常用" | "进阶";

export type CuratedFunctionGuide = {
  name: string;
  purpose: string;
  input: string;
  output: string;
  shape: string;
  pitfall: string;
  useWhen: string;
  avoidWhen: string;
  example: string;
  related: string[];
  level: GuideLevel;
  call?: string;
  autograd: string;
  sideEffect: string;
};

type GuideOptions = Partial<Pick<CuratedFunctionGuide, "call" | "autograd" | "sideEffect" | "level">>;

const TRACKS_GRADIENT = "浮点或复数输入需要梯度、且该运算可微时，结果会接入 autograd 计算图。";
const NO_TENSOR_GRADIENT = "它返回 Python 值、索引或运行时状态，本身不产生可反向传播的浮点结果。";
const NO_MUTATION = "返回新结果，不原地修改输入；但结果是否共享底层存储要看本函数的说明。";

function guide(
  name: string,
  purpose: string,
  input: string,
  output: string,
  shape: string,
  pitfall: string,
  useWhen: string,
  avoidWhen: string,
  example: string,
  related: string[] = [],
  options: GuideOptions = {},
): CuratedFunctionGuide {
  return {
    name,
    purpose,
    input,
    output,
    shape,
    pitfall,
    useWhen,
    avoidWhen,
    example,
    related,
    level: options.level ?? "常用",
    call: options.call,
    autograd: options.autograd ?? TRACKS_GRADIENT,
    sideEffect: options.sideEffect ?? NO_MUTATION,
  };
}

const guides: CuratedFunctionGuide[] = [
  // 01–10 · 创建与数据桥接
  guide("torch.tensor", "把 Python 数据复制成一个新 Tensor", "Python 标量、列表、元组或可转成数组的数据；可指定 dtype、device 和 requires_grad。", "一个拥有独立数据的新 Tensor。", "由输入的嵌套层级和每层长度决定；不规则嵌套不能直接转换。", "它通常会复制数据；若想尽量复用已有 Tensor 或 NumPy 内存，应先看 as_tensor / from_numpy。", "从少量常量、标签或 Python 列表开始创建张量。", "输入本来就是 Tensor，且你希望保留存储或梯度历史。", "x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)", ["torch.as_tensor", "torch.from_numpy"], { level: "入门" }),
  guide("torch.as_tensor", "尽量复用已有数据，把输入解释成 Tensor", "Tensor、NumPy 数组或 Python 序列；dtype/device 不变时更可能避免复制。", "一个 Tensor；在可复用条件满足时与输入共享数据。", "与输入数据的逻辑形状一致。", "“尽量不复制”不等于“保证不复制”；dtype 或 device 改变时仍会复制。", "已有数组或 Tensor，希望避免不必要的数据复制。", "你明确需要完全独立、可安全原地修改的新数据。", "x = torch.as_tensor(numpy_array)", ["torch.tensor", "torch.from_numpy"], { level: "入门" }),
  guide("torch.from_numpy", "让 CPU Tensor 与 NumPy 数组共享内存", "一个支持的 NumPy ndarray。", "CPU Tensor；默认与原 ndarray 共享底层内存。", "与 ndarray.shape 相同。", "修改任意一边通常会影响另一边；不可写 NumPy 数组会带来未定义写入风险。", "在 NumPy 与 PyTorch 间零拷贝交换 CPU 数据。", "需要独立副本，或数据要直接放到 GPU。", "x = torch.from_numpy(array)", ["torch.as_tensor", "torch.Tensor.numpy"], { level: "入门", sideEffect: "函数本身不修改输入，但返回值通常与 NumPy 数组共享存储。" }),
  guide("torch.zeros", "创建所有元素为 0 的 Tensor", "目标 size，以及可选 dtype、device、layout、requires_grad。", "元素全为 0 的新 Tensor。", "完全由 size 决定。", "未指定 dtype/device 时使用当前默认设置；zeros_like 才会继承参考 Tensor 的属性。", "初始化计数器、掩码、累加缓冲区或模型输入。", "你会立刻覆盖所有元素；此时 empty 更节省初始化开销。", "x = torch.zeros((2, 3))", ["torch.empty", "torch.ones"], { level: "入门" }),
  guide("torch.ones", "创建所有元素为 1 的 Tensor", "目标 size，以及可选 dtype、device、layout、requires_grad。", "元素全为 1 的新 Tensor。", "完全由 size 决定。", "ones 不是单位矩阵；二维单位矩阵请用 eye。", "构造乘法初值、全保留掩码或简单测试输入。", "你需要主对角线为 1、其余为 0 的矩阵。", "x = torch.ones((2, 3))", ["torch.zeros", "torch.eye"], { level: "入门" }),
  guide("torch.empty", "只分配存储，不初始化元素值", "目标 size，以及可选 dtype、device、layout、requires_grad。", "shape 正确但内容未定义的新 Tensor。", "完全由 size 决定。", "empty 不是 zeros；在完整写入前读取或展示其中数值都是错误用法。", "后续代码一定会覆盖每个元素，希望省去填零成本。", "任何元素可能在写入前被读取。", "x = torch.empty((2, 3)); x.fill_(0)", ["torch.zeros", "torch.empty_like"], { level: "入门" }),
  guide("torch.arange", "按固定步长生成半开区间序列", "start、end、step；只传一个值时它是 end，start 默认为 0。", "一维 Tensor，包含 start 但通常不包含 end。", "长度由区间跨度和 step 决定。", "浮点步长会有舍入误差；需要指定元素个数或稳定端点时用 linspace。", "生成整数索引、时间步或等间隔坐标。", "你关心精确包含两个端点或固定元素数量。", "idx = torch.arange(0, 6, 2)  # [0, 2, 4]", ["torch.linspace"], { level: "入门" }),
  guide("torch.linspace", "在两个端点之间生成固定数量的等间隔值", "start、end 和 steps。", "一维 Tensor；steps>1 时包含 start 与 end。", "shape 为 (steps,)。", "第三个参数是元素个数，不是步长；它与 arange 的思路不同。", "绘图坐标、插值位置和固定采样点。", "你更关心固定步长而不是固定点数。", "x = torch.linspace(0, 1, steps=5)", ["torch.arange"], { level: "入门" }),
  guide("torch.eye", "创建二维单位矩阵或矩形对角矩阵", "行数 n、可选列数 m，以及 dtype/device。", "主对角线为 1、其他位置为 0 的二维 Tensor。", "shape 为 (n, n)，或给出 m 时为 (n, m)。", "eye 只创建二维结果；批量单位矩阵需先扩展或 repeat。", "线性代数中的单位映射、对角正则项或测试矩阵。", "你只是想创建所有元素都为 1 的矩阵。", "identity = torch.eye(3)", ["torch.ones", "torch.diag"], { level: "入门" }),
  guide("torch.rand", "从 [0, 1) 均匀分布采样", "目标 size，以及可选 generator、dtype、device。", "随机浮点 Tensor。", "完全由 size 决定。", "每次调用会推进随机数状态；需要复现时固定生成器或 manual_seed。", "生成随机输入、概率或初始化测试数据。", "你需要正态分布、整数或密码学安全随机数。", "x = torch.rand((2, 3))", ["torch.randn", "torch.randint", "torch.manual_seed"], { level: "入门", sideEffect: "不修改输入，但会推进所用随机数生成器的状态。" }),

  // 11–25 · 形状、拼接与索引
  guide("torch.reshape", "改变 Tensor 的逻辑形状", "输入 Tensor 和目标 shape；可用一个 -1 自动推断维度。", "元素顺序相同、shape 改变的 Tensor。", "变为目标 shape，元素总数必须保持不变。", "reshape 可能返回 view，也可能复制；不要用它判断是否共享存储。", "只想重新组织维度，不改变元素顺序。", "你必须保证结果与输入共享存储；此时应明确检查并考虑 view。", "y = torch.reshape(x, (2, -1))", ["torch.Tensor.view", "torch.flatten"], { level: "入门" }),
  guide("torch.flatten", "把连续的一段维度压成一维", "输入 Tensor、start_dim 和 end_dim。", "指定维度区间被合并后的 Tensor。", "区间内维度大小相乘，区间外维度保持不变。", "默认从 dim=0 开始，会把 batch 维也压掉；神经网络常用 start_dim=1。", "从卷积特征转入全连接层，或简化连续维度。", "你要交换维度顺序，或只删除长度为 1 的维度。", "y = torch.flatten(x, start_dim=1)", ["torch.reshape", "torch.squeeze"], { level: "入门" }),
  guide("torch.squeeze", "删除长度为 1 的维度", "输入 Tensor，以及可选 dim。", "视图或等价结果；只移除大小为 1 的维度。", "指定的 1 维消失；未指定 dim 时所有 1 维都可能消失。", "省略 dim 可能意外删除 batch=1 的批次维。", "清理明确的单例通道或占位维度。", "你不知道哪些 1 维可以安全删除。", "y = torch.squeeze(x, dim=1)", ["torch.unsqueeze"], { level: "入门" }),
  guide("torch.unsqueeze", "在指定位置插入一个长度为 1 的维度", "输入 Tensor 和插入位置 dim；负维度从末尾计数。", "与输入共享数据的视图。", "rank 增加 1，新维度大小为 1。", "dim 是插入位置，不是已有维度编号；合法范围比输入 rank 多一个位置。", "添加 batch、channel 或广播所需的占位维。", "你需要复制数据或扩展维度长度；这时还需 expand/repeat。", "y = torch.unsqueeze(x, dim=0)", ["torch.squeeze", "torch.Tensor.expand"], { level: "入门", sideEffect: "不修改输入；结果通常与输入共享底层存储。" }),
  guide("torch.transpose", "交换两个维度", "输入 Tensor、dim0 和 dim1。", "交换指定维度后的视图。", "shape 中 dim0 与 dim1 的大小互换。", "结果常常不连续；后续 view 之前可能需要 contiguous。", "矩阵转置，或交换 batch/channel/空间维。", "你要一次重排三个以上维度；permute 更清晰。", "y = torch.transpose(x, 0, 1)", ["torch.permute", "torch.Tensor.T"], { level: "入门", sideEffect: "不修改输入；结果通常共享存储且可能是非连续视图。" }),
  guide("torch.permute", "按给定顺序重排全部维度", "输入 Tensor 和包含每个维度一次的 dims。", "重排维度后的视图。", "输出第 i 维大小等于输入 dims[i] 维大小。", "dims 必须包含所有维度且不能重复；结果通常不连续。", "在 NCHW 与 NHWC 等布局之间重排轴。", "只需要交换两个维度；transpose 更直接。", "y = torch.permute(x, (0, 2, 3, 1))", ["torch.transpose", "torch.movedim"], { level: "入门", sideEffect: "不修改输入；结果通常共享存储且可能是非连续视图。" }),
  guide("torch.cat", "沿已有维度首尾拼接多个 Tensor", "非空 Tensor 序列和已有维度 dim。", "一个拼接后的新 Tensor。", "除 dim 外各维大小必须匹配；dim 的大小相加。", "cat 不会新增维度；想把多个同 shape Tensor 组成新轴应使用 stack。", "合并批次、序列片段或特征。", "输入需要保留为一个新的独立维度。", "y = torch.cat([a, b], dim=0)", ["torch.stack"], { level: "入门" }),
  guide("torch.stack", "沿一个新维度堆叠多个 Tensor", "shape 完全相同的 Tensor 序列，以及新维度位置 dim。", "一个 rank 增加 1 的新 Tensor。", "在 dim 处插入大小为输入数量的新维度。", "所有输入 shape 必须完全一致；若只需延长已有轴请用 cat。", "把多个样本、时间步或预测结果组成批次。", "输入 shape 不同，或你不希望增加 rank。", "y = torch.stack([a, b], dim=0)", ["torch.cat"], { level: "入门" }),
  guide("torch.split", "按给定大小沿一个维度切分 Tensor", "输入 Tensor、每块大小或每块大小列表、dim。", "Tensor 元组；最后一块可以比固定 split_size 小。", "各块除 dim 外与输入相同，dim 大小按切分规则分配。", "固定块大小不保证整除；它与“切成几块”的 chunk 不同。", "按已知长度拆分特征、序列或批次。", "你只知道想要大致相等的块数。", "parts = torch.split(x, 3, dim=0)", ["torch.chunk"], { level: "常用" }),
  guide("torch.chunk", "把一个维度尽量均匀地切成若干块", "输入 Tensor、期望块数 chunks 和 dim。", "Tensor 元组；在某些尺寸下实际块数可能少于请求值。", "各块尽量接近，除 dim 外 shape 不变。", "返回块数不总是等于 chunks；代码不能盲目解包固定数量。", "并行处理，或只知道想切成大致相等的块数。", "你需要严格控制每块长度或返回块数。", "parts = torch.chunk(x, chunks=3, dim=0)", ["torch.split"], { level: "常用" }),
  guide("torch.gather", "按 index 在指定维度逐位置取值", "input、dim 和 int64 index；index 与 input 需要满足特定 rank/尺寸约束。", "取出的 Tensor，元素来自 input 在 dim 上的索引。", "与 index.shape 相同。", "gather 不会广播 input 与 index；index 通常必须是 torch.int64。", "按每行/每位置的不同索引收集元素。", "只需用同一组一维索引选择整行或整列。", "y = torch.gather(x, 1, index)", ["torch.index_select", "torch.take_along_dim"], { level: "常用" }),
  guide("torch.index_select", "用同一个一维索引表选择某个维度", "input、dim 和一维 int64 index。", "按 index 顺序复制所选切片的新 Tensor。", "指定 dim 的大小变为 len(index)，其他维不变。", "index 必须是一维；每个位置需要不同索引时应使用 gather。", "选择指定行、列、样本或通道。", "索引规则因输出位置而异。", "y = torch.index_select(x, 0, torch.tensor([2, 0]))", ["torch.gather"], { level: "常用" }),
  guide("torch.masked_select", "取出布尔掩码为 True 的所有元素", "input 和可广播到 input 的 bool mask。", "按遍历顺序排列的一维 Tensor。", "始终是一维，长度等于 True 的数量。", "它会丢失原 shape；若要在原位置二选一通常用 where。", "过滤满足条件的元素做统计或检查。", "你需要保留输入形状。", "picked = torch.masked_select(x, x > 0)", ["torch.where"], { level: "常用" }),
  guide("torch.where", "按布尔条件在两个候选值之间逐元素选择", "condition、input、other；三者需要可广播。", "选择后的 Tensor。", "等于 condition、input、other 广播后的共同 shape。", "单参数 where 返回索引元组，语义不同；这里的三参数版本才是逐元素选择。", "条件替换、裁剪、自定义分段函数。", "只想取出所有满足条件的元素并压成一维。", "y = torch.where(x > 0, x, torch.zeros_like(x))", ["torch.masked_select", "torch.clamp"], { level: "入门" }),
  guide("torch.take", "把输入视为一维后按扁平索引取值", "input 和任意 shape 的 int64 index。", "从扁平输入取出的 Tensor。", "与 index.shape 相同。", "索引针对扁平顺序，不保留原维度含义；多维按轴索引常用 gather/index_select。", "已知扁平位置，需要恢复指定元素。", "索引表达的是行、列或某一维坐标。", "y = torch.take(x, torch.tensor([0, 3]))", ["torch.gather", "torch.index_select"], { level: "常用" }),

  // 26–45 · 逐元素数学与归约
  guide("torch.add", "逐元素相加，并支持广播", "input、other，以及可选 alpha（计算 input + alpha × other）。", "相加后的 Tensor。", "等于两个输入广播后的共同 shape。", "整数 Tensor 与浮点数、不同 dtype/device 的组合要先确认类型提升规则。", "叠加偏置、残差、增量或两个同义张量。", "你实际需要矩阵乘法。", "y = torch.add(x, other, alpha=1)", ["torch.sub", "torch.Tensor.add_"], { level: "入门" }),
  guide("torch.sub", "逐元素相减，并支持广播", "input、other，以及可选 alpha（计算 input - alpha × other）。", "相减后的 Tensor。", "等于两个输入广播后的共同 shape。", "方向很重要：sub(input, other) 与 sub(other, input) 符号相反。", "计算误差、差分或去除偏置。", "你需要绝对误差或距离；还要再做 abs/norm。", "delta = torch.sub(prediction, target)", ["torch.add", "torch.abs"], { level: "入门" }),
  guide("torch.mul", "逐元素相乘，并支持广播", "两个可广播的 Tensor 或 Tensor 与标量。", "逐位置乘积。", "等于两个输入广播后的共同 shape。", "mul 不是矩阵乘法；线性代数乘法使用 matmul/mm。", "应用掩码、缩放、门控或逐元素权重。", "你希望做点积或矩阵乘法。", "y = torch.mul(x, scale)", ["torch.matmul", "torch.div"], { level: "入门" }),
  guide("torch.div", "逐元素相除，并支持广播", "input、other，以及可选 rounding_mode。", "逐位置商。", "等于两个输入广播后的共同 shape。", "整数输入在不同 rounding_mode 下结果不同；除零会产生 inf/nan 或错误，取决于 dtype。", "归一化、计算比率或缩放。", "分母可能为零且尚未定义处理策略。", "ratio = torch.div(x, denominator)", ["torch.mul", "torch.true_divide"], { level: "入门" }),
  guide("torch.pow", "逐元素做幂运算", "底数 input 和指数 exponent；两者可为标量或可广播 Tensor。", "逐位置幂结果。", "等于底数与指数广播后的共同 shape。", "负底数配非整数指数会产生 nan；大指数容易溢出。", "平方、开方的推广或幂律变换。", "只需要平方或平方根且想表达得更直接。", "y = torch.pow(x, 2)", ["torch.sqrt", "torch.exp"], { level: "入门" }),
  guide("torch.abs", "逐元素取绝对值或复数模", "数值 Tensor。", "非负实数 Tensor。", "与输入相同。", "在 x=0 处不可微，autograd 使用约定的次梯度；复数输入会输出实数 dtype。", "绝对误差、幅值或去除符号。", "你需要向量整体长度；应使用 norm。", "magnitude = torch.abs(x)", ["torch.linalg.vector_norm"], { level: "入门" }),
  guide("torch.exp", "逐元素计算自然指数 eˣ", "浮点或复数 Tensor。", "指数结果 Tensor。", "与输入相同。", "较大正数会溢出；softmax/logsumexp 等任务应使用数值稳定实现。", "指数变换、概率模型或公式实现。", "你正在手写 softmax 或 log(1+exp(x))。", "y = torch.exp(x)", ["torch.log", "torch.sigmoid"], { level: "入门" }),
  guide("torch.log", "逐元素计算自然对数", "正的浮点或复数 Tensor。", "自然对数结果 Tensor。", "与输入相同。", "实数输入 0 得 -inf，负数得 nan；必要时先 clamp 或使用专用稳定函数。", "把乘法变加法、计算对数概率或尺度。", "输入可能非正且没有明确处理规则。", "y = torch.log(x)", ["torch.exp", "torch.log1p"], { level: "入门" }),
  guide("torch.sqrt", "逐元素计算平方根", "非负实数或复数 Tensor。", "平方根 Tensor。", "与输入相同。", "实数负输入产生 nan；在 0 附近梯度可能很大。", "标准差、欧氏长度或反解平方。", "只想求向量整体范数。", "y = torch.sqrt(x)", ["torch.pow", "torch.linalg.vector_norm"], { level: "入门" }),
  guide("torch.clamp", "把每个元素限制在上下界内", "input，以及 min、max 至少一个边界。", "被截断到区间内的 Tensor。", "与输入相同。", "超出区间的区域梯度通常为 0；用它硬截概率可能妨碍学习。", "限制数值范围、防止非法输入或实现简单阈值。", "你需要平滑饱和函数或只想做条件选择。", "y = torch.clamp(x, min=0, max=1)", ["torch.where", "torch.nn.functional.relu"], { level: "入门" }),
  guide("torch.round", "逐元素舍入到最近整数值", "数值 Tensor；decimals 可控制保留小数位。", "舍入后的 Tensor，通常保留输入 dtype。", "与输入相同。", "正好在中点时采用 ties-to-even；梯度几乎处处为 0。", "显示、离散化前处理或数值格式化。", "训练中希望离散操作仍有有效梯度。", "y = torch.round(x)", ["torch.floor", "torch.ceil"], { level: "入门" }),
  guide("torch.floor", "逐元素向负无穷方向取整", "数值 Tensor。", "不大于输入的最大整数值。", "与输入相同。", "负数的 floor 不是截断：floor(-1.2) = -2。", "桶编号、下界或网格坐标。", "你想向 0 截断。", "y = torch.floor(x)", ["torch.ceil", "torch.trunc"], { level: "入门" }),
  guide("torch.ceil", "逐元素向正无穷方向取整", "数值 Tensor。", "不小于输入的最小整数值。", "与输入相同。", "负数仍向正无穷取整：ceil(-1.2) = -1。", "计算需要覆盖的块数、容量上界或网格尺寸。", "你想按最近整数舍入。", "y = torch.ceil(x)", ["torch.floor", "torch.round"], { level: "入门" }),
  guide("torch.sum", "沿指定维度求和", "input、可选 dim、keepdim 和 dtype。", "总和 Tensor；未指定 dim 时通常归约所有元素。", "被归约维消失；keepdim=True 时保留为大小 1。", "整数溢出、空 Tensor、dim 与 keepdim 都会影响结果形态。", "累计数量、损失或沿轴汇总特征。", "需要平均值或数值稳定的对数求和。", "total = torch.sum(x, dim=1)", ["torch.mean", "torch.prod"], { level: "入门" }),
  guide("torch.mean", "沿指定维度求算术平均", "浮点/复数 input、可选 dim、keepdim 和 dtype。", "平均值 Tensor。", "被归约维消失；keepdim=True 时保留为大小 1。", "整数 Tensor 不能直接求 mean；空切片会产生 nan。", "平均损失、批次统计或降噪汇总。", "样本权重不同，或你需要忽略 NaN。", "avg = torch.mean(x, dim=0)", ["torch.sum", "torch.nanmean"], { level: "入门" }),
  guide("torch.max", "求最大值，或沿维度同时返回最大值与索引", "input；可选 dim 和 keepdim；双输入重载则做逐元素最大。", "无 dim 时返回最大值；有 dim 时返回 values/indices；双输入时返回 Tensor。", "取决于重载：全局为标量，有 dim 时删除或保留该维。", "torch.max 有多种重载；先确认你需要全局值、逐维值+索引，还是逐元素比较。", "找峰值、类别索引或逐位置上界。", "你只需要索引；argmax 更清楚。", "values, indices = torch.max(x, dim=1)", ["torch.argmax", "torch.amax"], { level: "入门" }),
  guide("torch.min", "求最小值，或沿维度同时返回最小值与索引", "input；可选 dim 和 keepdim；双输入重载则做逐元素最小。", "无 dim 时返回最小值；有 dim 时返回 values/indices；双输入时返回 Tensor。", "取决于重载：全局为标量，有 dim 时删除或保留该维。", "torch.min 有多种重载；不要把逐维归约和逐元素 minimum 混为一谈。", "找最低值、最近候选或逐位置下界。", "你只需要索引；argmin 更清楚。", "values, indices = torch.min(x, dim=1)", ["torch.argmin", "torch.amin"], { level: "入门" }),
  guide("torch.argmax", "返回最大值所在位置的索引", "input、可选 dim 和 keepdim。", "dtype 通常为 int64 的索引 Tensor。", "未指定 dim 时输入先展平并返回标量索引；指定 dim 时归约该维。", "并列最大值通常返回第一个位置；无 dim 时索引是扁平索引。", "分类预测、峰值定位或离散选择。", "训练时需要可微的软选择。", "labels = torch.argmax(logits, dim=1)", ["torch.max", "torch.topk"], { level: "入门", autograd: NO_TENSOR_GRADIENT }),
  guide("torch.argmin", "返回最小值所在位置的索引", "input、可选 dim 和 keepdim。", "dtype 通常为 int64 的索引 Tensor。", "未指定 dim 时输入先展平并返回标量索引；指定 dim 时归约该维。", "并列最小值通常返回第一个位置；无 dim 时索引是扁平索引。", "最近距离、最低代价或最小误差位置。", "训练时需要可微的软选择。", "index = torch.argmin(distance, dim=1)", ["torch.min", "torch.kthvalue"], { level: "入门", autograd: NO_TENSOR_GRADIENT }),
  guide("torch.prod", "沿指定维度求乘积", "input、可选 dim、keepdim 和 dtype。", "乘积 Tensor。", "被归约维消失；keepdim=True 时保留为大小 1。", "长序列易上溢或下溢，含 0 时梯度行为也需检查。", "组合独立比例、计算尺寸乘积或概率乘积的小规模情况。", "大量小概率连乘；应改在对数域求和。", "product = torch.prod(x, dim=0)", ["torch.sum"], { level: "常用" }),

  // 46–55 · 线性代数
  guide("torch.matmul", "按输入 rank 执行向量点积、矩阵乘法或批量矩阵乘法", "两个 Tensor；最后两维遵循矩阵乘法，前导批次维可广播。", "乘法结果 Tensor。", "二维时 (M,K)×(K,N)→(M,N)；一维和批量情况按 matmul 规则增删维。", "rank 不同会触发不同语义；不要只凭“乘法”二字猜 shape。", "通用向量/矩阵/批量矩阵乘法。", "只做逐元素乘法，或你希望严格限制为二维/三维。", "y = torch.matmul(a, b)", ["torch.mul", "torch.mm", "torch.bmm"], { level: "入门" }),
  guide("torch.mm", "执行严格的二维矩阵乘法", "二维矩阵 input 和 mat2，内维必须相等。", "二维矩阵乘积。", "(M,K)×(K,N)→(M,N)。", "mm 不广播 batch；三维批量输入应用 bmm 或 matmul。", "你想用 rank 约束明确表达二维矩阵乘法。", "输入含批次维或向量。", "y = torch.mm(a, b)", ["torch.matmul", "torch.bmm"], { level: "常用" }),
  guide("torch.bmm", "对两个三维批次逐批做矩阵乘法", "shape 为 (B,M,K) 和 (B,K,N) 的两个三维 Tensor。", "每个批次的矩阵乘积。", "输出为 (B,M,N)。", "bmm 不广播 batch 维；两个输入的 B 必须相同。", "批量注意力、批量几何变换或明确的三维乘法。", "批次维需要广播，或输入 rank 不是 3。", "y = torch.bmm(a, b)", ["torch.matmul", "torch.mm"], { level: "常用" }),
  guide("torch.dot", "计算两个一维 Tensor 的点积", "两个元素数量相同的一维 Tensor。", "一个标量 Tensor。", "输出为 0 维 Tensor。", "dot 只接受一维输入；高维不会自动展平。", "向量内积、相似度公式的一部分。", "输入是矩阵或批量向量。", "score = torch.dot(a, b)", ["torch.matmul", "torch.vdot"], { level: "入门" }),
  guide("torch.einsum", "用爱因斯坦求和字符串描述乘法、归约和维度重排", "方程字符串和一个或多个 Tensor。", "由输出下标决定的 Tensor。", "只出现在输出部分的下标被保留，其余重复下标被求和。", "方程写错仍可能得到 shape 合法但含义错误的结果；先逐字核对每个下标。", "复杂张量收缩、注意力或用一条公式表达多步轴运算。", "简单矩阵乘法或转置已有更直白函数。", "y = torch.einsum('bij,bjk->bik', a, b)", ["torch.matmul"], { level: "进阶" }),
  guide("torch.norm", "计算向量或矩阵范数的旧式通用入口", "input、p、dim、keepdim 等。", "范数 Tensor。", "归约 dim 后删除或保留相应维度。", "torch.norm 已偏向兼容用途；新代码优先选 torch.linalg.vector_norm / matrix_norm。", "维护旧代码或需要兼容既有调用。", "编写新代码且能明确是向量范数还是矩阵范数。", "length = torch.norm(x, p=2, dim=-1)", ["torch.linalg.vector_norm", "torch.linalg.matrix_norm"], { level: "常用" }),
  guide("torch.linalg.vector_norm", "沿指定维度计算向量范数", "input、ord、dim、keepdim 和可选 dtype。", "非负范数 Tensor。", "归约 dim；keepdim=True 时保留为 1。", "dim 为多个维度时会先把这些维看作一个向量；不要和矩阵范数混淆。", "向量长度、归一化分母或梯度裁剪统计。", "你需要矩阵的谱范数、核范数等矩阵语义。", "length = torch.linalg.vector_norm(x, ord=2, dim=-1)", ["torch.norm", "torch.linalg.matrix_norm"], { level: "常用" }),
  guide("torch.linalg.solve", "直接求解线性方程 A X = B", "可逆方阵 A 和右端项 B；支持批次维。", "方程解 X。", "A 为 (...,n,n)，B 为 (...,n) 或 (...,n,k)，输出与 B 的核心 shape 对应。", "A 奇异或病态会失败或产生不可靠结果；通常不要显式求逆后再乘 B。", "已知线性系统并需要数值稳定地求解。", "你只想计算矩阵逆，或 A 不是方阵。", "x = torch.linalg.solve(A, b)", ["torch.linalg.inv", "torch.linalg.lstsq"], { level: "进阶" }),
  guide("torch.linalg.inv", "计算方阵的逆矩阵", "可逆方阵或一批方阵。", "逆矩阵 Tensor。", "与输入相同。", "显式求逆通常比 solve 更慢、更不稳定；解 AX=B 时优先 solve。", "确实需要逆矩阵本身，且已检查可逆性。", "只是为了把逆矩阵乘到右端项。", "A_inv = torch.linalg.inv(A)", ["torch.linalg.solve", "torch.linalg.pinv"], { level: "进阶" }),
  guide("torch.linalg.svd", "把矩阵分解为 U、奇异值 S 和 Vh", "矩阵或一批矩阵，以及 full_matrices。", "命名元组 (U, S, Vh)。", "对 (...,m,n)，S 为 (...,min(m,n))；U/Vh 大小受 full_matrices 影响。", "奇异向量在重复奇异值附近不唯一，相关梯度可能不稳定。", "降维、低秩近似、条件分析或伪逆。", "只需要奇异值；svdvals 更省。", "U, S, Vh = torch.linalg.svd(A, full_matrices=False)", ["torch.linalg.svdvals", "torch.pca_lowrank"], { level: "进阶" }),

  // 56–65 · 统计、排序与计数
  guide("torch.std", "沿指定维度计算标准差", "input、可选 dim、correction 和 keepdim。", "标准差 Tensor。", "归约 dim；keepdim=True 时保留为 1。", "默认 correction 会影响小样本结果；样本数不大于 correction 时结果可能无效。", "描述波动、标准化或监控特征分布。", "你需要方差或均值也要一起计算。", "scale = torch.std(x, dim=0, correction=1)", ["torch.var", "torch.std_mean"], { level: "常用" }),
  guide("torch.var", "沿指定维度计算方差", "input、可选 dim、correction 和 keepdim。", "方差 Tensor。", "归约 dim；keepdim=True 时保留为 1。", "correction=0 与 1 分别对应不同分母；不要混淆总体方差和样本方差。", "统计波动、归一化或不确定性。", "你实际需要标准差，或希望同时得到均值。", "variance = torch.var(x, dim=0, correction=1)", ["torch.std", "torch.var_mean"], { level: "常用" }),
  guide("torch.median", "求中位数，或沿维度返回中位数与索引", "input；可选 dim 和 keepdim。", "全局时为中位数 Tensor；沿 dim 时返回 values/indices。", "全局为标量；沿 dim 时归约该维。", "偶数个元素时 torch.median 取两个中间值中较小者，不等同于 quantile(..., 0.5) 的默认插值。", "需要抗离群点的中心统计，或沿轴找中位位置。", "你需要标准线性插值的 50% 分位数。", "values, indices = torch.median(x, dim=1)", ["torch.quantile"], { level: "常用" }),
  guide("torch.quantile", "计算一个或多个分位数", "input、q∈[0,1]、可选 dim、keepdim 和 interpolation。", "分位值 Tensor。", "q 为标量时归约 dim；q 为一维时会在前面增加分位数维。", "插值方式会改变结果；大 Tensor 上通常需要排序，成本不低。", "百分位阈值、稳健统计或分布摘要。", "只需要最小/最大/固定第 k 个值。", "q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))", ["torch.median", "torch.kthvalue"], { level: "常用" }),
  guide("torch.sort", "沿指定维度完整排序，并返回值与原索引", "input、dim、descending、stable。", "命名元组 (values, indices)。", "values 与 indices 都与输入同 shape。", "descending 控制方向，stable 控制相等元素的先后；完整排序比 topk 更贵。", "后续需要全部有序值及其来源位置。", "只需要最大或最小的少量元素。", "values, indices = torch.sort(x, dim=-1)", ["torch.argsort", "torch.topk"], { level: "常用" }),
  guide("torch.argsort", "返回能把输入排序的索引", "input、dim、descending、stable。", "dtype 通常为 int64 的索引 Tensor。", "与输入同 shape。", "它不直接返回排序后的值；需要值时可用 sort 或再 gather。", "需要排序顺序去重排其他对齐数据。", "同时需要有序值与索引。", "order = torch.argsort(x, dim=-1)", ["torch.sort"], { level: "常用", autograd: NO_TENSOR_GRADIENT }),
  guide("torch.topk", "沿指定维度取最大或最小的 k 个值及索引", "input、k、dim、largest、sorted。", "命名元组 (values, indices)。", "指定 dim 的大小变为 k，两个返回 Tensor shape 相同。", "k 不能超过该维长度；并列值的索引顺序不保证稳定。", "分类 Top-K、候选筛选或只保留少量极值。", "你需要完整排序，或依赖并列元素的稳定顺序。", "values, indices = torch.topk(x, k=3, dim=-1)", ["torch.sort", "torch.kthvalue"], { level: "常用" }),
  guide("torch.kthvalue", "沿指定维度找第 k 小的值及索引", "input、从 1 开始计数的 k、dim、keepdim。", "命名元组 (values, indices)。", "归约 dim；keepdim=True 时保留为 1。", "k 从 1 开始而不是 0；并列值可能返回任一合法索引。", "选择固定秩统计量而无需完整排序。", "你要前 k 个结果，或要按分位数插值。", "values, indices = torch.kthvalue(x, k=2, dim=-1)", ["torch.topk", "torch.quantile"], { level: "常用" }),
  guide("torch.unique", "去重，并可返回逆映射和计数", "input、sorted、return_inverse、return_counts、dim。", "唯一值，以及按开关可选的 inverse_indices/counts。", "未指定 dim 时通常展平后返回一维唯一值；指定 dim 时按切片去重。", "CPU/CUDA 实现常会排序，即使 sorted=False；性能敏感且输入已分组时看 unique_consecutive。", "类别压缩、去重、频数统计或建立编码映射。", "只想删除相邻重复项且数据已排序/分组。", "values, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)", ["torch.unique_consecutive", "torch.bincount"], { level: "常用", autograd: "去重和索引映射通常不提供有意义的输入梯度。" }),
  guide("torch.bincount", "统计每个非负整数出现的次数或加权和", "一维非负整数 input、可选 weights 和 minlength。", "从 0 到最大值的计数/权重和 Tensor。", "长度至少为 minlength，通常为 max(input)+1。", "input 必须是一维非负整数；类别编号很大且稀疏时会分配很长结果。", "类别频数、直方图或分组加权求和。", "类别键很稀疏、包含负数或不是整数。", "counts = torch.bincount(labels, minlength=num_classes)", ["torch.unique", "torch.histc"], { level: "常用", autograd: NO_TENSOR_GRADIENT }),

  // 66–80 · 神经网络函数
  guide("torch.nn.functional.relu", "逐元素把负数变为 0，正数保持不变", "Tensor 和可选 inplace。", "ReLU 激活 Tensor。", "与输入相同。", "inplace=True 会覆盖输入，可能破坏 autograd 需要的中间值。", "隐藏层引入简单非线性和稀疏激活。", "需要负区间仍保留梯度，或模型对平滑性敏感。", "y = torch.nn.functional.relu(x)", ["torch.nn.ReLU", "torch.nn.functional.gelu"], { level: "入门" }),
  guide("torch.nn.functional.gelu", "用平滑门控近似保留较大的正输入", "Tensor 和 approximate 模式。", "GELU 激活 Tensor。", "与输入相同。", "approximate='tanh' 与精确版本数值略有差异；需与模型训练配置一致。", "Transformer 和需要平滑激活的网络。", "你需要最简单、廉价的硬阈值激活。", "y = torch.nn.functional.gelu(x, approximate='none')", ["torch.nn.GELU", "torch.nn.functional.relu"], { level: "常用" }),
  guide("torch.sigmoid", "把每个实数映射到 0 与 1 之间", "浮点或复数 Tensor。", "逐元素 sigmoid 结果。", "与输入相同。", "大幅值区域梯度接近 0；二分类训练不要先 sigmoid 再 BCE，优先 logits 版本损失。", "输出独立概率、门控系数或二分类推理概率。", "直接把多类别 logits 归一化成总和为 1 的概率。", "probability = torch.sigmoid(logits)", ["torch.nn.functional.binary_cross_entropy_with_logits", "torch.nn.functional.softmax"], { level: "入门" }),
  guide("torch.nn.functional.softmax", "沿指定维度把 logits 归一化为概率分布", "input、明确的 dim 和可选 dtype。", "每个 dim 切片和为 1 的概率 Tensor。", "与输入相同。", "dim 选错会在错误的轴上归一化；交叉熵训练通常不要手动先 softmax。", "多类别推理概率、注意力权重或归一化权重。", "要把概率传给 cross_entropy，或需要更稳定的对数概率。", "prob = torch.nn.functional.softmax(logits, dim=-1)", ["torch.nn.functional.log_softmax", "torch.nn.functional.cross_entropy"], { level: "入门" }),
  guide("torch.nn.functional.log_softmax", "稳定地计算 softmax 后的对数", "input、明确的 dim 和可选 dtype。", "对数概率 Tensor。", "与输入相同。", "不要写成 torch.log(softmax(x))；组合实现更稳定。", "NLLLoss、对数概率或需要数值稳定的归一化。", "下游需要普通概率而不是对数概率。", "log_prob = torch.nn.functional.log_softmax(logits, dim=-1)", ["torch.nn.functional.softmax", "torch.nn.functional.nll_loss"], { level: "常用" }),
  guide("torch.nn.functional.dropout", "训练时随机置零并缩放其余元素", "input、丢弃概率 p、training、inplace。", "随机掩码后的 Tensor。", "与输入相同。", "函数式 dropout 必须正确传 training=self.training；否则评估时也可能继续随机丢弃。", "训练阶段做正则化，减少特征共适应。", "推理阶段，或你忘了传递模块的 training 状态。", "y = torch.nn.functional.dropout(x, p=0.5, training=self.training)", ["torch.nn.Dropout"], { level: "常用", sideEffect: "默认不修改输入，但训练模式会推进随机数生成器状态；inplace=True 还会覆盖输入。" }),
  guide("torch.nn.functional.linear", "计算 y = x Aᵀ + b", "input、weight(out_features,in_features) 和可选 bias。", "线性变换后的 Tensor。", "输入最后一维从 in_features 变为 out_features，其他前导维保留。", "weight 的布局是 (out,in)，不是 (in,out)；最后一维必须匹配。", "直接使用已有权重做全连接投影。", "希望参数自动注册到 Module 并参与 state_dict。", "y = torch.nn.functional.linear(x, weight, bias)", ["torch.nn.Linear", "torch.matmul"], { level: "常用" }),
  guide("torch.nn.functional.conv2d", "用二维卷积核扫描图像或特征图", "input(N,Cin,H,W)、weight(Cout,Cin/groups,kH,kW)、可选 bias/stride/padding/dilation/groups。", "卷积输出 Tensor。", "输出为 (N,Cout,Hout,Wout)，空间尺寸由卷积公式决定。", "通道必须满足 groups 约束；padding='same' 与 stride 组合也有限制。", "已有卷积权重，或实现无状态卷积计算。", "希望卷积参数自动注册和初始化。", "y = torch.nn.functional.conv2d(x, weight, padding=1)", ["torch.nn.Conv2d"], { level: "进阶" }),
  guide("torch.nn.functional.max_pool2d", "在每个二维窗口取最大值", "input、kernel_size、stride、padding、dilation、ceil_mode。", "下采样后的 Tensor；可选返回索引。", "N/C 通常不变，H/W 按池化公式缩小。", "默认不处理不完整尾窗；ceil_mode=True 才可能覆盖额外窗口。", "保留局部强响应并降低空间分辨率。", "需要保留平均信息或可学习下采样。", "y = torch.nn.functional.max_pool2d(x, kernel_size=2)", ["torch.nn.functional.avg_pool2d", "torch.nn.MaxPool2d"], { level: "常用" }),
  guide("torch.nn.functional.avg_pool2d", "在每个二维窗口取平均值", "input、kernel_size、stride、padding、ceil_mode、count_include_pad。", "下采样后的 Tensor。", "N/C 通常不变，H/W 按池化公式缩小。", "padding 是否计入平均分母由 count_include_pad 决定，边缘数值会因此不同。", "平滑下采样或提取局部平均响应。", "你需要保留最强激活或自适应到固定输出尺寸。", "y = torch.nn.functional.avg_pool2d(x, kernel_size=2)", ["torch.nn.functional.max_pool2d", "torch.nn.functional.adaptive_avg_pool2d"], { level: "常用" }),
  guide("torch.nn.functional.cross_entropy", "把 logits 与类别目标合成多分类交叉熵", "未归一化 logits、类别索引或概率目标，以及 weight/ignore_index/reduction。", "按 reduction 得到标量损失或逐位置损失。", "reduction='none' 时保留样本/空间位置；mean/sum 通常得到标量。", "输入应是 logits，不要先 softmax；类别索引目标通常要 int64 且不含类别维。", "单标签多分类训练。", "多标签独立二分类，或输入已经是对数概率。", "loss = torch.nn.functional.cross_entropy(logits, target)", ["torch.nn.CrossEntropyLoss", "torch.nn.functional.nll_loss"], { level: "常用" }),
  guide("torch.nn.functional.mse_loss", "计算预测与目标的平方误差", "可广播的 input 和 target，以及 reduction。", "逐元素平方误差或归约后的损失。", "none 时为广播后 shape；mean/sum 时通常为标量。", "广播可能悄悄掩盖错误 target shape；离群点会被平方放大。", "回归、重建误差或连续值拟合。", "目标含强离群点，或任务是分类。", "loss = torch.nn.functional.mse_loss(prediction, target)", ["torch.nn.functional.l1_loss", "torch.nn.functional.smooth_l1_loss"], { level: "入门" }),
  guide("torch.nn.functional.l1_loss", "计算预测与目标的绝对误差", "可广播的 input 和 target，以及 reduction。", "逐元素绝对误差或归约后的损失。", "none 时为广播后 shape；mean/sum 时通常为标量。", "在误差恰好为 0 处不可微，使用次梯度；广播仍可能掩盖 shape 错误。", "希望比 MSE 更抗离群点的回归。", "需要误差接近 0 时更平滑的梯度。", "loss = torch.nn.functional.l1_loss(prediction, target)", ["torch.nn.functional.mse_loss", "torch.nn.functional.smooth_l1_loss"], { level: "入门" }),
  guide("torch.nn.functional.binary_cross_entropy_with_logits", "稳定地把 sigmoid 与二元交叉熵合并", "任意实数 logits、同 shape target、可选 weight/pos_weight/reduction。", "逐元素或归约后的二元损失。", "none 时与广播后输入一致；mean/sum 时通常为标量。", "不要先 sigmoid；target 通常是浮点 0/1，pos_weight 与 weight 的广播含义不同。", "二分类或多标签分类训练。", "互斥的单标签多分类。", "loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)", ["torch.nn.BCEWithLogitsLoss", "torch.sigmoid"], { level: "常用" }),
  guide("torch.nn.functional.embedding", "用整数索引从词表权重中查向量", "int64 indices、weight(num_embeddings,embedding_dim)，以及 padding_idx 等选项。", "每个索引对应的嵌入向量。", "输出 shape = indices.shape + (embedding_dim,)。", "索引必须在范围内；函数式调用不会像 nn.Embedding 构造器那样自动初始化 padding 行。", "已有嵌入权重，需要查表。", "希望权重作为模块参数注册、初始化和保存。", "vectors = torch.nn.functional.embedding(token_ids, weight)", ["torch.nn.Embedding"], { level: "常用" }),

  // 81–90 · autograd、设备、随机性、编译与序列化
  guide("torch.autograd.grad", "直接计算指定输出对指定输入的梯度并返回", "outputs、inputs、可选 grad_outputs/create_graph/retain_graph/allow_unused。", "与 inputs 对应的梯度元组。", "每个梯度通常与对应 input 同 shape。", "它不会自动写入 .grad；非标量输出通常要提供 grad_outputs，重复高阶求导要理解 create_graph。", "函数式求导、梯度惩罚、雅可比相关计算。", "普通训练只需把叶子参数梯度累积到 .grad。", "(dx,) = torch.autograd.grad(loss, x)", ["torch.Tensor.backward", "torch.autograd.functional.jacobian"], { level: "进阶", sideEffect: "默认返回梯度而不累加到 input.grad；可能释放用于反向的计算图。" }),
  guide("torch.no_grad", "在作用域内关闭反向图记录", "作为 with 上下文或装饰器使用；不接收待计算 Tensor。", "上下文管理器/装饰器；内部表达式仍返回正常结果。", "不直接改变 Tensor shape。", "它不等于 model.eval()；前者控制梯度记录，后者控制 Dropout/BatchNorm 等模块行为。", "推理、参数手工更新或不需要梯度的评估。", "需要后续对作用域内结果反向传播。", "with torch.no_grad():\n    prediction = model(x)", ["torch.enable_grad", "torch.inference_mode"], { level: "入门", call: "作为 `with torch.no_grad():` 或装饰器使用。", autograd: "作用域内新运算通常不进入反向图；工厂函数的 requires_grad 参数有单独规则。", sideEffect: "临时修改当前线程的梯度记录模式，退出作用域后恢复。" }),
  guide("torch.enable_grad", "在外层禁用梯度时临时重新开启记录", "作为 with 上下文或装饰器使用。", "上下文管理器/装饰器。", "不直接改变 Tensor shape。", "只有输入本身允许求导且运算可微时，重新开启记录才会产生有效梯度。", "在 no_grad 环境中的局部训练或敏感性计算。", "整个流程都不需要梯度。", "with torch.enable_grad():\n    loss = model(x).sum()", ["torch.no_grad"], { level: "常用", call: "作为 `with torch.enable_grad():` 或装饰器使用。", autograd: "作用域内恢复 autograd 记录。", sideEffect: "临时修改当前线程的梯度记录模式，退出作用域后恢复。" }),
  guide("torch.compiler.is_compiling", "查询当前代码是否正由 torch.compile 捕获", "不接收 Tensor；读取当前编译上下文。", "一个 Python bool。", "没有 Tensor shape。", "它适合做兼容分支，但过度依赖编译状态可能让 eager 与 compiled 语义分叉。", "库代码需要在编译捕获期间避开不支持路径。", "只是想判断 CUDA、设备或某个算子是否可用。", "inside_compile = torch.compiler.is_compiling()", ["torch.compile"], { level: "进阶", autograd: NO_TENSOR_GRADIENT, sideEffect: "只读取当前编译上下文，不修改 Tensor 或编译状态。" }),
  guide("torch.cuda.is_available", "判断当前运行时能否使用 CUDA", "无参数；读取 PyTorch 构建、驱动与当前环境状态。", "一个 Python bool。", "没有 Tensor shape。", "True 也不保证任意分配都成功；显存、设备权限和具体算子仍可能失败。", "选择 cpu/cuda 设备或给出环境提示。", "判断某个 Tensor 当前在哪个设备。", "device = 'cuda' if torch.cuda.is_available() else 'cpu'", ["torch.Tensor.to", "torch.cuda.device_count"], { level: "入门", autograd: NO_TENSOR_GRADIENT, sideEffect: "读取运行时能力，不创建或移动 Tensor。" }),
  guide("torch.manual_seed", "设置 CPU 及相关设备的默认随机种子", "一个整数 seed。", "一个 Generator 对象。", "没有 Tensor shape。", "相同种子不保证跨平台、跨版本和所有非确定算子得到完全一致结果。", "让实验在同一环境中更容易复现。", "需要彼此独立的多个随机流；应创建单独 Generator。", "torch.manual_seed(42)", ["torch.Generator", "torch.use_deterministic_algorithms"], { level: "入门", autograd: NO_TENSOR_GRADIENT, sideEffect: "重置默认随机数生成器状态，会影响后续随机操作。" }),
  guide("torch.compile", "把 Python/PyTorch 可调用对象优化成可复用的编译版本", "函数或 nn.Module，以及 backend/mode/fullgraph/dynamic 等选项。", "语义等价的可调用对象。", "输出 shape 应与原函数一致，具体由实际输入和函数决定。", "首次调用有编译开销；shape、dtype 或 Python 分支变化可能触发重编译或 graph break。", "热点模型已正确运行，希望提升稳态性能。", "仍在频繁调试副作用、动态 Python 控制流或很短的一次性任务。", "compiled_model = torch.compile(model)", ["torch.compiler.is_compiling", "torch.export.export"], { level: "进阶", autograd: "目标函数中的可微 Tensor 运算仍按原语义参与 autograd。", sideEffect: "会创建编译缓存和守卫，首次运行可能启动后端编译。" }),
  guide("torch.save", "把对象序列化写入文件或文件对象", "要保存的 Python 对象和路径/可写二进制文件。", "返回 None。", "不改变 Tensor shape。", "保存整个 Module 会绑定 Python 类路径；长期模型文件通常优先保存 state_dict。", "保存权重、优化器状态、检查点或 Tensor。", "把不可信对象发送给他人，或需要跨语言通用格式。", "torch.save(model.state_dict(), 'model.pt')", ["torch.load", "torch.nn.Module.state_dict"], { level: "常用", autograd: NO_TENSOR_GRADIENT, sideEffect: "写入或覆盖目标文件/流。" }),
  guide("torch.load", "从文件或文件对象反序列化 PyTorch 数据", "路径/可读文件，以及 map_location、weights_only 等。", "保存时的对象结构或权重数据。", "由文件内容决定。", "不要加载不可信文件；反序列化可能执行恶意代码，并应明确 map_location/weights_only。", "恢复自己或可信来源的检查点和权重。", "文件来源不可信，或需要安全的跨语言数据格式。", "state = torch.load('model.pt', map_location='cpu', weights_only=True)", ["torch.save", "torch.nn.Module.load_state_dict"], { level: "常用", autograd: "加载只恢复数据；是否参与 autograd 取决于数据装入后的使用方式。", sideEffect: "读取文件并创建 Python/Tensor 对象，可能占用大量内存。" }),
  guide("torch.compiler.reset", "清空 torch.compile 的编译缓存", "通常无参数。", "通常返回 None。", "没有 Tensor shape。", "这是调试/测试工具，不应放进每个训练迭代；清缓存会失去复用收益。", "排查缓存、守卫或重新编译行为。", "正常训练或推理热路径。", "torch.compiler.reset()", ["torch.compile"], { level: "进阶", autograd: NO_TENSOR_GRADIENT, sideEffect: "清理进程内编译缓存，后续调用可能重新编译。" }),

  // 91–100 · 常见 Tensor 方法
  guide("torch.Tensor.clone", "复制 Tensor 数据，同时保留可微关系", "当前 Tensor；可选 memory_format。", "拥有独立存储的新 Tensor。", "与输入相同。", "clone 不是 detach：结果仍连接原计算图；需要独立梯度历史时常用 detach().clone()。", "避免共享存储，又希望梯度仍回传到原 Tensor。", "只想停止梯度，或不需要复制数据。", "y = x.clone()", ["torch.Tensor.detach", "torch.Tensor.contiguous"], { level: "入门", call: "先有 Tensor `x`，再调用 `x.clone()`。", sideEffect: "不修改输入；返回独立存储。" }),
  guide("torch.Tensor.detach", "从当前计算图分离 Tensor，但通常共享存储", "当前 Tensor；无位置参数。", "requires_grad=False 的 Tensor 视图。", "与输入相同。", "detach 不复制数据；原地修改任一共享存储对象可能影响另一边。", "记录指标、停止某条梯度路径或把值交给非训练逻辑。", "需要安全独立副本；应使用 x.detach().clone()。", "y = x.detach()", ["torch.Tensor.clone", "torch.no_grad"], { level: "入门", call: "先有 Tensor `x`，再调用 `x.detach()`。", autograd: "结果与当前计算图断开，后续梯度不会沿这条边回传。", sideEffect: "不修改输入，但结果通常与输入共享底层存储。" }),
  guide("torch.Tensor.contiguous", "在需要时把 Tensor 变成指定内存格式的连续布局", "当前 Tensor 和可选 memory_format。", "若已连续可返回自身，否则返回连续副本。", "与输入相同。", "它不是无条件复制；不要依赖对象身份判断是否发生复制。", "转置/permute 后某些算子或 view 需要连续内存。", "只为了“保险”在每一步调用，导致多余复制。", "y = x.transpose(0, 1).contiguous()", ["torch.Tensor.view", "torch.Tensor.clone"], { level: "常用", call: "先有 Tensor `x`，再调用 `x.contiguous()`。", sideEffect: "不修改输入；仅在布局不满足时分配并复制。" }),
  guide("torch.Tensor.view", "在共享存储的前提下改变 shape", "当前 Tensor 和目标 shape。", "共享底层数据的视图。", "变为目标 shape，元素总数不变。", "输入的 stride 必须兼容；transpose/permute 后常需 contiguous 或改用 reshape。", "你明确需要无复制视图并理解内存布局。", "你只关心 shape、不关心是否复制。", "y = x.view(2, -1)", ["torch.reshape", "torch.Tensor.contiguous"], { level: "入门", call: "先有 Tensor `x`，再调用 `x.view(...)`。", sideEffect: "不修改输入；结果与输入共享底层存储。" }),
  guide("torch.Tensor.reshape", "尽量用视图改变 shape，必要时自动复制", "当前 Tensor 和目标 shape。", "shape 改变的 Tensor，可能是视图也可能是副本。", "变为目标 shape，元素总数不变。", "不能保证共享存储；若代码依赖别名关系，不要只用 reshape。", "只关心结果形状，希望框架处理连续性。", "必须零拷贝共享存储。", "y = x.reshape(2, -1)", ["torch.Tensor.view", "torch.reshape"], { level: "入门", call: "先有 Tensor `x`，再调用 `x.reshape(...)`。" }),
  guide("torch.Tensor.to", "把 Tensor 转到目标 device/dtype 或匹配另一个 Tensor", "当前 Tensor，以及 device、dtype、non_blocking、copy 等。", "满足目标属性的 Tensor；属性已匹配且 copy=False 时可能返回自身。", "通常与输入相同。", "x.to(...) 不是原地操作，必须接住返回值；dtype 转成整数会停止梯度。", "统一模型与数据的设备/精度。", "只想查询当前 device/dtype，或忘记保存返回值。", "x = x.to(device='cuda', dtype=torch.float32)", ["torch.Tensor.cpu", "torch.cuda.is_available"], { level: "入门", call: "先有 Tensor `x`，再调用并接住 `x = x.to(...)`。", sideEffect: "不原地修改输入；属性相同可返回自身，否则通常复制数据。" }),
  guide("torch.Tensor.cpu", "把 Tensor 放到 CPU 内存", "当前 Tensor 和可选 memory_format。", "CPU Tensor；本来就在 CPU 时可能返回自身。", "与输入相同。", "设备传输可能同步并复制；cpu() 不等于转成 NumPy。", "把结果用于 CPU 算子、保存或转 NumPy。", "数据仍要立即回到 GPU 做计算。", "cpu_x = x.cpu()", ["torch.Tensor.to", "torch.Tensor.numpy"], { level: "入门", call: "先有 Tensor `x`，再调用 `cpu_x = x.cpu()`。", sideEffect: "不修改输入；跨设备时复制数据并可能触发同步。" }),
  guide("torch.Tensor.numpy", "把 CPU Tensor 暴露为 NumPy ndarray", "当前 CPU Tensor；较新版本可用 force 控制必要转换。", "NumPy ndarray，常与 Tensor 共享内存。", "与 Tensor shape 相同。", "需要处理 GPU、requires_grad、conjugate/negative 位；共享内存时修改任一边会影响另一边。", "把推理结果交给 NumPy/绘图库且理解共享关系。", "Tensor 仍在 GPU，或你需要完全独立数组。", "array = x.detach().cpu().numpy()", ["torch.from_numpy", "torch.Tensor.cpu"], { level: "入门", call: "先有 CPU Tensor `x`，再调用 `array = x.numpy()`。", autograd: "NumPy 不记录 autograd；通常先 detach。", sideEffect: "不修改输入；默认结果可能与 CPU Tensor 共享存储。" }),
  guide("torch.Tensor.item", "把单元素 Tensor 取成 Python 标量", "恰好包含一个元素的 Tensor。", "Python int、float 或 bool 等标量。", "没有 Tensor shape。", "多元素 Tensor 会报错；GPU 上调用可能同步，且返回值脱离 autograd。", "日志、条件判断或把单个损失值交给 Python。", "训练热循环中批量取很多元素，或后续仍需梯度。", "loss_value = loss.item()", ["torch.Tensor.tolist"], { level: "入门", call: "先有单元素 Tensor `x`，再调用 `value = x.item()`。", autograd: NO_TENSOR_GRADIENT, sideEffect: "不修改输入；GPU Tensor 上可能触发设备同步。" }),
  guide("torch.Tensor.backward", "从当前 Tensor 反向传播并把梯度累加到叶子 .grad", "当前标量 loss；非标量时还要提供同 shape gradient。", "返回 None。", "不会返回梯度 Tensor；叶子参数的 .grad 与参数 shape 相同。", "梯度默认累加而非覆盖；每轮训练前通常需要 optimizer.zero_grad()。", "标准训练循环从标量损失回传梯度。", "只想函数式获取指定输入梯度而不写入 .grad。", "optimizer.zero_grad(); loss.backward(); optimizer.step()", ["torch.autograd.grad"], { level: "常用", call: "先得到标量损失 `loss`，再调用 `loss.backward()`。", autograd: "沿当前计算图应用链式法则；retain_graph=False 时图通常在反向后释放。", sideEffect: "把梯度累加到相关叶子 Tensor 的 .grad，并可能释放反向图。" }),
];

const GUIDE_BY_NAME = new Map(guides.map((item) => [item.name, item]));

export const CURATED_FUNCTION_GUIDE_COUNT = guides.length;
export const CURATED_FUNCTION_NAMES = Object.freeze(guides.map((item) => item.name));

export function curatedFunctionGuideOf(name: string): CuratedFunctionGuide | null {
  return GUIDE_BY_NAME.get(name) ?? null;
}
