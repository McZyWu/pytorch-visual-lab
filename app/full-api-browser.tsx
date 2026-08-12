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
type Variable = { name: string; meaning: string; sample: string };
type Simulation = { mode: "数值计算" | "梯度计算" | "张量变换" | "对象与状态"; title: string; value: unknown; trace: string[] };
type TensorItem = { key: string; label: string; value: unknown };
type FormulaSpec = { latex: string; spoken: string; explanation: string; symbols: Array<{ symbol: string; meaning: string }> };

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

const xySymbols = [{ symbol: "X", meaning: "输入张量" }, { symbol: "Y", meaning: "输出张量，通常与 X 形状相同" }];
function formulaSpecOf(entry: ApiEntry): FormulaSpec {
  /* eslint-disable no-useless-escape -- LaTeX control sequences are intentionally stored in symbol strings. */
  const n = cleanLeaf(entry), family = familyOf(entry);
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
    const specs:Record<string,[string,string]>={relu:[String.raw`\operatorname{ReLU}(x)=\max(0,x)`,"负数变为 0，非负数保持不变。"],sigmoid:[String.raw`\sigma(x)=\frac{1}{1+\exp(-x)}`,"把任意实数压缩到 0 与 1 之间。"],tanh:[String.raw`\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}`,"把输入压缩到 −1 与 1 之间。"],gelu:[String.raw`\operatorname{GELU}(x)=x\,\Phi(x)`,"用标准正态分布累计概率平滑地门控输入。"],silu:[String.raw`\operatorname{SiLU}(x)=x\,\sigma(x)`,"输入乘以自身的 Sigmoid 门值。"],leaky_relu:[String.raw`\operatorname{LeakyReLU}(x)=\max(0,x)+\alpha\min(0,x)`,"负半轴保留斜率 α，避免梯度完全为零。"],elu:[String.raw`\operatorname{ELU}(x)=\begin{cases}x,&x>0\\ \alpha(\exp(x)-1),&x\le0\end{cases}`,"正半轴保持线性，负半轴指数饱和。"],softplus:[String.raw`\operatorname{Softplus}(x)=\frac1\beta\log(1+\exp(\beta x))`,"ReLU 的平滑近似。"]};
    const [latex,spoken]=specs[n]??[String.raw`Y=f(X)`,"把激活函数逐元素应用到输入张量。"];return {latex,spoken,explanation:"激活函数逐元素计算，因此默认不改变张量形状。",symbols:[{symbol:"x,\,y",meaning:"单个输入元素与对应输出元素"},{symbol:"\alpha,\,\beta",meaning:"部分激活函数使用的斜率或平滑参数"},{symbol:"\Phi",meaning:"标准正态分布的累积分布函数"}]};
  }
  if (family === "reduction") {
    if (/mean/.test(n)) return {latex:String.raw`\mu=\frac{1}{N}\sum_{i=1}^{N}x_i`,spoken:"把指定维度的 N 个元素相加后除以 N。",explanation:"未指定 dim 时归约全部元素；指定 dim 时对每个切片分别计算。",symbols:[{symbol:"x_i",meaning:"归约切片中的第 i 个元素"},{symbol:"N",meaning:"该切片的元素数量"},{symbol:"\mu",meaning:"算术平均值"}]};
    if (/prod/.test(n)) return {latex:String.raw`y=\prod_{i=1}^{N}x_i`,spoken:"把指定维度中的全部元素连续相乘。",explanation:"输出形状取决于 dim 与 keepdim。",symbols:[{symbol:"x_i",meaning:"第 i 个被乘元素"},{symbol:"N",meaning:"归约元素数"},{symbol:"y",meaning:"乘积结果"}]};
    if (/std|var/.test(n)) return {latex:n.includes("std")?String.raw`s=\sqrt{\frac{1}{N-\delta N}\sum_{i=1}^{N}(x_i-\bar{x})^2}`:String.raw`s^2=\frac{1}{N-\delta N}\sum_{i=1}^{N}(x_i-\bar{x})^2`,spoken:"计算每个元素与均值的偏差平方，再按校正后的自由度归一化。",explanation:"PyTorch 用 correction 控制自由度校正；默认 correction=1，即样本标准差/方差。",symbols:[{symbol:"\bar{x}",meaning:"归约切片的均值"},{symbol:"N",meaning:"元素数量"},{symbol:"\delta N",meaning:"correction 自由度校正值"},{symbol:"s,\,s^2",meaning:"标准差与方差"}]};
    const op=/max|amax|argmax/.test(n)?"\max":/min|amin|argmin/.test(n)?"\min":"\sum";return {latex:String.raw`y=${op}_{1\le i\le N}x_i`,spoken:op==="\\sum"?"把指定维度中的全部元素相加。":"在指定维度中寻找极值。",explanation:"argmax/argmin 返回极值位置索引；max/min 返回极值本身。",symbols:[{symbol:"x_i",meaning:"归约切片中的第 i 个元素"},{symbol:"N",meaning:"归约元素数量"},{symbol:"y",meaning:"归约后的值或索引"}]};
  }
  if (family === "binary") { const ops:Record<string,string>={add:"+",sub:"-",mul:"\\cdot",multiply:"\\cdot",div:"/",divide:"/",pow:"^{\,b}",remainder:"\\bmod"};const op=ops[n]??"+";const latex=n==="pow"?String.raw`Y_i=X_i^{\,b}`:String.raw`Y_i=X_i ${op} B_i`;return {latex,spoken:"按广播规则对两个输入逐元素计算。",explanation:"如果两个张量形状不同，PyTorch 会从最后一维开始按 broadcasting 规则对齐；标量会扩展到每个位置。",symbols:[{symbol:"X_i",meaning:"第一个输入在位置 i 的元素"},{symbol:"B_i\text{ 或 }b",meaning:"广播后的第二个输入元素或标量"},{symbol:"Y_i",meaning:"位置 i 的输出"}]}; }
  if (family === "unary") { const map:Record<string,string>={abs:String.raw`y=|x|`,absolute:String.raw`y=|x|`,neg:String.raw`y=-x`,negative:String.raw`y=-x`,exp:String.raw`y=e^x`,log:String.raw`y=\ln x`,log10:String.raw`y=\log_{10}x`,log2:String.raw`y=\log_2x`,sqrt:String.raw`y=\sqrt{x}`,square:String.raw`y=x^2`,reciprocal:String.raw`y=\frac1x`,sin:String.raw`y=\sin x`,cos:String.raw`y=\cos x`,tan:String.raw`y=\tan x`,floor:String.raw`y=\lfloor x\rfloor`,ceil:String.raw`y=\lceil x\rceil`};return {latex:map[n]??String.raw`Y_i=f(X_i)`,spoken:"对输入张量中的每个元素独立应用该数学函数。",explanation:"逐元素算子通常保持 shape 不变；对数、平方根、倒数等函数还要求输入位于有效定义域。",symbols:[...xySymbols,{symbol:"i",meaning:"张量中的元素位置"}]}; }
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
  if (family === "creation") return {latex:String.raw`Y_{i_1,\ldots,i_D}=c,\qquad \operatorname{shape}(Y)=(d_1,\ldots,d_D)`,spoken:"按照目标 shape 创建张量，并按接口规则写入元素。",explanation:"zeros、ones、full 分别令 c 为 0、1 或 fill_value；eye 只把主对角线置为 1。",symbols:[{symbol:"D",meaning:"张量维数"},{symbol:"d_1,\ldots,d_D",meaning:"各维长度"},{symbol:"c",meaning:"填充值"}]};
  if (family === "random" || family === "distribution") return {latex:String.raw`X_i\overset{\mathrm{i.i.d.}}{\sim}\mathcal{D}(\boldsymbol{\theta})`,spoken:"每个元素按照指定分布及其参数独立采样。",explanation:"设置随机种子可以复现实验；分布 D 和参数 θ 由当前随机接口决定。",symbols:[{symbol:"X_i",meaning:"第 i 个随机样本"},{symbol:"\mathcal D",meaning:"均匀、正态、伯努利等概率分布"},{symbol:"\boldsymbol\theta",meaning:"分布参数，如均值、标准差或概率"},{symbol:"\mathrm{i.i.d.}",meaning:"独立同分布"}]};
  return {latex:String.raw`Y=\mathcal{F}(X;\boldsymbol{\theta})`,spoken:"接口 F 使用参数 θ 处理输入 X 并产生输出 Y。",explanation:"该接口没有独立的通用标量公式；请以官方签名、输入约束和实验过程为准。",symbols:[{symbol:"\mathcal F",meaning:`当前接口 ${entry.name}`},{symbol:"X",meaning:"输入张量、对象或状态"},{symbol:"\boldsymbol\theta",meaning:"影响接口行为的参数集合"},{symbol:"Y",meaning:"返回值或更新后的状态"}]};
}

function MathExpression({ latex, spoken, inline = false }: { latex: string; spoken: string; inline?: boolean }) {
  const html = katex.renderToString(latex, { displayMode: !inline, throwOnError: false, strict: false, output: "html" });
  return <span className={inline ? "math-expression math-expression--inline" : "math-expression"} role="math" aria-label={spoken} dangerouslySetInnerHTML={{ __html: html }} />;
}

function FormulaPanel({ entry, compact = false }: { entry: ApiEntry; compact?: boolean }) {
  const formula = formulaSpecOf(entry);
  return <div className={`formula-panel ${compact ? "formula-panel--compact" : ""}`}>
    <MathExpression latex={formula.latex} spoken={formula.spoken} />
    <p className="formula-reading"><b>怎么读：</b>{formula.spoken}</p>
    {!compact && <><p className="formula-note">{formula.explanation}</p><div className="formula-symbols"><span>公式变量</span><dl>{formula.symbols.map((item) => <div key={item.symbol}><dt><MathExpression latex={item.symbol} spoken={item.symbol} inline /></dt><dd>{item.meaning}</dd></div>)}</dl></div><a className="formula-source" href={entry.url} target="_blank" rel="noreferrer">依据 PyTorch 官方定义 · 查看原文 ↗</a></>}
  </div>;
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
      <div className="operator-node"><span>{entry.leaf}</span><MathExpression latex={formulaSpecOf(entry).latex} spoken={formulaSpecOf(entry).spoken} /><i>→</i></div>
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
      <aside className="api-inspector"><div className="api-inspector__top"><span>{selected.group}</span><i>{selected.typeLabel}</i></div><h3>{selected.leaf}</h3><code className="api-inspector__path">{selected.name}</code><section><small>它做什么</small><p>{conceptOf(selected)}</p></section><section><small>官方数学定义</small><FormulaPanel entry={selected} compact /></section><section><small>应用场景</small><p>{scenarioOf(selected)}</p></section><a className="official-link" href="#deep-lab">进入本接口完整实验 ↓</a></aside>
    </div>

    <article className="deep-lab" id="deep-lab">
      <header><div><p className="eyebrow">FULL API EXPERIMENT</p><h3>{selected.name}</h3></div><span>{sim.mode}</span></header>
      <div className="deep-lab__grid">
        <section className="deep-card"><small>① 中文解析与官方公式</small><h4>{selected.summary}</h4><p>{conceptOf(selected)}</p><div className="deep-formula"><span>标准数学定义</span><FormulaPanel entry={selected} /></div></section>
        <section className="deep-card"><small>② 官方调用方法</small>{docLoading?<p className="loading-line">正在读取官方签名…</p>:<><pre><code>{remote?.signature||`${selected.name}(*args, **kwargs)`}</code></pre>{remote?.summary&&<p className="official-summary">官方说明：{remote.summary}</p>}</>}<a href={selected.url} target="_blank" rel="noreferrer">核对官方原文 ↗</a></section>
        <section className="deep-card deep-card--wide"><small>③ 参数与变量地图</small><div className="deep-vars">{variablesOf(selected,remote).map((v)=><div key={v.name}><code>{v.name}</code><p>{v.meaning}</p><span>例：{v.sample}</span></div>)}</div></section>
        <section className="deep-card"><small>④ 使用场景与 Example</small><p>{scenarioOf(selected)}</p><pre><code>{selected.type==="class"?`component = ${selected.name}(...)\noutput = component(input)`:selected.name.startsWith("torch.Tensor.")?`output = input.${selected.leaf}(...)`:`output = ${selected.name}(input, ...)`}</code></pre></section>
        <section className="deep-card simulator-card"><small>⑤ 输入与执行</small><label><span>实验输入（JSON）</span><textarea value={spec} onChange={(e)=>{setSpec(e.target.value);setRunState("idle");}} spellCheck={false}/></label><button type="button" onClick={run} disabled={runState==="running"}>{runState==="running"?"⏳ 正在执行…":runState==="success"?"✓ 已执行 · 再运行一次":"▶ 运行本接口实验"}</button>{runState==="idle"&&<p className="run-hint">修改输入后点击按钮，页面会自动定位到本次输出。</p>}{simError&&<p className="sim-error" role="alert">{simError}</p>}</section>
        <section ref={outputRef} className={`deep-card deep-card--wide output-card output-card--${runState}`} aria-live="polite"><small>⑥ 计算过程与最终结果</small><div className="run-receipt"><b>{runState==="running"?"正在计算…":runState==="success"?`运行成功 · 第 ${runCount} 次`:runState==="error"?"运行失败":"示例结果预览"}</b><span>{lastRunAt?`完成时间 ${lastRunAt}`:"点击上方按钮执行当前输入"}</span></div><div className="sim-mode"><b>{sim.title}</b><span>{sim.mode} · {sim.mode==="数值计算"||sim.mode==="梯度计算"?"下方展示实际算式、参与运算的单元格与数值":"下方展示张量形状、元素位置与状态变化"}</span></div><CalculationVisualizer key={`${selected.name}-${runCount}`} entry={selected} source={spec} sim={sim} /><div className="trace-row trace-row--compact">{sim.trace.map((step,i)=><div key={`${i}-${step}`}><span>{i+1}</span><p>{step}</p></div>)}</div><div className="result-label">最终结果（精确数据）</div><pre><code>{JSON.stringify(sim.value,null,2)}</code></pre></section>
      </div>
    </article>
  </section>;
}
