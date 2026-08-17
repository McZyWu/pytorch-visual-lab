const prerequisites = [
  {
    title: "Python 列表与函数",
    detail: "看得懂列表、变量和 def 即可；不会类也能先开始。",
  },
  {
    title: "基础代数",
    detail: "知道加减乘除、平均数，以及 y = wx + b 表示什么。",
  },
  {
    title: "无需先懂深度学习",
    detail: "神经网络、梯度和优化器会在实验中边做边理解。",
  },
] as const;

const learningPath = [
  {
    name: "Tensor",
    detail: "先把数字装进 PyTorch 的数据容器，学会创建和查看。",
  },
  {
    name: "shape / dim",
    detail: "看懂数据有几层、每层多大，以及沿哪个方向操作。",
  },
  {
    name: "计算",
    detail: "练习加减乘除、矩阵乘法、广播和常见聚合。",
  },
  {
    name: "网络",
    detail: "用 nn.Module 和层把计算步骤组合成一个模型。",
  },
  {
    name: "loss / backward / optimizer",
    detail: "衡量误差、算出调整方向，再让参数自动更新。",
  },
] as const;

const trainingLoop = `import torch
from torch import nn

# 训练数据：每行是一个样本，每个样本只有一个特征
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [5.0], [7.0]])

# 模型：尝试学会 y = wx + b
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()                         # loss：预测离答案有多远
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100):
    prediction = model(x)                     # 1. 让模型先猜答案
    loss = loss_fn(prediction, y)             # 2. 比较猜测和正确答案
    optimizer.zero_grad()                     # 3. 清掉上一轮留下的梯度
    loss.backward()                           # 4. 反向计算每个参数该怎么调
    optimizer.step()                          # 5. 按梯度更新模型参数

print(model(torch.tensor([[4.0]])))           # 应逐渐接近 9`;

const glossary = [
  ["Tensor", "装数字的多维容器，是 PyTorch 计算的基本单位。"],
  ["shape", "每个方向有多长；例如 [32, 3, 28, 28]。"],
  ["dim", "维度或轴；dim=0 就是在第 0 个方向上操作。"],
  ["batch", "一次交给模型处理的一小批样本。"],
  ["channel", "同一位置的不同信息层，例如彩色图像的 RGB 三层。"],
  ["logits", "模型还没变成概率的原始分类分数。"],
  ["gradient", "告诉参数往哪个方向、调整多少能让 loss 变小。"],
  ["broadcasting", "形状不同但兼容时，PyTorch 自动扩展较小张量来计算。"],
] as const;

export default function BeginnerStart() {
  return (
    <section
      id="beginner-start"
      className="beginner-start"
      aria-labelledby="beginner-start-title"
    >
      <header className="beginner-start__header">
        <p className="beginner-start__eyebrow">给只懂一点的你</p>
        <h2 id="beginner-start-title" className="beginner-start__title">
          先学什么：一条短而完整的 PyTorch 路线
        </h2>
        <p className="beginner-start__lead">
          <strong>先完成 15 个重点实验，不必先看 9,066 个 API。</strong>
          先建立完整的训练直觉，遇到具体问题时再查接口。
        </p>
      </header>

      <section
        className="beginner-start__prerequisites"
        aria-labelledby="beginner-prerequisites-title"
      >
        <h3 id="beginner-prerequisites-title">课前只需要这些</h3>
        <ul className="beginner-start__prerequisite-list">
          {prerequisites.map((item) => (
            <li className="beginner-start__prerequisite" key={item.title}>
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="beginner-start__path" aria-labelledby="beginner-path-title">
        <h3 id="beginner-path-title">按这 5 步学</h3>
        <ol className="beginner-start__path-list">
          {learningPath.map((item, index) => (
            <li className="beginner-start__path-step" key={item.name}>
              <span className="beginner-start__path-number" aria-hidden="true">
                {index + 1}
              </span>
              <div>
                <strong>{item.name}</strong>
                <p>{item.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <article className="training-loop" aria-labelledby="training-loop-title">
        <header className="training-loop__header">
          <div>
            <p className="training-loop__eyebrow">最小完整示例</p>
            <h3 id="training-loop-title">先读懂这一段训练循环</h3>
          </div>
          <p>从上往下读一遍，再按注释手动复述五个训练动作。</p>
        </header>
        <pre className="training-loop__code">
          <code>{trainingLoop}</code>
        </pre>
      </article>

      <section className="glossary" aria-labelledby="glossary-title">
        <header className="glossary__header">
          <p className="glossary__eyebrow">遇到生词先看这里</p>
          <h3 id="glossary-title">8 个核心术语的白话解释</h3>
        </header>
        <dl className="glossary__grid glossary__list">
          {glossary.map(([term, meaning]) => (
            <div className="glossary__item" key={term}>
              <dt>{term}</dt>
              <dd>{meaning}</dd>
            </div>
          ))}
        </dl>
      </section>

      <aside className="beginner-start__notice" role="note" aria-label="运行说明">
        <strong>注意：网页实验是教学模拟。</strong>
        <p>
          它用于帮助你观察形状、步骤和结果；真实训练应在 Python / PyTorch
          运行时中执行，并以实际输出为准。
        </p>
      </aside>
    </section>
  );
}
