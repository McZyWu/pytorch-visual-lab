import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("includes the full Chinese PyTorch learning experience", async () => {
  const [page, beginner, browser, index, layout, packageJson, readme] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/beginner-start.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/full-api-browser.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api-index.generated.json", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../README.md", import.meta.url), "utf8"),
  ]);

  assert.match(page, /TorchScope/);
  assert.match(page, /张量实验台/);
  assert.match(page, /torch\.matmul/);
  assert.match(page, /nn\.CrossEntropyLoss/);
  assert.match(page, /核心公式/);
  assert.match(page, /useState\("tensor"\)/);
  assert.match(page, /torchscope:completed-lessons/);
  assert.match(page, /上一课/);
  assert.match(page, /下一课/);
  assert.match(page, /教学模拟，不是真实 PyTorch 运行时/);
  assert.match(beginner, /先完成 15 个重点实验/);
  assert.match(beginner, /optimizer\.zero_grad/);
  assert.match(beginner, /loss\.backward/);
  assert.match(beginner, /optimizer\.step/);
  assert.match(beginner, /Tensor/);
  assert.match(beginner, /broadcasting/);
  assert.match(beginner, /真实训练应在 Python \/ PyTorch/);
  assert.match(browser, /API QUICK READ &amp; TEACHING LAB/);
  assert.match(browser, /按需查询与实验/);
  assert.match(browser, /event\.key === "\/"/);
  assert.match(browser, /searchRef\.current\?\.focus/);
  assert.match(browser, /运行教学模拟并打开输出/);
  assert.match(browser, /页面模拟器的 JSON（不是 Python 函数签名）/);
  assert.match(browser, /处理步骤和状态示意/);
  assert.doesNotMatch(browser, /结构预演/);
  assert.match(browser, /模拟完成/);
  assert.match(browser, /aria-live="polite"/);
  assert.match(browser, /链式法则得到梯度/);
  assert.match(browser, /张量计算过程可视化/);
  assert.match(browser, /tensor-visual--cuboid/);
  assert.match(browser, /切换查看层/);
  assert.match(browser, /完整展开/);
  assert.match(browser, /PyTorch 官方定义/);
  assert.match(browser, /公式变量/);
  assert.match(browser, /katex\.renderToString/);
  assert.match(browser, /位置计算小窗口/);
  assert.match(browser, /inputCoord/);
  assert.match(browser, /Conv1d 输出位置/);
  assert.match(browser, /自动播放/);
  assert.match(browser, /输出 Y\[h=\$\{i\},w=\$\{j\}\]/);
  assert.match(browser, /网页按简化规则算出的结果/);
  assert.match(browser, /规则 \/ 流程示意结果/);
  assert.match(browser, /30 秒读懂这个接口/);
  assert.match(browser, /最容易踩的坑/);
  assert.match(browser, /我能预测输出类型和 shape 吗/);
  assert.match(browser, /可复制到 Python 运行/);
  assert.match(browser, /调用骨架，不能直接运行/);
  assert.match(browser, /必填/);
  assert.match(browser, /返回说明/);
  assert.match(browser, /empty 不是 zeros/);
  assert.match(browser, /detach 不等于复制/);
  assert.match(browser, /simulated_available/);
  assert.match(browser, /searchRank\(a,keyword\)/);
  assert.match(browser, /url\.searchParams\.set\("api",entry\.name\)/);
  assert.doesNotMatch(browser, /n\.includes\("zeros"\)\|\|n\.includes\("empty"\)/);
  assert.match(browser, /全部细分类/);
  assert.match(browser, /按接口种类浏览/);
  assert.match(browser, /一级模块/);
  assert.match(browser, /二级分类/);
  assert.match(browser, /相似方法对比索引/);
  assert.match(browser, /区别表 \/ 模拟表格在哪/);
  assert.match(browser, /接口详情标签页/);
  assert.match(browser, /介绍与公式/);
  assert.match(browser, /计算过程与输出/);
  assert.match(browser, /它具体做什么/);
  assert.match(browser, /它做什么、怎么算、返回什么/);
  assert.match(browser, /foreachOperation/);
  assert.match(browser, /向正无穷取整/);
  assert.match(browser, /列表长度与输入相同/);
  assert.match(browser, /张量之间不广播/);
  assert.doesNotMatch(browser, /实验时先确认输入类型和形状/);
  assert.doesNotMatch(browser, /Y=f\(X\)/);
  assert.doesNotMatch(browser, /Y_i=f\(X_i\)/);
  assert.match(browser, /公式里的 x 就是当前元素 X\^\(k\)_i/);
  assert.match(browser, /原地版本以下划线结尾，返回 None/);
  assert.match(browser, /不会互相广播/);
  assert.match(browser, /正好在中点时取偶数/);
  assert.match(browser, /同类算法区别/);
  assert.match(browser, /Conv1d、Conv2d、Conv3d/);
  assert.match(browser, /subcategoryOf/);
  assert.match(browser, /交互式滑窗演示/);
  assert.match(browser, /实际扫描顺序/);
  assert.match(browser, /核心区别是卷积核拥有 1 \/ 2 \/ 3 个空间维度/);
  assert.match(browser, /常用输出遍历顺序/);
  assert.match(browser, /W → H → D/);
  assert.match(browser, /不会影响最终输出值/);
  assert.match(browser, /输入 shape 是否有限制/);
  assert.match(browser, /S_in \+ 2p/);
  assert.match(browser, /padding=&quot;same&quot;/);
  assert.doesNotMatch(browser, /output_contract|接口生命周期|返回值以官方签名为准/);
  assert.ok(JSON.parse(index).length > 9000);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(packageJson, /"name": "pytorch-visual-lab"/);
  assert.match(readme, /## 适合谁/);
  assert.match(readme, /## 教学模拟边界/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});

test("keeps exactly 100 curated function guides complete and connected to the API index", async () => {
  const [source, indexSource, browser] = await Promise.all([
    readFile(new URL("../app/function-guides.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api-index.generated.json", import.meta.url), "utf8"),
    readFile(new URL("../app/full-api-browser.tsx", import.meta.url), "utf8"),
  ]);
  const indexNames = new Set(JSON.parse(indexSource).map((entry) => entry.name));
  const guideLines = source.split(/\r?\n/).filter((line) => /^\s*guide\("torch\./.test(line));
  const guideNames = guideLines.map((line) => line.match(/guide\("([^"]+)"/)?.[1]);

  assert.equal(guideNames.length, 100, "the curated audit must cover exactly 100 functions");
  assert.equal(new Set(guideNames).size, 100, "every curated function must be unique");
  assert.deepEqual(guideNames.filter((name) => !indexNames.has(name)), [], "every curated function must exist in the generated API index");
  for (const line of guideLines) {
    assert.ok(line.length > 180, `guide is too vague: ${line.slice(0, 60)}`);
    assert.ok((line.match(/"/g) ?? []).length >= 20, `guide is missing a readability field: ${line.slice(0, 60)}`);
  }

  assert.match(browser, /CURATED_FUNCTION_GUIDE_COUNT/);
  assert.match(browser, /curatedFunctionGuideOf\(selected\.name\)/);
  assert.match(browser, /适合什么时候用/);
  assert.match(browser, /什么时候先别用/);
  assert.match(browser, /容易混淆的函数/);
  assert.match(browser, /人工精读/);
  assert.match(browser, /搜索全部 PyTorch 接口/);
  assert.match(browser, /全库搜索 · 不受类型与模块筛选限制/);
  assert.match(browser, /只看 \{CURATED_FUNCTION_GUIDE_COUNT\} 个人工精读/);
  assert.match(browser, /keyword\?searchable\.includes\(keyword\):browsingScope/);
  assert.match(browser, /choose\(related,true,"overview"\)/);
  assert.match(browser, /点一下直接对比/);
  assert.match(source, /function defaultAutogradOf/);
  assert.match(source, /新的叶子 Tensor，默认 requires_grad=False/);
  assert.match(source, /该离散取整操作的反向梯度为 0/);
  assert.match(source, /转成整数或 bool 会使结果不再需要梯度/);
  assert.match(source, /function defaultSideEffectOf/);
  assert.match(source, /新的独立 Tensor 存储/);
  assert.match(source, /结果通常与输入共享底层存储/);
  assert.match(source, /不能依赖固定的别名关系/);
});
