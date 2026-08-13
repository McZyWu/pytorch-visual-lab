import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("includes the full Chinese PyTorch learning experience", async () => {
  const [page, browser, index, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/full-api-browser.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api-index.generated.json", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /TorchScope/);
  assert.match(page, /张量实验台/);
  assert.match(page, /torch\.matmul/);
  assert.match(page, /nn\.CrossEntropyLoss/);
  assert.match(page, /核心公式/);
  assert.match(browser, /FULL API EXPERIMENT/);
  assert.match(browser, /运行本接口实验/);
  assert.match(browser, /具体步骤、返回值和状态变化/);
  assert.doesNotMatch(browser, /结构预演/);
  assert.match(browser, /运行成功/);
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
  assert.match(browser, /最终结果/);
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
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});
