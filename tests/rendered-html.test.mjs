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
  assert.match(browser, /结构预演/);
  assert.match(browser, /运行成功/);
  assert.match(browser, /aria-live="polite"/);
  assert.match(browser, /链式法则得到梯度/);
  assert.match(browser, /最终结果/);
  assert.doesNotMatch(browser, /output_contract|接口生命周期|返回值以官方签名为准/);
  assert.ok(JSON.parse(index).length > 9000);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(packageJson, /"name": "pytorch-visual-lab"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});
