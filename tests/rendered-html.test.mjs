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
  assert.ok(JSON.parse(index).length > 9000);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(packageJson, /"name": "pytorch-visual-lab"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});
