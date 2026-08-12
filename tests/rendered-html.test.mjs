import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("includes the Chinese PyTorch learning experience", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /TorchScope/);
  assert.match(page, /张量实验台/);
  assert.match(page, /torch\.matmul/);
  assert.match(page, /nn\.CrossEntropyLoss/);
  assert.match(page, /核心公式/);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(packageJson, /"name": "pytorch-visual-lab"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});
