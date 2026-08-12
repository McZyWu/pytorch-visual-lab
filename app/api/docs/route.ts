const DOCS_ORIGIN = "https://docs.pytorch.org";

function decodeHtml(value: string) {
  return value.replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;|&#x27;/g, "'").replace(/\s+/g, " ").trim();
}
function escapeRegExp(value: string) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

export async function GET(request: Request) {
  const incoming = new URL(request.url); const name = incoming.searchParams.get("name") ?? ""; const rawUrl = incoming.searchParams.get("url") ?? "";
  try {
    const target = new URL(rawUrl);
    if (target.origin !== DOCS_ORIGIN || !target.pathname.startsWith("/docs/2.13/") || !name.startsWith("torch")) return Response.json({ error: "无效的官方文档地址" }, { status: 400 });
    const response = await fetch(target.toString(), { headers: { "user-agent": "TorchScope/1.0 PyTorch learning tool" }, cf: { cacheTtl: 86400 } } as RequestInit);
    if (!response.ok) throw new Error(`docs ${response.status}`);
    const html = await response.text(); const escaped = escapeRegExp(name);
    const block = html.match(new RegExp(`<dt[^>]*id=["']${escaped}["'][^>]*>([\\s\\S]*?)<\\/dt>([\\s\\S]*?)<\\/dd>`, "i"));
    const signature = block ? decodeHtml(block[1]).replace(/\s*¶\s*$/, "") : name;
    const params = block ? [...block[1].matchAll(/<em[^>]*class=["'][^"']*sig-param[^"']*["'][^>]*>([\s\S]*?)<\/em>/gi)].map((match) => decodeHtml(match[1])) : [];
    const summaryMatch = block?.[2]?.match(/<p>([\s\S]*?)<\/p>/i); const summary = summaryMatch ? decodeHtml(summaryMatch[1]) : "";
    return Response.json({ signature, parameters: params, summary, source: target.toString() }, { headers: { "cache-control": "public, max-age=86400" } });
  } catch { return Response.json({ error: "官方详情暂时无法读取，可使用页面内官方链接查看。", signature: `${name}(*args, **kwargs)`, parameters: [] }); }
}
