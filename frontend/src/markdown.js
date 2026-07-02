// 轻量 markdown 渲染：把模型产出的 #/**/列表 等转成 HTML。
// 故意不引第三方库——只覆盖知识档案正文用到的子集（标题/加粗/斜体/列表/行内代码/分隔线），
// 安全第一：先整体 HTML 转义（防 v-html XSS），再在转义后的文本上做受控替换。
// 模型正文是内部生成、非用户输入，但仍按不可信处理。

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// 行内：加粗 / 斜体 / 行内代码。在已转义文本上操作，标记符本身是安全字面量。
function renderInline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // 斜体：单 * 或 _，避免吃掉已处理的 **（上面已转成 <strong>）
  out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  return out;
}

/**
 * 把 markdown 文本渲染成安全 HTML 字符串，供 v-html 使用。
 * 支持：# ~ ###### 标题、- / * / 1. 列表、--- 分隔线、空行分段、行内加粗/斜体/代码。
 */
export function renderMarkdown(src) {
  if (!src) return "";
  const lines = String(src).split("\n");
  const html = [];
  let listType = null; // 'ul' | 'ol' | null

  const closeList = () => {
    if (listType) {
      html.push(listType === "ul" ? "</ul>" : "</ol>");
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) {
      closeList();
      continue;
    }
    // 分隔线
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      closeList();
      html.push("<hr>");
      continue;
    }
    // 标题
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      closeList();
      const level = h[1].length;
      html.push(`<h${level}>${renderInline(h[2])}</h${level}>`);
      continue;
    }
    // 有序列表
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ol) {
      if (listType !== "ol") { closeList(); html.push("<ol>"); listType = "ol"; }
      html.push(`<li>${renderInline(ol[1])}</li>`);
      continue;
    }
    // 无序列表
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    if (ul) {
      if (listType !== "ul") { closeList(); html.push("<ul>"); listType = "ul"; }
      html.push(`<li>${renderInline(ul[1])}</li>`);
      continue;
    }
    // 普通段落
    closeList();
    html.push(`<p>${renderInline(line)}</p>`);
  }
  closeList();
  return html.join("");
}
