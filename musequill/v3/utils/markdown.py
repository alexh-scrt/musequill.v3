from typing import Any, Iterable, Mapping, Optional
import re
import textwrap

def dict_to_markdown(
    data: Any,
    title: Optional[str] = None,
    *,
    heading_level: int = 1,
    max_heading: int = 6,
    sort_keys: bool = False,
    wrap_width: Optional[int] = None,
    prefer_titles: Iterable[str] = ("title", "name", "id", "chapter", "act"),
    bullet: str = "-",
) -> str:
    """
    Convert arbitrary nested Python data (dict/list/scalars) to Markdown.

    Design goals for LLMs:
    - Deterministic structure (preserve insertion order by default).
    - Headings for dict keys; bullets for scalar lists.
    - Lists of dicts become repeated sub-sections with inferred titles when possible.
    - Long scalar strings are fenced for clarity.

    Args:
        data: Any Python object (dict/list/scalars).
        title: Optional top-level title (H1).
        heading_level: Starting heading level (1..6).
        max_heading: Largest heading level to use. Beyond this, use bold labels.
        sort_keys: If True, sort dict keys (otherwise preserve insertion order).
        wrap_width: If set, soft-wrap long scalar lines at this width.
        prefer_titles: Keys to use as a section title when rendering list items.
        bullet: Bullet character for scalar lists.
    """
    def esc(s: str) -> str:
        # Minimal Markdown escaping for headings/labels
        return re.sub(r"([*_`#>|])", r"\\\1", s)

    def key_to_label(k: Any) -> str:
        s = str(k)
        s = s.replace("_", " ").strip()
        # Title-case but keep ALLCAPS blocks like IDs intact
        if not s.isupper():
            s = s[:1].upper() + s[1:]
        return esc(s)

    def is_short(s: str) -> bool:
        return len(s) <= 80 and "\n" not in s

    def fence_block(s: str) -> str:
        return f"```\n{s}\n```"

    def maybe_wrap(s: str) -> str:
        if wrap_width and "\n" not in s and len(s) > wrap_width:
            return "\n".join(textwrap.wrap(s, wrap_width))
        return s

    def guess_item_title(d: Mapping[str, Any], idx: int) -> str:
        # Choose first available preferred key; fallback to index label
        for k in prefer_titles:
            if k in d and isinstance(d[k], (str, int, float)):
                return f"{key_to_label(k)}: {d[k]}"
        return f"Item {idx}"

    def render_scalar(val: Any) -> str:
        if val is None:
            return "`null`"
        if isinstance(val, bool):
            return "`true`" if val else "`false`"
        if isinstance(val, (int, float)):
            return f"`{val}`"
        s = str(val)
        s = maybe_wrap(s)
        return f"`{s}`" if is_short(s) else fence_block(s)

    def render_list(lst: list, level: int) -> str:
        if not lst:
            return f"{bullet} (empty list)"

        # If list of dicts -> sub-sections
        if all(isinstance(x, Mapping) for x in lst):
            parts = []
            for i, item in enumerate(lst, 1):
                title_line = guess_item_title(item, i)
                if level <= max_heading:
                    parts.append(f"\n{'#' * level} {esc(title_line)}")
                else:
                    parts.append(f"\n**{esc(title_line)}**")
                parts.append(render(item, level + 1))
            return "\n".join(parts).lstrip()

        # If list of scalars or mixed -> bullets
        lines = []
        for x in lst:
            if isinstance(x, (Mapping, list)):
                # Nested structure as a sub-block under a bullet
                lines.append(f"{bullet} (nested)")
                block = render(x, level + 1)
                # Indent nested block
                block = textwrap.indent(block, "  ")
                lines.append(block)
            else:
                lines.append(f"{bullet} {render_scalar(x)}")
        return "\n".join(lines)

    def render_dict(d: Mapping[str, Any], level: int) -> str:
        keys = list(d.keys())
        if sort_keys:
            # Sort with numbers first, then lexicographically by str
            keys.sort(key=lambda k: (not isinstance(k, (int, float)), str(k)))
        parts = []
        for k in keys:
            v = d[k]
            label = key_to_label(k)
            if isinstance(v, Mapping):
                if level <= max_heading:
                    parts.append(f"\n{'#' * level} {label}")
                else:
                    parts.append(f"\n**{label}**")
                parts.append(render(v, level + 1))
            elif isinstance(v, list):
                if level <= max_heading:
                    parts.append(f"\n{'#' * level} {label}")
                else:
                    parts.append(f"\n**{label}**")
                parts.append(render_list(v, level + 1))
            else:
                # leaf value on a single line for compact LLM parsing
                val = render_scalar(v)
                parts.append(f"- **{label}:** {val}")
        return "\n".join(parts).lstrip()

    def render(obj: Any, level: int) -> str:
        if isinstance(obj, Mapping):
            return render_dict(obj, level)
        if isinstance(obj, list):
            return render_list(obj, level)
        return render_scalar(obj)

    md_parts = []
    if title:
        hl = min(max(1, heading_level), max_heading)
        md_parts.append(f"{'#' * hl} {esc(title)}")
        md_parts.append(render(data, heading_level + 1))
    else:
        md_parts.append(render(data, heading_level))
    return "\n".join(md_parts).strip()
