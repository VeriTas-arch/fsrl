"""Deterministic formatting helpers for generated Markdown navigation."""

from __future__ import annotations

import re
import textwrap

_MARKDOWN_ATOM = re.compile(r"`[^`\n]+`|\[[^]\n]+\]\((?:<[^>\n]+>|[^)\n]+)\)")
_SPACE_SENTINEL = "\0"


def wrap_markdown(
    text: str,
    *,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> list[str]:
    """Wrap prose without splitting inline code or Markdown links."""

    protected = _MARKDOWN_ATOM.sub(
        lambda match: match.group(0).replace(" ", _SPACE_SENTINEL),
        text,
    )
    wrapped = textwrap.wrap(
        protected,
        width=88,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [line.replace(_SPACE_SENTINEL, " ") for line in wrapped]
