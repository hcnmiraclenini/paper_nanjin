#!/usr/bin/env python3
"""Deprecated legacy generator for the old §2.5-§3 Word draft.

The current manuscript is maintained in:

- ``paper/论文终稿.md``
- ``paper/2026-0268-基于多专家融合的铁路客流多尺度预测方法.docx``

This path is kept only so older notes that reference the tool fail with a clear
message instead of regenerating superseded tables, gate statistics, or Traffic
claims.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Deprecated: do not use this legacy draft generator. Use the final "
        "manuscript files under paper/ instead."
    )


if __name__ == "__main__":
    main()
