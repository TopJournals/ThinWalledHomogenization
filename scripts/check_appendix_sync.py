"""Check that the manuscript appendix exactly matches the archived core solver."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path


def appendix_lines(manuscript: Path) -> list[str]:
    text = manuscript.read_text(encoding="utf-8")
    label = r"\label{sec:appendix_code}"
    start = text.index(label)
    begin = text.index(r"\begin{lstlisting}", start) + len(r"\begin{lstlisting}")
    end = text.index(r"\end{lstlisting}", begin)
    return text[begin:end].strip("\r\n").splitlines()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=root.parent / "R2603_paper" / "11_Manuscript.tex",
    )
    args = parser.parse_args()

    expected = (root / "core" / "plate_homogenizer.py").read_text(encoding="utf-8").splitlines()
    actual = appendix_lines(args.manuscript)
    if actual != expected:
        diff = difflib.unified_diff(
            expected,
            actual,
            fromfile="core/plate_homogenizer.py",
            tofile=str(args.manuscript),
            lineterm="",
        )
        raise SystemExit("Appendix mismatch:\n" + "\n".join(diff))

    digest = hashlib.sha256("\n".join(expected).encode("utf-8")).hexdigest()
    print(f"Exact appendix match: {len(expected)} lines; SHA-256 {digest}")


if __name__ == "__main__":
    main()
