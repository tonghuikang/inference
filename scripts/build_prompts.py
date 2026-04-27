"""Generate the line-jumping prompts used by the observer demo buttons.

For each N in {10, 100, 1000, 10000} we write `prompts/{N}.txt` containing:
    Line 1: if you are here, go to line k_1.
    Line 2: if you are here, go to line k_2.
    ...
    Line N: if you are here, go to line k_N.

    Starting at line 1, follow the instructions step by step for J jumps.
    For each jump, write the line number you are on, then the rule on that
    line, then the line you go to next. Format: ...

The k_i are derived from a deterministic LCG seeded with N + 42 so the file
content is reproducible. The same numbers don't need to match what JS
produced — these files are now the source of truth, and app.js fetches them
on demand.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "prompts"
SIZES = [10, 100, 1000]


def build_lines(n: int, seed: int) -> tuple[str, list[int]]:
    """Build a deterministic line-jumping graph with NO cycles in the path
    starting from line 1, where the path covers exactly n/2 lines.

    Construction:
      1. Sample a random permutation of [1..n].
      2. Slice the first n/2 positions to be the path; force position 0 to be 1.
      3. For each path[k] for k in 0..n/2-2, set rules[path[k]] = path[k+1].
      4. The last path entry path[n/2-1] gets a tail target — point it back
         at line 1 to make it obvious there's no further useful path; the
         model won't follow past the requested jump count anyway.
      5. All non-path lines get random targets.

    Returns (text, path) so callers can sanity-check the expected answer.
    """
    import random

    rng = random.Random(seed)
    half = n // 2

    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    # Make sure 1 is at position 0 of the path.
    one_idx = perm.index(1)
    perm[0], perm[one_idx] = perm[one_idx], perm[0]
    path = perm[:half]

    rules: dict[int, int] = {}
    for k in range(half - 1):
        rules[path[k]] = path[k + 1]
    # Tail of the path points back at 1 — it's outside the requested jumps.
    rules[path[half - 1]] = 1

    # Fill in the rest of the lines with random targets, never pointing at self.
    for i in range(1, n + 1):
        if i in rules:
            continue
        t = rng.randint(1, n)
        if t == i:
            t = (t % n) + 1
        rules[i] = t

    text = "\n".join(f"{i}: {rules[i]}" for i in range(1, n + 1))
    return text, path


def task_for(jumps: int) -> str:
    # The model should traverse the graph and recite each line number it
    # visits, comma-separated. With path length n/2, that means n/2 numbers
    # starting at 1. Demonstrates that the model is actually reading the
    # rules from context (vs guessing or memorising).
    return (
        f"\n\nStart at line 1 and follow the jumps. Output the line numbers "
        f"you visit, comma-separated, until you have visited {jumps + 1} lines. "
        "No commentary, just the numbers."
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        # The path length is n/2 (per the spec). Number of jumps the model
        # performs is path_len - 1 = n/2 - 1.
        jumps = n // 2 - 1
        text, path = build_lines(n, seed=n + 42)
        body = text + task_for(jumps)
        out_path = OUT / f"{n}.txt"
        out_path.write_text(body)
        # Drop the expected path next to the prompt so we can grade outputs.
        (OUT / f"{n}.path.txt").write_text(",".join(str(p) for p in path) + "\n")
        print(f"wrote {out_path} ({len(body):,} chars, path_len={len(path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
