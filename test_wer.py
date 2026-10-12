#!/usr/bin/env python3
"""Isolated WER check using the same protocol as train_modal / loadtest_modal.

Protocol
--------
Shared:
  1) Strip optional ``language …<asr_text>`` prefix
  2) Lowercase + collapse whitespace

Two scores:
  - ``wer``          : punctuation kept
  - ``wer_no_punct`` : punctuation removed (``string.punctuation``)

Corpus WER via ``jiwer.process_words`` over non-empty refs (punct-kept emptiness).

Examples
--------
  python test_wer.py \\
    --ref 'language English<asr_text>Hello world, you are the best' \\
    --hyp 'hello world you are the best'

  python test_wer.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def normalize_for_wer(text: str, *, strip_punct: bool = False) -> str:
    """Strip Qwen language prefix; lowercase; optionally drop punctuation."""
    import string

    text = text or ""
    m = re.search(r"<asr_text>(.*)$", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    text = text.lower()
    if strip_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus WER with punctuation kept (`wer`) and stripped (`wer_no_punct`)."""
    from jiwer import process_words

    if len(references) != len(hypotheses):
        raise ValueError(
            f"ref/hyp length mismatch: {len(references)} vs {len(hypotheses)}"
        )

    kept_refs, kept_hyps = [], []
    kept_refs_np, kept_hyps_np = [], []
    per_utt = []
    for i, (ref_raw, hyp_raw) in enumerate(zip(references, hypotheses)):
        ref_n = normalize_for_wer(ref_raw, strip_punct=False)
        hyp_n = normalize_for_wer(hyp_raw, strip_punct=False)
        ref_np = normalize_for_wer(ref_raw, strip_punct=True)
        hyp_np = normalize_for_wer(hyp_raw, strip_punct=True)
        skipped = not bool(ref_n.strip())
        row = {
            "index": i,
            "ref_raw": ref_raw,
            "hyp_raw": hyp_raw,
            "ref": ref_n,
            "hyp": hyp_n,
            "ref_no_punct": ref_np,
            "hyp_no_punct": hyp_np,
            "skipped_empty_ref": skipped,
            "wer": None,
            "wer_no_punct": None,
        }
        if not skipped:
            kept_refs.append(ref_n)
            kept_hyps.append(hyp_n)
            kept_refs_np.append(ref_np)
            kept_hyps_np.append(hyp_np)
            row["wer"] = float(process_words(ref_n, hyp_n).wer)
            row["wer_no_punct"] = float(process_words(ref_np, hyp_np).wer)
        per_utt.append(row)

    def _corpus(refs: list[str], hyps: list[str]) -> dict:
        if not refs:
            return {
                "wer": 1.0,
                "n": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
            }
        out = process_words(refs, hyps)
        return {
            "wer": float(out.wer),
            "n": len(refs),
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
        }

    with_p = _corpus(kept_refs, kept_hyps)
    no_p = _corpus(kept_refs_np, kept_hyps_np)
    return {
        "wer": with_p["wer"],
        "wer_no_punct": no_p["wer"],
        "n": with_p["n"],
        "substitutions": with_p["substitutions"],
        "deletions": with_p["deletions"],
        "insertions": with_p["insertions"],
        "hits": with_p["hits"],
        "substitutions_no_punct": no_p["substitutions"],
        "deletions_no_punct": no_p["deletions"],
        "insertions_no_punct": no_p["insertions"],
        "hits_no_punct": no_p["hits"],
        "per_utt": per_utt,
    }


def _load_jsonl_pairs(path: str) -> tuple[list[str], list[str]]:
    refs, hyps = [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row.get("ref", row.get("reference", row.get("text")))
            hyp = row.get("hyp", row.get("hypothesis", row.get("pred")))
            if ref is None or hyp is None:
                raise ValueError(
                    f"{path}:{line_no}: need ref/hyp "
                    f"(or reference/hypothesis, or text/pred); got keys={list(row)}"
                )
            refs.append(str(ref))
            hyps.append(str(hyp))
    return refs, hyps


def _self_test() -> int:
    cases = [
        {
            "name": "exact_match",
            "refs": ["hello world"],
            "hyps": ["hello world"],
            "expect_wer": 0.0,
            "expect_wer_no_punct": 0.0,
        },
        {
            "name": "one_sub_of_two",
            "refs": ["hello world"],
            "hyps": ["hello word"],
            "expect_wer": 0.5,
            "expect_wer_no_punct": 0.5,
        },
        {
            "name": "strip_asr_text_prefix",
            "refs": ["language English<asr_text>Hello World"],
            "hyps": ["hello world"],
            "expect_wer": 0.0,
            "expect_wer_no_punct": 0.0,
        },
        {
            "name": "punct_differs_between_metrics",
            "refs": ["language English<asr_text>Hello world, you are the best"],
            "hyps": ["hello world you are the best"],
            # "world," vs "world" → 1/6 with punct; 0/6 without
            "expect_wer": 1.0 / 6.0,
            "expect_wer_no_punct": 0.0,
        },
        {
            "name": "ignore_punct_apostrophe",
            "refs": ["Haven't you got any feeling?"],
            "hyps": ["havent you got any feeling"],
            "expect_wer": None,  # with punct: tokens differ
            "expect_wer_no_punct": 0.0,
        },
        {
            "name": "corpus_two_utts",
            "refs": ["hello world", "foo bar"],
            "hyps": ["hello word", "foo bar"],
            "expect_wer": 0.25,
            "expect_wer_no_punct": 0.25,
        },
        {
            "name": "empty_ref_skipped",
            "refs": ["", "hello"],
            "hyps": ["noise", "hello"],
            "expect_wer": 0.0,
            "expect_wer_no_punct": 0.0,
            "expect_n": 1,
        },
    ]

    failed = 0
    for c in cases:
        got = compute_wer(c["refs"], c["hyps"])
        ok = True
        if c.get("expect_wer") is not None and abs(got["wer"] - c["expect_wer"]) >= 1e-9:
            ok = False
        if (
            c.get("expect_wer_no_punct") is not None
            and abs(got["wer_no_punct"] - c["expect_wer_no_punct"]) >= 1e-9
        ):
            ok = False
        if "expect_n" in c and got["n"] != c["expect_n"]:
            ok = False
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(
            f"[{status}] {c['name']}: "
            f"wer={got['wer']:.4f} (expect {c.get('expect_wer')})  "
            f"wer_no_punct={got['wer_no_punct']:.4f} "
            f"(expect {c.get('expect_wer_no_punct')})  n={got['n']}"
        )
        for u in got["per_utt"]:
            print(
                f"       utt[{u['index']}] ref={u['ref']!r} hyp={u['hyp']!r} "
                f"wer={u.get('wer')} wer_no_punct={u.get('wer_no_punct')} "
                f"skipped={u['skipped_empty_ref']}"
            )
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Isolated WER check (train/loadtest protocol)")
    p.add_argument("--ref", action="append", default=[], help="Reference text (repeatable)")
    p.add_argument("--hyp", action="append", default=[], help="Hypothesis text (repeatable)")
    p.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help='JSONL with {"ref","hyp"} (or reference/hypothesis, text/pred)',
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in sanity cases with expected WER",
    )
    p.add_argument(
        "--show-norm",
        action="store_true",
        help="Print normalized ref/hyp for each pair",
    )
    args = p.parse_args()

    if args.self_test:
        return _self_test()

    if args.jsonl:
        refs, hyps = _load_jsonl_pairs(args.jsonl)
    else:
        refs, hyps = list(args.ref), list(args.hyp)

    if not refs or not hyps:
        p.error("provide --ref/--hyp pairs, --jsonl, or --self-test")
    if len(refs) != len(hyps):
        p.error(f"--ref count ({len(refs)}) != --hyp count ({len(hyps)})")

    stats = compute_wer(refs, hyps)

    if args.show_norm or len(refs) <= 20:
        print("=== Per utterance ===")
        for u in stats["per_utt"]:
            print(
                f"[{u['index']}] skipped={u['skipped_empty_ref']} "
                f"utt_wer={u.get('wer')} utt_wer_no_punct={u.get('wer_no_punct')}\n"
                f"  raw_ref: {u['ref_raw']!r}\n"
                f"  raw_hyp: {u['hyp_raw']!r}\n"
                f"  norm_ref: {u['ref']!r}\n"
                f"  norm_hyp: {u['hyp']!r}\n"
                f"  norm_ref_no_punct: {u['ref_no_punct']!r}\n"
                f"  norm_hyp_no_punct: {u['hyp_no_punct']!r}"
            )

    print("\n=== Corpus ===")
    print(f"n:                    {stats['n']}")
    print(f"wer:                  {stats['wer']:.6f}  ({stats['wer'] * 100:.2f}%)")
    print(
        f"wer_no_punct:         {stats['wer_no_punct']:.6f}  "
        f"({stats['wer_no_punct'] * 100:.2f}%)"
    )
    print(
        f"sub/del/ins (punct):  {stats['substitutions']}/"
        f"{stats['deletions']}/{stats['insertions']}  hits={stats['hits']}"
    )
    print(
        f"sub/del/ins (none):   {stats['substitutions_no_punct']}/"
        f"{stats['deletions_no_punct']}/{stats['insertions_no_punct']}  "
        f"hits={stats['hits_no_punct']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
