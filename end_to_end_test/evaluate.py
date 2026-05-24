"""
evaluate.py
───────────
Evaluates the CRNN model by importing predictions from Predicted.py
and computing word-level accuracy, character-level accuracy,
and per-source-directory metrics.

Results are written to ``Evaluate_resutl.txt`` in the same directory.

Usage:
    python end_to_end_test/evaluate.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from datetime import datetime

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(SCRIPT_DIR, 'Evaluate_resutl.txt')

# Import prediction module
sys.path.insert(0, SCRIPT_DIR)
from Predicted import run_prediction


# ──────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = prev_row[j + 1] + 1
            deletions  = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute evaluation metrics from prediction results.

    Returns
    -------
    metrics : dict with keys:
        - total_samples
        - word_correct, word_accuracy
        - total_chars, char_correct, char_accuracy
        - avg_edit_distance
        - per_source : dict[source_name → sub-metrics]
    """
    total = len(results)
    word_correct = 0
    total_chars = 0
    char_correct = 0
    total_edit_dist = 0

    per_source = {}

    for r in results:
        true_label = r['true_label']
        pred_label = r['predicted_label']
        source     = r['source']

        # Initialise per-source if needed
        if source not in per_source:
            per_source[source] = {
                'total': 0, 'word_correct': 0,
                'total_chars': 0, 'char_correct': 0,
                'total_edit_dist': 0,
            }

        # Word-level accuracy (exact match)
        is_correct = (true_label == pred_label)
        if is_correct:
            word_correct += 1
            per_source[source]['word_correct'] += 1

        per_source[source]['total'] += 1

        # Character-level accuracy
        max_len = max(len(true_label), len(pred_label))
        for i in range(max_len):
            total_chars += 1
            per_source[source]['total_chars'] += 1
            if i < len(true_label) and i < len(pred_label) and true_label[i] == pred_label[i]:
                char_correct += 1
                per_source[source]['char_correct'] += 1

        # Edit distance
        ed = levenshtein_distance(true_label, pred_label)
        total_edit_dist += ed
        per_source[source]['total_edit_dist'] += ed

    metrics = {
        'total_samples':     total,
        'word_correct':      word_correct,
        'word_accuracy':     word_correct / max(total, 1),
        'total_chars':       total_chars,
        'char_correct':      char_correct,
        'char_accuracy':     char_correct / max(total_chars, 1),
        'avg_edit_distance': total_edit_dist / max(total, 1),
        'per_source':        {},
    }

    for src, s in per_source.items():
        metrics['per_source'][src] = {
            'total':             s['total'],
            'word_correct':      s['word_correct'],
            'word_accuracy':     s['word_correct'] / max(s['total'], 1),
            'total_chars':       s['total_chars'],
            'char_correct':      s['char_correct'],
            'char_accuracy':     s['char_correct'] / max(s['total_chars'], 1),
            'avg_edit_distance': s['total_edit_dist'] / max(s['total'], 1),
        }

    return metrics


# ──────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────
def write_report(results: list[dict], metrics: dict, path: str):
    """Write a human-readable evaluation report to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  CRNN CAPTCHA Recognition — Evaluation Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # ── Overall metrics ──
        f.write("─" * 70 + "\n")
        f.write("  OVERALL METRICS\n")
        f.write("─" * 70 + "\n")
        f.write(f"  Total samples evaluated    : {metrics['total_samples']}\n")
        f.write(f"  Word-level accuracy        : {metrics['word_correct']}/{metrics['total_samples']} "
                f"= {metrics['word_accuracy']*100:.2f}%\n")
        f.write(f"  Character-level accuracy   : {metrics['char_correct']}/{metrics['total_chars']} "
                f"= {metrics['char_accuracy']*100:.2f}%\n")
        f.write(f"  Average edit distance      : {metrics['avg_edit_distance']:.4f}\n")
        f.write("\n")

        # ── Per-source metrics ──
        f.write("─" * 70 + "\n")
        f.write("  PER-SOURCE METRICS\n")
        f.write("─" * 70 + "\n")
        for src, sm in metrics['per_source'].items():
            f.write(f"\n  [{src}]\n")
            f.write(f"    Samples              : {sm['total']}\n")
            f.write(f"    Word accuracy        : {sm['word_correct']}/{sm['total']} "
                    f"= {sm['word_accuracy']*100:.2f}%\n")
            f.write(f"    Character accuracy   : {sm['char_correct']}/{sm['total_chars']} "
                    f"= {sm['char_accuracy']*100:.2f}%\n")
            f.write(f"    Avg edit distance    : {sm['avg_edit_distance']:.4f}\n")
        f.write("\n")

        # ── Detailed predictions ──
        f.write("─" * 70 + "\n")
        f.write("  DETAILED PREDICTIONS\n")
        f.write("─" * 70 + "\n")
        f.write(f"  {'Filename':<35s} {'Source':<18s} {'True':<10s} {'Predicted':<10s} {'Match'}\n")
        f.write(f"  {'-'*35} {'-'*18} {'-'*10} {'-'*10} {'-'*5}\n")

        for r in results:
            match = 'OK' if r['true_label'] == r['predicted_label'] else 'FAIL'
            f.write(f"  {r['filename']:<35s} {r['source']:<18s} "
                    f"{r['true_label']:<10s} {r['predicted_label']:<10s} {match}\n")

        # ── Error analysis ──
        errors = [r for r in results if r['true_label'] != r['predicted_label']]
        f.write(f"\n{'─'*70}\n")
        f.write(f"  ERROR ANALYSIS\n")
        f.write(f"{'─'*70}\n")
        f.write(f"  Total errors: {len(errors)}/{len(results)}\n\n")

        if errors:
            f.write(f"  {'Filename':<35s} {'True':<10s} {'Predicted':<10s} {'Edit Dist'}\n")
            f.write(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}\n")
            for r in errors:
                ed = levenshtein_distance(r['true_label'], r['predicted_label'])
                f.write(f"  {r['filename']:<35s} {r['true_label']:<10s} "
                        f"{r['predicted_label']:<10s} {ed}\n")

        f.write(f"\n{'='*70}\n")
        f.write("  End of Report\n")
        f.write(f"{'='*70}\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run_evaluation():
    """Run prediction, compute metrics, and write the evaluation report."""
    print(f"\n{'='*60}")
    print(f"  Running Evaluation Pipeline")
    print(f"{'='*60}\n")

    # Step 1: Get predictions
    results = run_prediction()

    # Step 2: Compute metrics
    metrics = compute_metrics(results)

    # Step 3: Print summary to console
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Word-level accuracy     : {metrics['word_accuracy']*100:.2f}%")
    print(f"  Character-level accuracy: {metrics['char_accuracy']*100:.2f}%")
    print(f"  Average edit distance   : {metrics['avg_edit_distance']:.4f}")

    for src, sm in metrics['per_source'].items():
        print(f"  [{src}] Word acc: {sm['word_accuracy']*100:.2f}%, "
              f"Char acc: {sm['char_accuracy']*100:.2f}%")

    # Step 4: Write report
    write_report(results, metrics, RESULT_FILE)
    print(f"\n  Report saved → {RESULT_FILE}\n")


if __name__ == '__main__':
    run_evaluation()
