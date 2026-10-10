"""Answer-quality evaluation. Run from the repository root with the existing venv."""
import argparse

from gitcontext import config
from gitcontext.judge import evaluate_answers, render_report


def main():
    """Evaluate the baseline and optionally rewriting; save a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite', action='store_true', help='also evaluate query rewriting')
    parser.add_argument('--limit', type=int, help='evaluate the first N questions per configuration')
    parser.add_argument('--no-cache', action='store_true', help='ignore and do not write cached results')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    from evaluate import load_questions
    questions = load_questions()
    if args.limit is not None:
        questions = questions[:args.limit]
    if not questions:
        parser.error('no evaluation questions found')
    rows = evaluate_answers(questions, rewrite=args.rewrite, no_cache=args.no_cache)
    report = render_report(rows)
    config.JUDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.JUDGE_RESULTS_PATH.write_text(report, encoding='utf-8')
    print(report)
    print(f'Saved to {config.JUDGE_RESULTS_PATH}')


if __name__ == '__main__':
    main()
