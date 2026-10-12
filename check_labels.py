"""Check that every expected answer in eval/questions.jsonl exists in the index."""
from evaluate import is_match, load_questions
from gitcontext.store import VectorStore


def main():
    """Validate labels against the selected index."""
    chunks = [p for _, p in VectorStore().all_payloads()]
    missing = 0
    for q in load_questions():
        for target in q["expected"]:
            if not any(is_match(p, target) for p in chunks):
                print(f"NOT FOUND: {target}  (question: {q['question']})")
                missing += 1
    print("All labels found in the index." if missing == 0 else f"{missing} label(s) not found.")


if __name__ == "__main__":
    from gitcontext.database import run_cli
    run_cli(main)
