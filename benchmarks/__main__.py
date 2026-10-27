import argparse
from pathlib import Path

from .core import load_dataset
from .runner import configuration, run


def main():
    parser = argparse.ArgumentParser(description="Controlled title/abstract, D/T/M and graph benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic schema/embedding fixtures; no API or GPU")
    demo.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="Validate dataset and required graph edges")
    validate.add_argument("--data", required=True)
    prep = sub.add_parser("prepare", help="Build new input-bound Qwen NF4 text and text_dtm embeddings")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--no-dtm", action="store_true", help="Only prepare title/abstract embeddings")
    bench = sub.add_parser("run", help="Run frozen experiments; no model loading or API")
    bench.add_argument("--data", required=True)
    bench.add_argument("--embeddings", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--config")
    compare_parser = sub.add_parser("compare", help="Inspect four recommendations using existing embeddings; no qrels")
    from .compare import ROOT
    from datetime import datetime
    compare_parser.add_argument("--master", type=Path, default=ROOT / "subdataset/arxiv_master_final.json")
    compare_parser.add_argument("--embeddings", type=Path, default=ROOT / "output/nl_recommendation_qwen4b_nf4/paper_text_embeddings.pt")
    compare_parser.add_argument("--citations", type=Path, default=ROOT / "subdataset/paper_paper_edges.pt")
    compare_parser.add_argument("--authors", type=Path, default=ROOT / "subdataset/author_paper_edges.pt")
    compare_parser.add_argument("--query", action="append")
    compare_parser.add_argument("--queries-json")
    compare_parser.add_argument("--interactive", action="store_true")
    compare_parser.add_argument("--open", action="store_true", help="Open generated report in default browser")
    compare_parser.add_argument("--top-k", type=int, default=10)
    compare_parser.add_argument("--config")
    compare_parser.add_argument("--output", default=str(ROOT / "output" / ("qualitative_" + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import create
        create(args.output)
    elif args.command == "validate":
        from .core import Graph
        papers, queries, labels = load_dataset(args.data)
        Graph(args.data, list(papers))
        print(f"Valid: {len(papers)} papers, {len(queries)} queries, {len(labels)} judged queries")
    elif args.command == "prepare":
        from .embeddings import prepare
        prepare(args.data, args.output, not args.no_dtm)
    elif args.command == "compare":
        from .compare import compare
        if args.interactive:
            args.query = [*(args.query or []), input("추천을 비교할 질의: ").strip()]
        compare(args)
        if args.open:
            import webbrowser
            webbrowser.open((Path(args.output) / "report.html").resolve().as_uri())
    else:
        run(args.data, args.embeddings, args.output, configuration(args.config))


if __name__ == "__main__":
    main()
