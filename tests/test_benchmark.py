import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from benchmarks.core import Graph, document_text, load_dataset, metrics, read, write
from benchmarks.demo import create
from benchmarks.embeddings import load_bundle, prepare
from benchmarks.runner import configuration, run


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "output" / "benchmark_v3_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        create(self.root / "fixture")
        self.data = self.root / "fixture" / "data"
        self.embeddings = self.root / "fixture" / "embeddings"
        self.cfg = configuration(self.root / "fixture" / "config.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_text_never_includes_dtm(self):
        paper = {"title": "A", "abstract": "B", "dtm": {"domain": ["SECRET"]}}
        self.assertNotIn("SECRET", document_text(paper))
        self.assertIn("SECRET", document_text(paper, True))

    def test_ndcg_uses_all_judgments_not_just_retrieved(self):
        value = metrics(["bad", "weak"], {"best": 2, "weak": 1, "bad": 0}, 2)
        self.assertAlmostEqual(value["recall"], .5)
        self.assertLess(value["ndcg"], .2)
        self.assertEqual(value["mrr"], .5)
        self.assertEqual(metrics(["bad"], {"bad": 0}, 1)["ndcg"], 0)

    def test_relation_ablation_and_probability_conservation(self):
        graph = Graph(self.data, [f"p{i}" for i in range(5)])
        seed = np.array([1., 0, 0, 0, 0])
        citation, _, converged = graph.diffuse(seed, True, False)
        author, _, _ = graph.diffuse(seed, False, True)
        both, _, _ = graph.diffuse(seed, True, True)
        self.assertTrue(converged)
        self.assertGreater(citation[1], 0)
        self.assertEqual(author[1], 0)
        self.assertGreater(author[3], 0)
        for value in (citation, author, both):
            self.assertAlmostEqual(value.sum(), 1)
            self.assertTrue((value >= 0).all())
        isolated, _, _ = graph.diffuse(np.array([0., 0, 0, 0, 1]), True, True)
        np.testing.assert_allclose(isolated, [0, 0, 0, 0, 1])

    def test_duplicate_edges_do_not_change_graph(self):
        original = Graph(self.data, [f"p{i}" for i in range(5)])
        edges = read(self.data / "authorships.json")
        write(self.data / "authorships.json", edges + edges)
        duplicated = Graph(self.data, [f"p{i}" for i in range(5)])
        np.testing.assert_equal(original.membership.toarray(), duplicated.membership.toarray())

    def test_stale_embeddings_and_unknown_graph_nodes_fail(self):
        papers, queries, _ = load_dataset(self.data)
        papers["p0"]["abstract"] = "Changed input"
        with self.assertRaises(ValueError):
            load_bundle(self.embeddings / "text", papers, queries, False)
        write(self.data / "citations.json", [["p0", "unknown"]])
        with self.assertRaises(ValueError):
            Graph(self.data, list(papers))

    def test_missing_labels_and_incomplete_pool_fail(self):
        labels = read(self.data / "qrels.json")
        labels[0]["labels"].pop()
        write(self.data / "qrels.json", labels)
        with self.assertRaises(ValueError):
            load_dataset(self.data)

    def test_all_five_variants_and_repeatability(self):
        out = self.root / "results"
        run(self.data, self.embeddings, out, self.cfg)
        second = self.root / "repeat"
        run(self.data, self.embeddings, second, self.cfg)
        summary = read(out / "summary.json")
        self.assertEqual(len(summary["metrics"]), 5)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["metrics"], read(second / "summary.json")["metrics"])
        self.assertEqual(summary["paired_deltas_vs_text"], read(second / "summary.json")["paired_deltas_vs_text"])
        self.assertEqual(len(read(out / "rankings.json")), 10)

    def test_zero_graph_weight_matches_text(self):
        self.cfg["graph_weight"] = 0
        out = self.root / "zero"
        run(self.data, self.embeddings, out, self.cfg)
        rows = read(out / "rankings.json")
        base = {r["query_id"]: r["results"] for r in rows if r["variant"] == "text"}
        for row in rows:
            if row["variant"] not in ("text", "text_dtm"):
                self.assertEqual(row["results"], base[row["query_id"]])

    def test_full_mode_and_demo_guard(self):
        self.cfg["allow_demo"] = False
        with self.assertRaises(ValueError):
            run(self.data, self.embeddings, self.root / "no-demo", self.cfg)
        self.cfg.update(allow_demo=True, mode="full")
        run(self.data, self.embeddings, self.root / "full", self.cfg)
        self.assertIn("known positives", read(self.root / "full" / "summary.json")["evaluation_note"])

    def test_labels_do_not_affect_rankings(self):
        first = self.root / "before"
        run(self.data, self.embeddings, first, self.cfg)
        judgments = read(self.data / "qrels.json")
        for row in judgments:
            for item in row["labels"]:
                item["relevance"] = 2 - item["relevance"]
        write(self.data / "qrels.json", judgments)
        second = self.root / "after"
        run(self.data, self.embeddings, second, self.cfg)
        a = read(first / "rankings.json")
        b = read(second / "rankings.json")
        self.assertEqual([r["results"] for r in a], [r["results"] for r in b])

    def test_prepare_without_gpu_and_resume(self):
        calls = []
        def encode(texts, query=False):
            calls.append((texts, query))
            return SimpleNamespace(numpy=lambda: np.ones((1, 2560), dtype=np.float32))
        encoder = SimpleNamespace(tokenizer=lambda text: {"input_ids": text.split()}, encode=encode)
        module = SimpleNamespace(LocalEmbedder=lambda: encoder, IDENTITY="fake", INSTRUCTION="retrieve papers")
        output = self.root / "prepared"
        with patch.dict(sys.modules, {"local_embeddings": module}):
            prepare(self.data, output)
            first_calls = len(calls)
            prepare(self.data, output)
            self.assertEqual(len(calls), first_calls)
        papers, queries, _ = load_dataset(self.data)
        text = load_bundle(output / "text", papers, queries, False)
        dtm = load_bundle(output / "text_dtm", papers, queries, True)
        np.testing.assert_equal(text[1], dtm[1])
        self.assertEqual(text[0].shape, (5, 2560))

    def test_prepare_rejects_silent_token_truncation(self):
        encoder = SimpleNamespace(tokenizer=lambda text: {"input_ids": [1] * 513})
        module = SimpleNamespace(LocalEmbedder=lambda: encoder, IDENTITY="fake", INSTRUCTION="retrieve")
        with patch.dict(sys.modules, {"local_embeddings": module}):
            with self.assertRaisesRegex(ValueError, "exceeds 512 tokens"):
                prepare(self.data, self.root / "too-long")


if __name__ == "__main__":
    unittest.main()
