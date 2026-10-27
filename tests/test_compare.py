import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare import rank_methods
from benchmarks.core import Graph
from benchmarks.report import render
from benchmarks.runner import configuration


class CompareTests(unittest.TestCase):
    def test_rank_changes_and_relation_isolation(self):
        graph = Graph(None, ["a", "b", "c"], citation_edges=[("a", "b")], author_edges=[("author", "a"), ("author", "c")])
        cfg = configuration()
        cfg.update(seed_k=1, graph_weight=.9)
        rows = rank_methods(np.array([.9, .2, .1]), graph, ["a", "b", "c"], cfg, 3)
        self.assertEqual(len(rows), 4)
        cite = {r['paper_id']: r for r in rows['text_citation']['rows']}
        author = {r['paper_id']: r for r in rows['text_author']['rows']}
        self.assertGreater(cite['b']['graph_probability'], 0)
        self.assertEqual(cite['c']['graph_probability'], 0)
        self.assertGreater(author['c']['graph_probability'], 0)
        self.assertEqual(author['b']['graph_probability'], 0)
        for method in rows.values():
            for row in method['rows']:
                self.assertEqual(row['rank_change'], row['baseline_rank'] - row['rank'])

    def test_zero_graph_weight_identical_rankings(self):
        graph = Graph(None, ["a", "b"], citation_edges=[], author_edges=[])
        cfg = configuration()
        cfg['graph_weight'] = 0
        rows = rank_methods(np.array([.9, .2]), graph, ["a", "b"], cfg, 2)
        for method in rows.values():
            self.assertEqual([r['paper_id'] for r in method['rows']], ['a', 'b'])
            self.assertEqual(method['overlap_with_baseline'], 2)

    def test_report_escapes_script_content(self):
        root = Path(__file__).resolve().parents[1] / 'output' / 'benchmark_v3_tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            path = Path(temp) / 'report.html'
            render(path, {'query': '</script><img src=x onerror=alert(1)>'})
            html = path.read_text(encoding='utf-8')
            self.assertNotIn('</script><img', html)
            self.assertIn('\\u003c/script>', html)


if __name__ == '__main__':
    unittest.main()
