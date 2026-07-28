import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LowerRouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        cls.main = (ROOT / "tex" / "lower" / "main.tex").read_text(
            encoding="utf-8"
        )

    def test_every_track_has_a_scored_diagnostic_and_three_outcomes(self) -> None:
        diagnostics = (ROOT / "tex" / "lower" / "chapters" / "route-diagnostics.tex")
        self.assertTrue(diagnostics.is_file())
        text = diagnostics.read_text(encoding="utf-8")
        self.assertIn("直接进入", text)
        self.assertIn("先完成桥接", text)
        self.assertIn("回到上册", text)
        for track in self.manifest["tracks"]:
            label = f"route:{track['id']}"
            self.assertIn(label, text)
            self.assertIn(track["title"], text)
            route = text.split(label, maxsplit=1)[1].split(r"\section", maxsplit=1)[0]
            self.assertEqual(route.count(r"\item "), 10)
            self.assertEqual(route.count(r"\RouteScoreInterpretation"), 1)

    def test_bridge_map_is_generated_from_manifest_prerequisites(self) -> None:
        bridge_map = ROOT / "tex" / "generated" / "lower-route-bridges.tex"
        self.assertTrue(bridge_map.is_file())
        text = bridge_map.read_text(encoding="utf-8")
        for track in self.manifest["tracks"]:
            self.assertIn(track["title"], text)
            for prerequisite in track["bridge_prerequisites"]:
                self.assertRegex(prerequisite, r"^upper\.ch\d{2}$")
                self.assertIn(prerequisite, text)

    def test_capstone_template_requires_research_evidence_and_claim_boundaries(self) -> None:
        template = ROOT / "tex" / "lower" / "templates" / "capstone-evidence.tex"
        self.assertTrue(template.is_file())
        text = template.read_text(encoding="utf-8")
        for marker in (
            "研究问题",
            "数据与时间协议",
            "独立基线",
            "主模型",
            "样本外设计",
            "成本与容量",
            "故障注入",
            "限制报告",
            "一键复现",
            "不得声称生产部署",
            "实盘业绩",
            "团队经验",
        ):
            self.assertIn(marker, text)
        for level in ("口述题", "推导题", "计算题", "研究判断题"):
            self.assertIn(level, text)
        self.assertIn("分级反馈", text)

    def test_lower_volume_uses_shared_contracts_and_references(self) -> None:
        for marker in (
            r"\input{tex/common/evidence-contract}",
            r"\input{tex/common/notation}",
            r"\input{tex/common/glossary}",
            r"\addbibresource{tex/common/references.bib}",
            r"\printbibliography",
            r"\input{tex/lower/chapters/route-diagnostics}",
            r"\input{tex/lower/templates/capstone-evidence}",
        ):
            self.assertIn(marker, self.main)


if __name__ == "__main__":
    unittest.main()
