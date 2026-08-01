from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

from notebooks.lower.brainteasers import recover_missing
from tools.integrate_source_mapped_exercises import REPLACEMENT_SELECTORS, UNITS


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "curriculum" / "interview-problem-ledger.json"


class InterviewProblemBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(LEDGER.read_text(encoding="utf-8"))

    def test_generator_is_reproducible_and_freezes_the_denominator(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/build_interview_problem_bank.py", "--check"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("catalogued=146 mapped=73 coverage=0.500", result.stdout)
        self.assertEqual(self.data["summary"]["catalogued"], 146)
        self.assertEqual(self.data["summary"]["mapped_entries"], 73)
        self.assertEqual(
            self.data["summary"]["coverage"],
            self.data["summary"]["mapped_entries"] / self.data["summary"]["catalogued"],
        )
        self.assertGreaterEqual(self.data["summary"]["coverage"], 0.50)
        self.assertEqual(
            self.data["source"]["catalog_scope"],
            "all named contents entries; every frozen entry enters the coverage denominator",
        )

    def test_four_axis_rule_and_unique_primary_home_are_enforced(self) -> None:
        ids = [item["id"] for item in self.data["problems"]]
        self.assertEqual(len(ids), len(set(ids)))
        homes = {}
        for item in self.data["problems"]:
            scores = item["scores"]
            self.assertEqual(
                set(scores),
                {"qr_relevance", "interview_frequency", "transferability", "curriculum_fit"},
            )
            self.assertTrue(all(0 <= value <= 3 for value in scores.values()))
            self.assertEqual(item["total"], sum(scores.values()))
            self.assertEqual(
                item["high_priority"],
                item["kind"] == "problem" and item["total"] >= 9 and scores["qr_relevance"] > 0,
            )
            if item["home_unit"]:
                self.assertNotIn(item["id"], homes)
                homes[item["id"]] = item["home_unit"]

    def test_catalog_matches_the_source_contents_structure(self) -> None:
        by_title = {item["original_title"]: item for item in self.data["problems"]}
        self.assertIn("LU decomposition and Cholesky decomposition", by_title)
        self.assertNotIn("Cholesky decomposition", by_title)
        self.assertEqual(by_title["Dice game III"]["source_section"], "5.3")
        self.assertTrue(all(item["kind"] == "problem" for item in self.data["problems"] if item["high_priority"]))

    def test_every_mapped_problem_has_one_question_marker_and_full_metadata(self) -> None:
        question_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "tex" / "lower" / "chapters" / "brainteasers.tex",
                ROOT / "tex" / "common" / "source-mapped-exercises.tex",
            )
        )
        mapped = [item for item in self.data["problems"] if item["home_unit"]]
        for item in mapped:
            self.assertEqual(
                len(re.findall(r"\\MFQInterviewMeta\{[^\n]+?\{" + re.escape(item["id"]) + r"(?:\s|\})", question_sources)),
                1,
                f'{item["id"]} must have exactly one primary question marker',
            )
        self.assertEqual(question_sources.count(r"\MFQInterviewMeta"), len(mapped))

    def test_brainteaser_is_a_learning_unit_and_is_published_with_hints_and_solutions(self) -> None:
        chapter = (ROOT / "tex/lower/chapters/brainteasers.tex").read_text(encoding="utf-8")
        for required in ("先缩小状态", "逻辑约束", "不变量", "条件概率", "递推", "作答协议"):
            self.assertIn(required, chapter)
        for level in ("口述概念", "笔试推导", "数值编程", "研究判断"):
            self.assertIn(rf"\subsection*{{{level}}}", chapter)
        lower_main = (ROOT / "tex/lower/main.tex").read_text(encoding="utf-8")
        solutions_main = (ROOT / "tex/solutions-main.tex").read_text(encoding="utf-8")
        self.assertLess(lower_main.index(r"\part{通用笔面试训练}"), lower_main.index(r"\part{方向模型与研究项目}"))
        self.assertIn(r"\input{tex/lower/chapters/brainteasers}", lower_main)
        self.assertIn(r"\input{tex/lower/chapters/brainteasers-hints}", lower_main)
        self.assertIn(r"\input{tex/lower/chapters/brainteasers-solutions}", solutions_main)

    def test_public_oracle_covers_invariant_and_recovery_examples(self) -> None:
        result = subprocess.run(
            [sys.executable, "notebooks/lower/brainteasers.py", "evidence/brainteasers/oracle.json"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "brainteasers=passed reachable=1 missing=8,15")

    def test_missing_number_recovery_rejects_inconsistent_ledgers(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent missing-number ledger"):
            recover_missing(2, 3, 4)

    def test_source_mapped_questions_are_audited_as_replacements(self) -> None:
        corpus = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.glob("tex/**/*.tex"))
        for unit, selectors in REPLACEMENT_SELECTORS.items():
            count = len(selectors["questions"])
            self.assertEqual(count, len(selectors["solutions"]))
            marker = f"% MFQ source replacements removed={count} unit={unit}"
            self.assertEqual(corpus.count(marker), 2, f"{unit} must replace questions and solutions")
        mapped = (ROOT / "tex/common/source-mapped-exercises.tex").read_text(encoding="utf-8")
        self.assertEqual(mapped.count(r"\MFQInterviewFollowup{"), len(REPLACEMENT_SELECTORS))

    def test_replacements_preserve_four_levels_and_followups_stay_with_their_answers(self) -> None:
        common = (ROOT / "tex/common/source-mapped-exercises.tex").read_text(encoding="utf-8")

        def macro_block(name: str) -> str:
            marker = rf"\providecommand{{\{name}}}"
            start = common.index(marker)
            end = common.find(r"\providecommand", start + len(marker))
            return common[start:] if end < 0 else common[start:end]

        for stem, (question_path, _hint_path, _solution_path) in UNITS.items():
            questions = (ROOT / question_path).read_text(encoding="utf-8")
            question_macro = macro_block(f"MFQMapped{stem}Questions")
            solution_macro = macro_block(f"MFQMapped{stem}Solutions")
            combined = questions + "\n" + question_macro
            for level in ("口述概念", "笔试推导", "数值编程", "研究判断"):
                self.assertIn(level, combined, f"{stem} lost the {level} level")
            self.assertNotIn(r"\MFQInterviewFollowup", question_macro)
            self.assertIn(r"\MFQInterviewFollowup", solution_macro)
            question_ids = set(re.findall(r"GB-\d+-\d+", question_macro))
            followup = solution_macro[solution_macro.index(r"\MFQInterviewFollowup"):]
            followup_ids = set(re.findall(r"GB-\d+-\d+", followup))
            self.assertEqual(question_ids, followup_ids, f"{stem} follow-up IDs drifted")

    def test_russian_roulette_solution_conditions_on_four_empty_chambers(self) -> None:
        solutions = (ROOT / "tex/common/source-mapped-exercises.tex").read_text(encoding="utf-8")
        self.assertIn("故直接再扣风险为 $1/4$", solutions)
        self.assertNotIn("故直接再扣风险为 $1/5$", solutions)

    def test_all_73_questions_have_one_answer_and_a_matching_followup(self) -> None:
        common = (ROOT / "tex/common/source-mapped-exercises.tex").read_text(encoding="utf-8")
        brain_questions = (ROOT / "tex/lower/chapters/brainteasers.tex").read_text(encoding="utf-8")
        brain_solutions = (ROOT / "tex/lower/chapters/brainteasers-solutions.tex").read_text(encoding="utf-8")
        mapped_ids = {item["id"] for item in self.data["problems"] if item["home_unit"]}
        question_ids = re.findall(r"\\MFQInterviewMeta\{[^\n]+?\{(GB-\d+-\d+)", common + brain_questions)
        answer_ids = re.findall(r"\\item\s+\\textbf\{(GB-\d+-\d+)。\}", common + brain_solutions)
        followup_blocks = re.findall(r"\\MFQInterviewFollowup\{(.*?)\}", common + brain_solutions, re.DOTALL)
        followup_ids = re.findall(r"GB-\d+-\d+", "\n".join(followup_blocks))
        self.assertEqual(len(question_ids), 73)
        self.assertEqual(len(answer_ids), 73)
        self.assertEqual(len(followup_ids), 73)
        self.assertEqual(set(question_ids), mapped_ids)
        self.assertEqual(set(answer_ids), mapped_ids)
        self.assertEqual(set(followup_ids), mapped_ids)
        self.assertEqual(len(question_ids), len(set(question_ids)))
        self.assertEqual(len(answer_ids), len(set(answer_ids)))
        self.assertEqual(len(followup_ids), len(set(followup_ids)))


if __name__ == "__main__":
    unittest.main()
