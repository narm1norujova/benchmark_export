#!/usr/bin/env python3
"""
Export Declaration Benchmark Script
- Labels in: <root>/labels/
- Outputs in: <root>/outputs/
- Reports in: <root>/reports/
- Matches files by number: export_1.json <-> ixrac_1.json
- Produces: benchmark_report.json, mismatch_report.json
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
from datetime import datetime
import os
import re


class BenchmarkTool:
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.labels_dir = self.root / "labels"
        self.outputs_dir = self.root / "outputs"
        self.reports_dir = self.root / "reports"
        self.countries_path = self.root / "countries_202603171724.csv"
        self.country_codes = self._load_country_codes()

    # ------------------------------------------------------------------ #
    #  Loaders                                                             #
    # ------------------------------------------------------------------ #

    def _load_country_codes(self) -> Dict:
        country_codes = {}
        try:
            with open(self.countries_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    numeric_code = row["numeric_code"].strip()
                    country_codes[numeric_code] = {
                        "alpha2": row["alpha2"],
                        "alpha3": row["alpha3"],
                    }
        except Exception as e:
            print(f"Error loading country codes: {e}")
            sys.exit(1)
        return country_codes

    def _load_json(self, file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _normalize_hs_code(self, hs_code) -> str:
        if hs_code is None:
            return ""
        cleaned = str(hs_code).replace(" ", "").strip()
        return cleaned[:6] if len(cleaned) >= 6 else cleaned

    def _safe_float(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _weights_match(self, w1: float, w2: float) -> bool:
        return abs(w1 - w2) <= 0.01

    def _extract_number(self, filename: str) -> Optional[str]:
        """Extract the first integer found in a filename stem."""
        numbers = re.findall(r"\d+", Path(filename).stem)
        return numbers[0] if numbers else None

    # ------------------------------------------------------------------ #
    #  File pairing                                                        #
    # ------------------------------------------------------------------ #

    def find_file_pairs(self) -> List[Tuple[Path, Path, str]]:
        """Return list of (output_file, label_file, number)."""
        output_files = sorted(self.outputs_dir.glob("*.json"))
        label_files = sorted(self.labels_dir.glob("*.json"))

        output_map = {}
        for f in output_files:
            num = self._extract_number(f.name)
            if num:
                output_map[num] = f

        label_map = {}
        for f in label_files:
            num = self._extract_number(f.name)
            if num:
                label_map[num] = f

        pairs = []
        for num in sorted(output_map.keys(), key=lambda x: int(x)):
            if num in label_map:
                pairs.append((output_map[num], label_map[num], num))
            else:
                print(f"  WARNING: No label found for output #{num} ({output_map[num].name})")

        for num in label_map:
            if num not in output_map:
                print(f"  WARNING: No output found for label #{num} ({label_map[num].name})")

        return pairs

    # ------------------------------------------------------------------ #
    #  Core analysis                                                       #
    # ------------------------------------------------------------------ #

    def analyze_pair(self, output_path: Path, label_path: Path, number: str) -> Dict:
        print(f"  Analyzing pair #{number}: {output_path.name} <-> {label_path.name}")

        output_data = self._load_json(output_path)
        label_data = self._load_json(label_path)

        if not output_data or not label_data:
            return {"error": "Failed to load files"}

        output_items = output_data.get("items", [])
        gt_groups = label_data.get("groups", [])

        # Parse price from output file (e.g. "0.004 USD")
        raw_price = output_data.get("price", "0 USD")
        try:
            parts = str(raw_price).split()
            price_value = float(parts[0])
            price_currency = parts[1] if len(parts) > 1 else "USD"
        except (ValueError, IndexError):
            price_value = 0.0
            price_currency = "USD"

        total_gt_items = sum(len(g.get("items", [])) for g in gt_groups)
        total_out_items = len(output_items)
        total_groups = len(gt_groups)

        # Counters
        feat = {"hs_code": 0, "country_numeric": 0, "country_alpha2": 0,
                "gross_weight": 0, "net_weight": 0}
        perfect_groups = 0
        perfect_items = 0

        mismatches = []   # collected per-item / per-group mismatches
        output_idx = 0

        for g_idx, group in enumerate(gt_groups):
            group_num = g_idx + 1  # 1-based for readability

            g_hs        = self._normalize_hs_code(group.get("hs_code"))
            g_country   = group.get("country_of_origin", {})
            g_num_code  = g_country.get("numeric_code")
            g_alpha2    = g_country.get("alpha2")
            g_gross     = self._safe_float(group.get("brutto_w"))
            g_net       = self._safe_float(group.get("netto_w"))
            group_items = group.get("items", [])
            n_items     = len(group_items)

            out_gross_sum = 0.0
            out_net_sum   = 0.0
            group_all_ok  = True

            for i in range(n_items):
                item_num = i + 1  # 1-based
                abs_idx  = output_idx + i
                item_ok  = True

                if abs_idx >= total_out_items:
                    # Missing item in output
                    mismatches.append({
                        "item_index": group_num,
                        "type": "missing_item",
                        "description": "Expected item not found in output"
                    })
                    group_all_ok = False
                    continue

                out = output_items[abs_idx]

                # --- HS code ---
                out_hs = self._normalize_hs_code(out.get("hs_code"))
                if out_hs and out_hs == g_hs:
                    feat["hs_code"] += 1
                else:
                    item_ok = False
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "hs_code",
                        "label": g_hs,
                        "output": out_hs
                    })

                # --- Country numeric ---
                out_country = out.get("country_of_origin", {})
                out_num_code = out_country.get("numeric_code")
                out_alpha2   = out_country.get("alpha2")

                if out_num_code and out_num_code == g_num_code:
                    feat["country_numeric"] += 1
                else:
                    item_ok = False
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "country_numeric",
                        "label": g_num_code,
                        "output": out_num_code
                    })

                # --- Country alpha2 ---
                if out_alpha2 and out_alpha2 == g_alpha2:
                    feat["country_alpha2"] += 1
                else:
                    item_ok = False
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "country_alpha2",
                        "label": g_alpha2,
                        "output": out_alpha2
                    })

                # Collect weights for group sum
                if "weight" in out:
                    out_gross_sum += self._safe_float(out["weight"].get("gross"))
                    out_net_sum   += self._safe_float(out["weight"].get("net"))
                else:
                    out_gross_sum += self._safe_float(out.get("gross_weight"))
                    out_net_sum   += self._safe_float(out.get("net_weight"))

                if item_ok:
                    perfect_items += 1

            # --- Group-level weight check ---
            gross_ok = self._weights_match(out_gross_sum, g_gross)
            net_ok   = self._weights_match(out_net_sum, g_net)

            if gross_ok:
                feat["gross_weight"] += 1
            else:
                group_all_ok = False
                mismatches.append({
                    "item_index": group_num,
                    "type": "gross_weight",
                    "label": round(g_gross, 4),
                    "output": round(out_gross_sum, 4)
                })

            if net_ok:
                feat["net_weight"] += 1
            else:
                group_all_ok = False
                mismatches.append({
                    "item_index": group_num,
                    "type": "net_weight",
                    "label": round(g_net, 4),
                    "output": round(out_net_sum, 4)
                })

            if group_all_ok:
                perfect_groups += 1

            output_idx += n_items

        # Extra items in output beyond GT
        extra_start = output_idx
        for extra_idx in range(extra_start, total_out_items):
            mismatches.append({
                "item_index": extra_idx + 1,
                "type": "extra_item",
                "description": "Extra output item with no matching ground truth"
            })

        # Percentages
        def pct(num, den):
            return round(num / den * 100, 1) if den > 0 else 0.0

        hs_pct          = pct(feat["hs_code"],          total_gt_items)
        c_num_pct       = pct(feat["country_numeric"],   total_gt_items)
        c_alpha_pct     = pct(feat["country_alpha2"],    total_gt_items)
        gross_pct       = pct(feat["gross_weight"],      total_groups)
        net_pct         = pct(feat["net_weight"],        total_groups)
        group_acc_pct   = pct(perfect_groups,            total_groups)
        item_acc_pct    = pct(perfect_items,             total_gt_items)
        overall_acc_pct = round((hs_pct + c_num_pct + c_alpha_pct + gross_pct + net_pct) / 5, 1)

        return {
            "declaration_number": number,
            "label_file": label_path.name,
            "output_file": output_path.name,
            "counts": {
                "output_items": total_out_items,
                "ground_truth_items": total_gt_items,
                "ground_truth_groups": total_groups,
                "item_count_match": total_out_items == total_gt_items
            },
            "feature_matching": {
                "hs_code":         {"matched": feat["hs_code"],        "total": total_gt_items, "percentage": hs_pct},
                "country_numeric": {"matched": feat["country_numeric"], "total": total_gt_items, "percentage": c_num_pct},
                "country_alpha2":  {"matched": feat["country_alpha2"],  "total": total_gt_items, "percentage": c_alpha_pct},
                "gross_weight":    {"matched": feat["gross_weight"],    "total": total_groups,   "percentage": gross_pct},
                "net_weight":      {"matched": feat["net_weight"],      "total": total_groups,   "percentage": net_pct}
            },
            "accuracy": {
                "overall_accuracy_pct": overall_acc_pct,
                "item_based_accuracy_pct": item_acc_pct,
                "perfect_items": perfect_items,
                "total_items": total_gt_items,
                "perfect_groups": perfect_groups,
                "total_groups": total_groups
            },
            "mismatches": mismatches,
            "price_value": price_value,
            "price_currency": price_currency
        }

    # ------------------------------------------------------------------ #
    #  Report builders                                                     #
    # ------------------------------------------------------------------ #

    def build_benchmark_report(self, results: List[Dict]) -> Dict:
        """Aggregate all file results into benchmark_report.json structure."""

        # Overall totals
        total_gt_items  = sum(r["counts"]["ground_truth_items"] for r in results)
        total_out_items = sum(r["counts"]["output_items"]       for r in results)
        total_groups    = sum(r["counts"]["ground_truth_groups"] for r in results)
        total_perf_grps = sum(r["accuracy"]["perfect_groups"]   for r in results)
        total_perf_itms = sum(r["accuracy"]["perfect_items"]    for r in results)

        feat_totals = {k: {"matched": 0, "total": 0} for k in
                       ["hs_code", "country_numeric", "country_alpha2", "gross_weight", "net_weight"]}
        for r in results:
            for k, v in r["feature_matching"].items():
                feat_totals[k]["matched"] += v["matched"]
                feat_totals[k]["total"]   += v["total"]

        def pct(n, d):
            return round(n / d * 100, 1) if d > 0 else 0.0

        feat_summary = {
            k: {
                "matched": v["matched"],
                "total":   v["total"],
                "percentage": pct(v["matched"], v["total"])
            }
            for k, v in feat_totals.items()
        }

        overall_feat_pct = round(
            sum(feat_summary[k]["percentage"] for k in feat_summary) / len(feat_summary), 1
        )

        # Total price across all output files
        total_price = round(sum(r.get("price_value", 0.0) for r in results), 6)
        currency = results[0].get("price_currency", "USD") if results else "USD"

        # Strip internal fields from per-file results
        clean_results = []
        for r in results:
            entry = {k: v for k, v in r.items() if k not in ("mismatches", "price_value", "price_currency")}
            entry["mismatch_count"] = len(r["mismatches"])
            entry["price"] = f"{round(r['price_value'], 6)} {r['price_currency']}"
            clean_results.append(entry)

        return {
            "report_type": "BENCHMARK_REPORT",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "files_analyzed": len(results),
            "results": clean_results,
            "overall_statistics": {
                "total_ground_truth_items": total_gt_items,
                "total_output_items":       total_out_items,
                "total_groups":             total_groups,
                "total_perfect_groups":     total_perf_grps,
                "total_perfect_items":      total_perf_itms,
                "overall_accuracy_pct":      round(
                    sum(r["accuracy"]["overall_accuracy_pct"] for r in results) / len(results), 1
                ),
                "overall_item_based_accuracy_pct": pct(total_perf_itms, total_gt_items),
                "feature_summary": feat_summary,
                "total_price": f"{total_price} {currency}"
            }
        }

    def build_mismatch_report(self, results: List[Dict]) -> Dict:
        """Build mismatch_report.json from per-file mismatch lists."""

        files_with_mismatches = []
        summary_counts = {
            "total_hs_code_mismatches":       0,
            "total_country_numeric_mismatches": 0,
            "total_country_alpha2_mismatches":  0,
            "total_gross_weight_mismatches":    0,
            "total_net_weight_mismatches":      0,
            "total_missing_items":             0,
            "total_extra_items":               0
        }

        type_to_summary_key = {
            "hs_code":         "total_hs_code_mismatches",
            "country_numeric": "total_country_numeric_mismatches",
            "country_alpha2":  "total_country_alpha2_mismatches",
            "gross_weight":    "total_gross_weight_mismatches",
            "net_weight":      "total_net_weight_mismatches",
            "missing_item":    "total_missing_items",
            "extra_item":      "total_extra_items"
        }

        for r in results:
            mismatches = r.get("mismatches", [])
            if not mismatches:
                continue

            for m in mismatches:
                key = type_to_summary_key.get(m["type"])
                if key:
                    summary_counts[key] += 1

            files_with_mismatches.append({
                "declaration_number": r["declaration_number"],
                "label_file":         r["label_file"],
                "output_file":        r["output_file"],
                "item_count_match":   r["counts"]["item_count_match"],
                "output_items":       r["counts"]["output_items"],
                "ground_truth_items": r["counts"]["ground_truth_items"],
                "total_mismatches":   len(mismatches),
                "mismatches":         mismatches
            })

        return {
            "report_type": "MISMATCH_REPORT",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "total_files_analyzed": len(results),
            "total_files_with_mismatches": len(files_with_mismatches),
            "summary": summary_counts,
            "files": files_with_mismatches
        }

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self):
        print("\n" + "=" * 60)
        print("EXPORT DECLARATION BENCHMARK TOOL")
        print("=" * 60)
        print(f"Root:    {self.root}")
        print(f"Labels:  {self.labels_dir}")
        print(f"Outputs: {self.outputs_dir}")
        print(f"Reports: {self.reports_dir}")
        print("=" * 60)

        # Auto-create reports folder
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Validate required directories
        for d in [self.labels_dir, self.outputs_dir]:
            if not d.exists():
                print(f"\nERROR: Directory not found: {d}")
                sys.exit(1)

        pairs = self.find_file_pairs()
        if not pairs:
            print("\nNo matching file pairs found!")
            return

        print(f"\nFound {len(pairs)} matching pair(s):\n")
        for out_f, lbl_f, num in pairs:
            print(f"  #{num}: {out_f.name} <-> {lbl_f.name}")

        print("\nRunning analysis...\n")

        all_results = []
        for out_f, lbl_f, num in pairs:
            result = self.analyze_pair(out_f, lbl_f, num)
            if "error" not in result:
                all_results.append(result)
            else:
                print(f"  ERROR in pair #{num}: {result['error']}")

        if not all_results:
            print("\nNo results to report.")
            return

        # Build reports
        benchmark_report = self.build_benchmark_report(all_results)
        mismatch_report  = self.build_mismatch_report(all_results)

        # Save reports
        benchmark_path = self.reports_dir / "benchmark_report.json"
        mismatch_path  = self.reports_dir / "mismatch_report.json"

        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"  benchmark_report.json  -> {benchmark_path}")
        print(f"  mismatch_report.json   -> {mismatch_path}")

        # Quick console summary
        ov = benchmark_report["overall_statistics"]
        print(f"\n  Files analyzed:              {benchmark_report['files_analyzed']}")
        print(f"  Total GT items:              {ov['total_ground_truth_items']}")
        print(f"  Overall accuracy:            {ov['overall_accuracy_pct']}%")
        print(f"  Overall item-based accuracy: {ov['overall_item_based_accuracy_pct']}%")
        print(f"  Files with mismatches:       {mismatch_report['total_files_with_mismatches']}")
        print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("EXPORT DECLARATION BENCHMARK TOOL")
    print("=" * 60)

    default_folder = os.getcwd()
    print(f"\nCurrent folder: {default_folder}")

    folder_path = input("\nEnter root folder path (or press Enter for current folder): ").strip()
    if not folder_path:
        folder_path = default_folder

    tool = BenchmarkTool(folder_path)
    tool.run()


if __name__ == "__main__":
    main()