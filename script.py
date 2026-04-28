#!/usr/bin/env python3
"""
Export Declaration Benchmark Script
- Labels in: <root>/labels/
- Outputs in: <root>/outputs/
- Reports in: <root>/reports/
- Matches: export_{num}.json <-> #{num} ixrac.json
- Matches: packing_list_{num}.json <-> #{num} packing list.json
- Produces: benchmark_report.json, mismatch_report.json

ID-BASED ALIGNMENT:
  Output items carry stable IDs (item_1, item_2, ...) that correspond
  1-to-1 with ground-truth groups by position (group index → item_{index+1}).
  failed_ids lists items the model intentionally skipped (null weights).
  Items absent from both the items list and failed_ids are treated as
  genuinely missing extractions.  No positional shifting occurs.
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

    def _get_weight_by_type(self, weights_data, doc_type: str, is_label: bool = True) -> Tuple[float, float]:
        """
        Extract gross/brutto and net/netto weights for a specific doc_type.
        """
        # ── Single weight object (not array) ──────────────────────────────
        if weights_data and isinstance(weights_data, dict):
            dt = weights_data.get("doc_type")
            if dt == doc_type or dt is None:
                if is_label:
                    return (
                        self._safe_float(weights_data.get("brutto_w")),
                        self._safe_float(weights_data.get("netto_w")),
                    )
                else:
                    return (
                        self._safe_float(weights_data.get("gross")),
                        self._safe_float(weights_data.get("net")),
                    )
            return 0.0, 0.0

        # ── Array of weight objects ────────────────────────────────────────
        if not weights_data or not isinstance(weights_data, list):
            return 0.0, 0.0

        # First pass: look for an exact doc_type match
        for w in weights_data:
            if w.get("doc_type") == doc_type:
                if is_label:
                    return (
                        self._safe_float(w.get("brutto_w")),
                        self._safe_float(w.get("netto_w")),
                    )
                else:
                    return (
                        self._safe_float(w.get("gross")),
                        self._safe_float(w.get("net")),
                    )

        # Second pass: if untagged, use first item
        untagged = [w for w in weights_data if "doc_type" not in w]
        if len(untagged) == len(weights_data) and untagged:
            w = untagged[0]
            if is_label:
                return (
                    self._safe_float(w.get("brutto_w")),
                    self._safe_float(w.get("netto_w")),
                )
            else:
                return (
                    self._safe_float(w.get("gross")),
                    self._safe_float(w.get("net")),
                )

        return 0.0, 0.0

    # ------------------------------------------------------------------ #
    #  ID-based output parsing                                             #
    # ------------------------------------------------------------------ #

    def _build_output_id_map(self, output_data: Dict) -> Tuple[Dict[str, Dict], set]:
        """
        Return:
          id_map     – {item_id: item_dict} for every extracted item
          failed_ids – set of item IDs the model explicitly skipped
        
        Ground-truth group index g (0-based) maps to item ID "item_{g+1}".
        This means comparisons are always keyed by ID, never by position,
        so gaps in the output never shift subsequent comparisons.
        """
        id_map: Dict[str, Dict] = {}
        failed_ids: set = set(output_data.get("failed_ids", []))

        items = output_data.get("items", [])
        for item in items:
            item_id = item.get("id")
            if item_id:
                id_map[item_id] = item

        return id_map, failed_ids

    def _extract_label_groups(self, label_data: Dict) -> List[Dict]:
        """Extract groups from label data."""
        return label_data.get("groups", [])

    # ------------------------------------------------------------------ #
    #  File pairing                                                        #
    # ------------------------------------------------------------------ #

    def find_file_pairs(self) -> List[Tuple[Path, Path, str, str]]:
        """Return list of (output_file, label_file, number, file_type)."""
        output_files = sorted(self.outputs_dir.glob("*.json"))
        label_files = sorted(self.labels_dir.glob("*.json"))

        pairs = []

        for label_file in label_files:
            label_num = self._extract_number(label_file.name)
            if not label_num:
                continue

            label_name_lower = label_file.name.lower()
            if "packing_list" in label_name_lower:
                file_type = "packing_list"
                output_pattern = f"#{label_num} packing list.json"
            elif "export" in label_name_lower:
                file_type = "export_declaration"
                output_pattern = f"#{label_num} ixrac.json"
            else:
                continue

            output_file = self.outputs_dir / output_pattern
            if output_file.exists():
                pairs.append((output_file, label_file, label_num, file_type))
            else:
                print(f"  WARNING: No matching output found for {label_file.name} (expected: {output_pattern})")

        return sorted(pairs, key=lambda x: int(x[2]))

    # ------------------------------------------------------------------ #
    #  Core analysis (ID-based)                                            #
    # ------------------------------------------------------------------ #

    def analyze_pair(
        self, output_path: Path, label_path: Path, number: str, file_type: str
    ) -> Dict:
        print(
            f"  Analyzing {file_type} pair #{number}: "
            f"{output_path.name} <-> {label_path.name}"
        )

        output_data = self._load_json(output_path)
        label_data = self._load_json(label_path)

        if not output_data or not label_data:
            return {"error": "Failed to load files"}

        # ── Build ID-keyed structures ──────────────────────────────────────
        # id_map: item_id -> output item dict (only successfully extracted)
        # failed_ids: set of item_ids the model intentionally skipped
        id_map, failed_ids = self._build_output_id_map(output_data)

        gt_groups = self._extract_label_groups(label_data)

        # Parse price and tokens from output file
        raw_price = output_data.get("price", "0 USD")
        try:
            parts = str(raw_price).split()
            price_value = float(parts[0])
            price_currency = parts[1] if len(parts) > 1 else "USD"
        except (ValueError, IndexError):
            price_value = 0.0
            price_currency = "USD"

        tokens = output_data.get("tokens", 0)
        if isinstance(tokens, str):
            try:
                tokens = int(tokens)
            except Exception:
                tokens = 0

        total_gt_items = sum(len(g.get("items", [])) for g in gt_groups)
        total_out_items = len(id_map)          # only successfully extracted
        total_groups = len(gt_groups)

        # ── What to benchmark per file type ──────────────────────────────
        check_hs_country    = file_type == "export_declaration"
        check_export_weights  = file_type == "export_declaration"
        check_packing_weights = file_type == "packing_list"

        field_matches = {
            "hs_code":            {"matched": 0, "total": 0},
            "country_of_origin":  {"matched": 0, "total": 0},
            "weight.gross":       {"matched": 0, "total": 0},
            "weight.net":         {"matched": 0, "total": 0},
        }

        perfect_items = 0
        mismatches = []

        # ── Iterate ground-truth groups; look up output by stable item ID ──
        #
        # Group index g (0-based) → item ID "item_{g+1}"
        # This is the only mapping needed; no running counter required.
        #
        # Outcome for each group:
        #   A) item present in id_map          → normal field comparison
        #   B) item in failed_ids only         → model correctly skipped
        #                                         (null weight in GT confirms)
        #   C) item absent from both           → genuine extraction failure
        #
        for g_idx, group in enumerate(gt_groups):
            item_id  = f"item_{g_idx + 1}"   # stable, position-independent
            group_num = g_idx + 1             # 1-based for human-readable reports

            # ── Ground-truth values ───────────────────────────────────────
            g_hs       = self._normalize_hs_code(group.get("hs_code"))
            g_country  = group.get("country_of_origin", {})
            g_num_code = g_country.get("numeric_code")

            g_weights_list = group.get("weights", [])
            if not g_weights_list:
                g_export_brutto = self._safe_float(group.get("brutto_w", 0.0))
                g_export_netto  = self._safe_float(group.get("netto_w",  0.0))
                g_packing_brutto = 0.0
                g_packing_netto  = 0.0
            else:
                g_export_brutto, g_export_netto = self._get_weight_by_type(
                    g_weights_list, "Export", is_label=True
                )
                g_packing_brutto, g_packing_netto = self._get_weight_by_type(
                    g_weights_list, "Packing", is_label=True
                )

            # Determine whether this group's weight is genuinely null in GT
            gt_weight_is_null = (
                all(
                    w.get("brutto_w") is None and w.get("netto_w") is None
                    for w in (g_weights_list if g_weights_list else [])
                )
                if g_weights_list
                else (group.get("brutto_w") is None and group.get("netto_w") is None)
            )

            group_items = group.get("items", [])
            n_items = len(group_items)

            # ── Outcome A: item was extracted ─────────────────────────────
            if item_id in id_map:
                out = id_map[item_id]
                item_ok = True

                # HS code & country (export_declaration only)
                if check_hs_country:
                    field_matches["hs_code"]["total"] += 1
                    out_hs = self._normalize_hs_code(out.get("hs_code"))
                    if out_hs and out_hs == g_hs:
                        field_matches["hs_code"]["matched"] += 1
                    else:
                        item_ok = False
                        mismatches.append({
                            "item_id": item_id,
                            "item_index": group_num,
                            "type": "hs_code",
                            "label": g_hs,
                            "output": out_hs,
                            "file_type": file_type,
                        })

                    field_matches["country_of_origin"]["total"] += 1
                    out_country  = out.get("country_of_origin") or {}
                    out_num_code = out_country.get("numeric_code")
                    if out_num_code == g_num_code:
                        field_matches["country_of_origin"]["matched"] += 1
                    else:
                        item_ok = False
                        mismatches.append({
                            "item_id": item_id,
                            "item_index": group_num,
                            "type": "country",
                            "label": g_num_code,
                            "output": out_num_code,
                            "file_type": file_type,
                        })

                # Weight comparison
                out_weight = out.get("weight")

                if check_packing_weights:
                    out_gross, out_net = self._get_weight_by_type(
                        out_weight, "Packing", is_label=False
                    ) if out_weight else (0.0, 0.0)

                    if g_packing_brutto > 0 or out_gross > 0:
                        field_matches["weight.gross"]["total"] += 1
                        if self._weights_match(out_gross, g_packing_brutto):
                            field_matches["weight.gross"]["matched"] += 1
                        else:
                            item_ok = False
                            mismatches.append({
                                "item_id": item_id,
                                "item_index": group_num,
                                "type": "gross_weight",
                                "label": round(g_packing_brutto, 4),
                                "output": round(out_gross, 4),
                                "file_type": file_type,
                            })

                        field_matches["weight.net"]["total"] += 1
                        if self._weights_match(out_net, g_packing_netto):
                            field_matches["weight.net"]["matched"] += 1
                        else:
                            item_ok = False
                            mismatches.append({
                                "item_id": item_id,
                                "item_index": group_num,
                                "type": "net_weight",
                                "label": round(g_packing_netto, 4),
                                "output": round(out_net, 4),
                                "file_type": file_type,
                            })

                elif check_export_weights:
                    out_gross, out_net = self._get_weight_by_type(
                        out_weight, "Export", is_label=False
                    ) if out_weight else (
                        self._safe_float(out.get("gross_weight")),
                        self._safe_float(out.get("net_weight")),
                    )

                    if g_export_brutto > 0 or out_gross > 0:
                        field_matches["weight.gross"]["total"] += 1
                        if self._weights_match(out_gross, g_export_brutto):
                            field_matches["weight.gross"]["matched"] += 1
                        else:
                            item_ok = False
                            mismatches.append({
                                "item_id": item_id,
                                "item_index": group_num,
                                "type": "gross_weight",
                                "label": round(g_export_brutto, 4),
                                "output": round(out_gross, 4),
                                "file_type": file_type,
                            })

                        field_matches["weight.net"]["total"] += 1
                        if self._weights_match(out_net, g_export_netto):
                            field_matches["weight.net"]["matched"] += 1
                        else:
                            item_ok = False
                            mismatches.append({
                                "item_id": item_id,
                                "item_index": group_num,
                                "type": "net_weight",
                                "label": round(g_export_netto, 4),
                                "output": round(out_net, 4),
                                "file_type": file_type,
                            })

                if item_ok:
                    perfect_items += 1

            # ── Outcome B: model correctly skipped (in failed_ids) ────────
            elif item_id in failed_ids:
                # Model declared it couldn't extract this item.
                # If GT weight is also null this is correct behaviour —
                # we do not penalise it or count it toward any field total.
                # If GT has a real weight the model should have extracted it.
                if not gt_weight_is_null:
                    mismatches.append({
                        "item_id": item_id,
                        "item_index": group_num,
                        "type": "incorrectly_skipped",
                        "description": (
                            "Model put this item in failed_ids but ground truth "
                            "contains a non-null weight — extraction was expected."
                        ),
                        "file_type": file_type,
                    })
                # If GT weight IS null: silently pass — correct skip.

            # ── Outcome C: completely missing ─────────────────────────────
            else:
                mismatches.append({
                    "item_id": item_id,
                    "item_index": group_num,
                    "type": "missing_item",
                    "description": (
                        "Item absent from both output items list and failed_ids."
                    ),
                    "file_type": file_type,
                })

        # ── Extra items in output that have no GT group at all ─────────────
        all_known_ids = {f"item_{i + 1}" for i in range(len(gt_groups))}
        extra_ids = set(id_map.keys()) - all_known_ids
        for extra_id in sorted(extra_ids):
            mismatches.append({
                "item_id": extra_id,
                "item_index": None,
                "type": "extra_item",
                "description": "Output item has no corresponding ground-truth group.",
                "file_type": file_type,
            })

        return {
            "declaration_number": number,
            "file_type": file_type,
            "label_file": label_path.name,
            "output_file": output_path.name,
            "counts": {
                "output_items": total_out_items,
                "ground_truth_items": total_gt_items,
                "ground_truth_groups": total_groups,
                "failed_ids_count": len(failed_ids),
                "item_count_match": total_out_items == total_gt_items,
            },
            "field_matches": field_matches,
            "perfect_items": perfect_items,
            "has_mismatch": len(mismatches) > 0,
            "mismatches": mismatches,
            "price_value": price_value,
            "price_currency": price_currency,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------ #
    #  Report builders                                                     #
    # ------------------------------------------------------------------ #

    def build_benchmark_report(self, results: List[Dict]) -> Dict:
        """Aggregate all file results into benchmark_report.json structure."""

        export_results  = [r for r in results if r.get("file_type") == "export_declaration"]
        packing_results = [r for r in results if r.get("file_type") == "packing_list"]

        total_documents = len(results)

        # ── Export Statistics ──────────────────────────────────────────────
        export_field_accuracies = []

        if export_results:
            total_hs_matched = sum(r["field_matches"]["hs_code"]["matched"] for r in export_results)
            total_hs_total   = sum(r["field_matches"]["hs_code"]["total"]   for r in export_results)
            export_field_accuracies.append({
                "field_name": "hs_code",
                "accuracy": round(total_hs_matched / total_hs_total, 3) if total_hs_total > 0 else 1.0,
            })

            total_country_matched = sum(r["field_matches"]["country_of_origin"]["matched"] for r in export_results)
            total_country_total   = sum(r["field_matches"]["country_of_origin"]["total"]   for r in export_results)
            export_field_accuracies.append({
                "field_name": "country_of_origin",
                "accuracy": round(total_country_matched / total_country_total, 3) if total_country_total > 0 else 1.0,
            })

            total_gross_matched = sum(r["field_matches"]["weight.gross"]["matched"] for r in export_results)
            total_gross_total   = sum(r["field_matches"]["weight.gross"]["total"]   for r in export_results)
            export_field_accuracies.append({
                "field_name": "weight.gross",
                "accuracy": round(total_gross_matched / total_gross_total, 3) if total_gross_total > 0 else 1.0,
            })

            total_net_matched = sum(r["field_matches"]["weight.net"]["matched"] for r in export_results)
            total_net_total   = sum(r["field_matches"]["weight.net"]["total"]   for r in export_results)
            export_field_accuracies.append({
                "field_name": "weight.net",
                "accuracy": round(total_net_matched / total_net_total, 3) if total_net_total > 0 else 1.0,
            })

            export_failed_count = sum(1 for r in export_results if r["has_mismatch"])
            export_failed_rate  = round(export_failed_count / len(export_results), 3) if export_results else 0
            export_total_cost   = sum(r["price_value"] for r in export_results)
            export_avg_cost     = round(export_total_cost / len(export_results), 3) if export_results else 0
            export_total_tokens = sum(r["tokens"] for r in export_results)
            export_avg_tokens   = int(export_total_tokens / len(export_results)) if export_results else 0
            export_total_items  = sum(r["counts"]["ground_truth_items"] for r in export_results)
        else:
            export_failed_rate = export_total_cost = export_avg_cost = 0
            export_total_tokens = export_avg_tokens = export_total_items = 0

        # ── Packing Statistics ─────────────────────────────────────────────
        packing_field_accuracies = []

        if packing_results:
            total_gross_matched = sum(r["field_matches"]["weight.gross"]["matched"] for r in packing_results)
            total_gross_total   = sum(r["field_matches"]["weight.gross"]["total"]   for r in packing_results)
            packing_field_accuracies.append({
                "field_name": "weight.gross",
                "accuracy": round(total_gross_matched / total_gross_total, 3) if total_gross_total > 0 else 1.0,
            })

            total_net_matched = sum(r["field_matches"]["weight.net"]["matched"] for r in packing_results)
            total_net_total   = sum(r["field_matches"]["weight.net"]["total"]   for r in packing_results)
            packing_field_accuracies.append({
                "field_name": "weight.net",
                "accuracy": round(total_net_matched / total_net_total, 3) if total_net_total > 0 else 1.0,
            })

            packing_failed_count = sum(1 for r in packing_results if r["has_mismatch"])
            packing_failed_rate  = round(packing_failed_count / len(packing_results), 3) if packing_results else 0
            packing_total_cost   = sum(r["price_value"] for r in packing_results)
            packing_avg_cost     = round(packing_total_cost / len(packing_results), 3) if packing_results else 0
            packing_total_tokens = sum(r["tokens"] for r in packing_results)
            packing_avg_tokens   = int(packing_total_tokens / len(packing_results)) if packing_results else 0
            packing_total_items  = sum(r["counts"]["ground_truth_items"] for r in packing_results)
        else:
            packing_failed_rate = packing_total_cost = packing_avg_cost = 0
            packing_total_tokens = packing_avg_tokens = packing_total_items = 0

        # Clean results (strip internal mismatch detail for benchmark report)
        clean_results = []
        for r in results:
            clean_results.append({
                "declaration_number": r["declaration_number"],
                "file_type":          r["file_type"],
                "label_file":         r["label_file"],
                "output_file":        r["output_file"],
                "counts":             r["counts"],
                "field_matches":      r["field_matches"],
                "has_mismatch":       r["has_mismatch"],
                "price":              f"{round(r['price_value'], 6)} {r['price_currency']}",
                "tokens":             r["tokens"],
            })

        return {
            "report_type": "BENCHMARK_REPORT",
            "generated":   datetime.now().isoformat(timespec="seconds"),
            "results":     clean_results,
            "overall_statistics": {
                "total_documents": total_documents,
                "export": {
                    "total_documents":  len(export_results),
                    "total_items":      export_total_items,
                    "field_accuracies": export_field_accuracies,
                    "failed_ids_rate":  export_failed_rate,
                    "cost_metrics": {
                        "avg_cost_per_doc":  export_avg_cost,
                        "total_cost":        round(export_total_cost, 3),
                        "avg_tokens_per_doc": export_avg_tokens,
                        "total_tokens":      export_total_tokens,
                    },
                },
                "packing": {
                    "total_documents":  len(packing_results),
                    "total_items":      packing_total_items,
                    "field_accuracies": packing_field_accuracies,
                    "failed_ids_rate":  packing_failed_rate,
                    "cost_metrics": {
                        "avg_cost_per_doc":  packing_avg_cost,
                        "total_cost":        round(packing_total_cost, 3),
                        "avg_tokens_per_doc": packing_avg_tokens,
                        "total_tokens":      packing_total_tokens,
                    },
                },
            },
        }

    def build_mismatch_report(self, results: List[Dict]) -> Dict:
        """Build mismatch_report.json from per-file mismatch lists."""

        files_with_mismatches = []
        summary_counts = {
            "total_hs_code_mismatches":        0,
            "total_country_mismatches":        0,
            "total_gross_weight_mismatches":   0,
            "total_net_weight_mismatches":     0,
            "total_missing_items":             0,
            "total_extra_items":               0,
            "total_incorrectly_skipped":       0,
        }

        type_to_summary_key = {
            "hs_code":             "total_hs_code_mismatches",
            "country":             "total_country_mismatches",
            "gross_weight":        "total_gross_weight_mismatches",
            "net_weight":          "total_net_weight_mismatches",
            "missing_item":        "total_missing_items",
            "extra_item":          "total_extra_items",
            "incorrectly_skipped": "total_incorrectly_skipped",
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
                "declaration_number":  r["declaration_number"],
                "file_type":           r.get("file_type", "unknown"),
                "label_file":          r["label_file"],
                "output_file":         r["output_file"],
                "item_count_match":    r["counts"]["item_count_match"],
                "output_items":        r["counts"]["output_items"],
                "ground_truth_items":  r["counts"]["ground_truth_items"],
                "failed_ids_count":    r["counts"]["failed_ids_count"],
                "total_mismatches":    len(mismatches),
                "mismatches":          mismatches,
            })

        return {
            "report_type":                 "MISMATCH_REPORT",
            "generated":                   datetime.now().isoformat(timespec="seconds"),
            "total_files_analyzed":        len(results),
            "total_files_with_mismatches": len(files_with_mismatches),
            "summary":                     summary_counts,
            "files":                       files_with_mismatches,
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
        print("\nMatching rules:")
        print("  export_{num}.json      <-> #{num} ixrac.json")
        print("  packing_list_{num}.json <-> #{num} packing list.json")
        print("=" * 60)

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        for d in [self.labels_dir, self.outputs_dir]:
            if not d.exists():
                print(f"\nERROR: Directory not found: {d}")
                sys.exit(1)

        pairs = self.find_file_pairs()
        if not pairs:
            print("\nNo matching file pairs found!")
            return

        print(f"\nFound {len(pairs)} matching pair(s):\n")
        for out_f, lbl_f, num, ftype in pairs:
            print(f"  #{num:>3} ({ftype}): {out_f.name} <-> {lbl_f.name}")

        print("\nRunning analysis...\n")

        all_results = []
        for out_f, lbl_f, num, ftype in pairs:
            result = self.analyze_pair(out_f, lbl_f, num, ftype)
            if "error" not in result:
                all_results.append(result)
            else:
                print(f"  ERROR in pair #{num} ({ftype}): {result['error']}")

        if not all_results:
            print("\nNo results to report.")
            return

        benchmark_report = self.build_benchmark_report(all_results)
        mismatch_report  = self.build_mismatch_report(all_results)

        benchmark_path = self.reports_dir / "benchmark_report.json"
        mismatch_path  = self.reports_dir / "mismatch_report.json"

        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"  benchmark_report.json -> {benchmark_path}")
        print(f"  mismatch_report.json  -> {mismatch_path}")

        ov = benchmark_report["overall_statistics"]
        print(f"\n  Total Documents:              {ov['total_documents']}")
        print(f"\n  Export Documents:             {ov['export']['total_documents']}")
        print(f"    Total Items:                 {ov['export']['total_items']}")
        print(f"    Failed IDs Rate:             {ov['export']['failed_ids_rate']}")
        print(f"    Cost Metrics:")
        print(f"      Avg cost/doc:              ${ov['export']['cost_metrics']['avg_cost_per_doc']}")
        print(f"      Total cost:                ${ov['export']['cost_metrics']['total_cost']}")
        print(f"      Avg tokens/doc:            {ov['export']['cost_metrics']['avg_tokens_per_doc']}")
        print(f"      Total tokens:              {ov['export']['cost_metrics']['total_tokens']}")

        print(f"\n  Packing Documents:            {ov['packing']['total_documents']}")
        print(f"    Total Items:                 {ov['packing']['total_items']}")
        print(f"    Failed IDs Rate:             {ov['packing']['failed_ids_rate']}")
        print(f"    Cost Metrics:")
        print(f"      Avg cost/doc:              ${ov['packing']['cost_metrics']['avg_cost_per_doc']}")
        print(f"      Total cost:                ${ov['packing']['cost_metrics']['total_cost']}")
        print(f"      Avg tokens/doc:            {ov['packing']['cost_metrics']['avg_tokens_per_doc']}")
        print(f"      Total tokens:              {ov['packing']['cost_metrics']['total_tokens']}")

        print(f"\n  Files with mismatches:        {mismatch_report['total_files_with_mismatches']}")
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