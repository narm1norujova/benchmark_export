#!/usr/bin/env python3
"""
Export Declaration Benchmark Script
- Labels in: <root>/labels/
- Outputs in: <root>/outputs/
- Reports in: <root>/reports/
- Matches files by number: export_1.json <-> ixrac_1.json, packing_list_4.json <-> packing_list_4.json
- Produces: benchmark_report.json, mismatch_report.json

FILE TYPE RULES (hardcoded):
  - Files #4 and #9 → packing_list  → only packing weights are benchmarked
  - All other files  → export_declaration → HS code, country, and export weights benchmarked
  - Filename keywords are used as a secondary hint but the number-based rule always wins
    for #4 and #9.
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


# ── Hardcoded packing-list file numbers ──────────────────────────────────────
PACKING_LIST_NUMBERS = {"4", "9"}


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

    def _get_file_type(self, number: str, filename: str) -> str:
        """
        Determine file type.

        PRIMARY RULE  – number-based (authoritative):
          Numbers in PACKING_LIST_NUMBERS → "packing_list"
          All others                       → "export_declaration"

        The filename is only used to emit a warning when it contradicts the
        number-based rule, so operators can catch naming mistakes early.
        """
        # Number-based decision (always wins)
        if number in PACKING_LIST_NUMBERS:
            canonical_type = "packing_list"
        else:
            canonical_type = "export_declaration"

        # Secondary: guess from filename for warning purposes only
        name_lower = filename.lower()
        if "packing_list" in name_lower or "packinglist" in name_lower:
            name_type = "packing_list"
        elif "export" in name_lower or "ixrac" in name_lower:
            name_type = "export_declaration"
        else:
            name_type = None  # can't tell from name — that's fine

        if name_type and name_type != canonical_type:
            print(
                f"  WARNING: File #{number} filename suggests '{name_type}' "
                f"but number-based rule assigns '{canonical_type}'. "
                f"Using '{canonical_type}'."
            )

        return canonical_type

    def _get_weight_by_type(self, weights_data, doc_type: str, is_label: bool = True) -> Tuple[float, float]:
        """
        Extract gross/brutto and net/netto weights for a specific doc_type.

        Labels  : list/dict with brutto_w / netto_w fields, optionally tagged with doc_type.
        Outputs : list/dict with gross / net fields, optionally tagged with doc_type.

        Fallback rule (both label and output):
          If the weights array has exactly one entry and it carries no doc_type tag,
          we treat it as matching any requested doc_type.  This handles label files
          for files #4/#9 where some groups omit the doc_type key entirely.
        """
        # ── Single weight object (not array) ──────────────────────────────
        if weights_data and isinstance(weights_data, dict):
            dt = weights_data.get("doc_type")
            if dt == doc_type or dt is None:          # untagged → accept for any type
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

        # Second pass: if the whole array has no doc_type tags at all, use first item.
        # This covers labels/outputs where doc_type was simply omitted.
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

    def _extract_output_items(self, output_data: Dict, file_type: str) -> List[Dict]:
        """
        Extract items from output data handling multiple possible formats:
        1. Direct "items" array (ProcessedItem objects)
        2. "groups" array (same structure as labels)
        """
        if "items" in output_data and isinstance(output_data["items"], list):
            return output_data["items"]
        elif "groups" in output_data and isinstance(output_data["groups"], list):
            items = []
            for group in output_data["groups"]:
                group_items = group.get("items", [])
                for _ in group_items:
                    item = {
                        "hs_code": group.get("hs_code"),
                        "country_of_origin": group.get("country_of_origin"),
                        "weight": group.get("weights", []),
                    }
                    items.append(item)
            return items
        return []

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

        # Build maps keyed by number only
        output_map: Dict[str, List[Path]] = {}
        for f in output_files:
            num = self._extract_number(f.name)
            if num:
                if num not in output_map:
                    output_map[num] = []
                output_map[num].append(f)

        label_map: Dict[str, List[Path]] = {}
        for f in label_files:
            num = self._extract_number(f.name)
            if num:
                if num not in label_map:
                    label_map[num] = []
                label_map[num].append(f)

        pairs = []
        
        for num in sorted(set(output_map.keys()) | set(label_map.keys()), key=lambda x: int(x)):
            # Determine expected file type based on number (hardcoded rule)
            if num in PACKING_LIST_NUMBERS:
                expected_type = "packing_list"
            else:
                expected_type = "export_declaration"
            
            # Get output files for this number
            outputs = output_map.get(num, [])
            labels = label_map.get(num, [])
            
            # Find matching output file by expected type
            matched_output = None
            for out_f in outputs:
                # For packing_list, look for filename with "packing" in name
                if expected_type == "packing_list":
                    if "packing" in out_f.name.lower():
                        matched_output = out_f
                        break
                else:
                    # For export_declaration, look for ixrac or export
                    if "ixrac" in out_f.name.lower() or "export" not in out_f.name.lower():
                        matched_output = out_f
                        break
            
            # Find matching label file by expected type
            matched_label = None
            for lbl_f in labels:
                if expected_type == "packing_list":
                    if "packing_list" in lbl_f.name.lower():
                        matched_label = lbl_f
                        break
                else:
                    if "export" in lbl_f.name.lower():
                        matched_label = lbl_f
                        break
            
            if matched_output and matched_label:
                pairs.append((matched_output, matched_label, num, expected_type))
            elif outputs or labels:
                print(f"  WARNING: Could not find proper pair for #{num} (expected {expected_type})")
                if outputs:
                    print(f"    Outputs found: {[f.name for f in outputs]}")
                if labels:
                    print(f"    Labels found: {[f.name for f in labels]}")
        
        return pairs

    # ------------------------------------------------------------------ #
    #  Core analysis                                                       #
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

        output_items = self._extract_output_items(output_data, file_type)
        gt_groups = self._extract_label_groups(label_data)

        # Parse price from output file
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

        # ── What to benchmark per file type ──────────────────────────────
        # export_declaration → HS code + country + Export weights ONLY
        # packing_list       → Packing weights ONLY
        check_hs_country = file_type == "export_declaration"
        check_export_weights = file_type == "export_declaration"
        check_packing_weights = file_type == "packing_list"

        feat = {
            "hs_code": 0,
            "country_numeric": 0,
            "country_alpha2": 0,
            "gross_weight_export": 0,
            "net_weight_export": 0,
            "gross_weight_packing": 0,
            "net_weight_packing": 0,
        }

        perfect_groups = 0
        perfect_items = 0
        mismatches = []
        output_idx = 0

        for g_idx, group in enumerate(gt_groups):
            group_num = g_idx + 1

            g_hs = self._normalize_hs_code(group.get("hs_code"))
            g_country = group.get("country_of_origin", {})
            g_num_code = g_country.get("numeric_code")
            g_alpha2 = g_country.get("alpha2")

            g_weights_list = group.get("weights", [])
            if not g_weights_list:
                # Legacy flat fields
                g_export_brutto = self._safe_float(group.get("brutto_w", 0.0))
                g_export_netto = self._safe_float(group.get("netto_w", 0.0))
                g_packing_brutto = 0.0
                g_packing_netto = 0.0
            else:
                g_export_brutto, g_export_netto = self._get_weight_by_type(
                    g_weights_list, "Export", is_label=True
                )
                g_packing_brutto, g_packing_netto = self._get_weight_by_type(
                    g_weights_list, "Packing", is_label=True
                )

            group_items = group.get("items", [])
            n_items = len(group_items)

            out_export_gross_sum = 0.0
            out_export_net_sum = 0.0
            out_packing_gross_sum = 0.0
            out_packing_net_sum = 0.0
            group_all_ok = True

            # ── Per-item checks ──────────────────────────────────────────
            for i in range(n_items):
                abs_idx = output_idx + i
                item_ok = True

                if abs_idx >= total_out_items:
                    mismatches.append({
                        "item_index": group_num,
                        "type": "missing_item",
                        "description": f"Expected item '{group_items[i]}' not found in output",
                        "file_type": file_type,
                    })
                    group_all_ok = False
                    continue

                out = output_items[abs_idx]

                # HS code & country (export_declaration only)
                if check_hs_country:
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
                            "output": out_hs,
                            "file_type": file_type,
                        })

                    out_country = out.get("country_of_origin") or {}
                    out_num_code = out_country.get("numeric_code")
                    out_alpha2 = out_country.get("alpha2")

                    if out_num_code == g_num_code:
                        feat["country_numeric"] += 1
                    else:
                        item_ok = False
                        group_all_ok = False
                        mismatches.append({
                            "item_index": group_num,
                            "type": "country_numeric",
                            "label": g_num_code,
                            "output": out_num_code,
                            "file_type": file_type,
                        })

                    if out_alpha2 == g_alpha2:
                        feat["country_alpha2"] += 1
                    else:
                        item_ok = False
                        group_all_ok = False
                        mismatches.append({
                            "item_index": group_num,
                            "type": "country_alpha2",
                            "label": g_alpha2,
                            "output": out_alpha2,
                            "file_type": file_type,
                        })

                # Accumulate weights from output item
                out_weight = out.get("weight")
                if out_weight:
                    if check_packing_weights:
                        # For packing-list files, read packing weight from output.
                        # _get_weight_by_type handles both tagged {"doc_type":"Packing",...}
                        # and untagged {"gross":X,"net":X} plain objects correctly.
                        pack_gross, pack_net = self._get_weight_by_type(
                            out_weight, "Packing", is_label=False
                        )
                        out_packing_gross_sum += pack_gross
                        out_packing_net_sum   += pack_net
                    else:
                        exp_gross, exp_net = self._get_weight_by_type(
                            out_weight, "Export", is_label=False
                        )
                        out_export_gross_sum += exp_gross
                        out_export_net_sum   += exp_net
                else:
                    # Old schema fallback (no weight field at all)
                    out_export_gross_sum += self._safe_float(out.get("gross_weight"))
                    out_export_net_sum   += self._safe_float(out.get("net_weight"))

                if item_ok:
                    perfect_items += 1

            # ── Group-level export weight checks ────────────────────────
            if check_export_weights and (g_export_brutto > 0 or out_export_gross_sum > 0):
                if self._weights_match(out_export_gross_sum, g_export_brutto):
                    feat["gross_weight_export"] += 1
                else:
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "gross_weight_export",
                        "label": round(g_export_brutto, 4),
                        "output": round(out_export_gross_sum, 4),
                        "file_type": file_type,
                    })

                if self._weights_match(out_export_net_sum, g_export_netto):
                    feat["net_weight_export"] += 1
                else:
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "net_weight_export",
                        "label": round(g_export_netto, 4),
                        "output": round(out_export_net_sum, 4),
                        "file_type": file_type,
                    })

            # ── Group-level packing weight checks ───────────────────────
            # Fire whenever the label has an explicit Packing entry OR output
            # produced a packing weight (catches label=0 edge cases too).
            _has_packing_label = any(
                w.get("doc_type") == "Packing" for w in group.get("weights", [])
            ) if group.get("weights") else (
                check_packing_weights and not group.get("weights")
            )
            if check_packing_weights and (_has_packing_label or out_packing_gross_sum > 0):
                if self._weights_match(out_packing_gross_sum, g_packing_brutto):
                    feat["gross_weight_packing"] += 1
                else:
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "gross_weight_packing",
                        "label": round(g_packing_brutto, 4),
                        "output": round(out_packing_gross_sum, 4),
                        "file_type": file_type,
                    })

                if self._weights_match(out_packing_net_sum, g_packing_netto):
                    feat["net_weight_packing"] += 1
                else:
                    group_all_ok = False
                    mismatches.append({
                        "item_index": group_num,
                        "type": "net_weight_packing",
                        "label": round(g_packing_netto, 4),
                        "output": round(out_packing_net_sum, 4),
                        "file_type": file_type,
                    })

            if group_all_ok:
                perfect_groups += 1

            output_idx += n_items

        # Extra items in output beyond GT
        for extra_idx in range(output_idx, total_out_items):
            mismatches.append({
                "item_index": extra_idx + 1,
                "type": "extra_item",
                "description": "Extra output item with no matching ground truth",
                "file_type": file_type,
            })

        # ── Denominator counts for weight fields ─────────────────────────
        total_gross_export = 0
        total_net_export = 0
        total_gross_packing = 0
        total_net_packing = 0

        for group in gt_groups:
            g_weights = group.get("weights", [])
            if not g_weights:
                # Legacy flat: count once if any weight exists
                if group.get("brutto_w", 0):
                    if check_export_weights:
                        total_gross_export += 1
                        total_net_export += 1
                    if check_packing_weights:
                        total_gross_packing += 1
                        total_net_packing += 1
            else:
                for w in g_weights:
                    dt = w.get("doc_type")
                    if dt == "Export" and check_export_weights:
                        total_gross_export += 1
                        total_net_export += 1
                    elif dt == "Packing" and check_packing_weights:
                        total_gross_packing += 1
                        total_net_packing += 1

        # ── Percentages ──────────────────────────────────────────────────
        def pct(num, den):
            return round(num / den * 100, 1) if den > 0 else 0.0

        hs_pct = pct(feat["hs_code"], total_gt_items) if check_hs_country else 100.0
        c_num_pct = pct(feat["country_numeric"], total_gt_items) if check_hs_country else 100.0
        c_alpha_pct = pct(feat["country_alpha2"], total_gt_items) if check_hs_country else 100.0

        export_gross_pct = pct(feat["gross_weight_export"], total_gross_export)
        export_net_pct = pct(feat["net_weight_export"], total_net_export)
        packing_gross_pct = pct(feat["gross_weight_packing"], total_gross_packing)
        packing_net_pct = pct(feat["net_weight_packing"], total_net_packing)

        # Combined weight accuracy (only over what is actually benchmarked)
        total_weight_matches = (
            feat["gross_weight_export"]
            + feat["net_weight_export"]
            + feat["gross_weight_packing"]
            + feat["net_weight_packing"]
        )
        total_weight_possible = (
            total_gross_export
            + total_net_export
            + total_gross_packing
            + total_net_packing
        )
        combined_weight_pct = pct(total_weight_matches, total_weight_possible)

        # Overall accuracy per file type
        if file_type == "export_declaration":
            # HS code + country numeric + country alpha2 + export weights
            overall_acc_pct = round(
                (hs_pct + c_num_pct + c_alpha_pct + combined_weight_pct) / 4, 1
            )
        else:
            # packing_list: weights only
            overall_acc_pct = combined_weight_pct

        item_acc_pct = pct(perfect_items, total_gt_items)

        # ── feature_matching dict ─────────────────────────────────────────
        feature_matching = {
            "hs_code": {
                "matched": feat["hs_code"],
                "total": total_gt_items if check_hs_country else 0,
                "percentage": hs_pct,
            },
            "country_numeric": {
                "matched": feat["country_numeric"],
                "total": total_gt_items if check_hs_country else 0,
                "percentage": c_num_pct,
            },
            "country_alpha2": {
                "matched": feat["country_alpha2"],
                "total": total_gt_items if check_hs_country else 0,
                "percentage": c_alpha_pct,
            },
            "gross_weight_export": {
                "matched": feat["gross_weight_export"],
                "total": total_gross_export,
                "percentage": export_gross_pct,
            },
            "net_weight_export": {
                "matched": feat["net_weight_export"],
                "total": total_net_export,
                "percentage": export_net_pct,
            },
            "gross_weight_packing": {
                "matched": feat["gross_weight_packing"],
                "total": total_gross_packing,
                "percentage": packing_gross_pct,
            },
            "net_weight_packing": {
                "matched": feat["net_weight_packing"],
                "total": total_net_packing,
                "percentage": packing_net_pct,
            },
            # Combined aliases for backward compatibility
            "gross_weight": {
                "matched": feat["gross_weight_export"] + feat["gross_weight_packing"],
                "total": total_gross_export + total_gross_packing,
                "percentage": pct(
                    feat["gross_weight_export"] + feat["gross_weight_packing"],
                    total_gross_export + total_gross_packing,
                ),
            },
            "net_weight": {
                "matched": feat["net_weight_export"] + feat["net_weight_packing"],
                "total": total_net_export + total_net_packing,
                "percentage": pct(
                    feat["net_weight_export"] + feat["net_weight_packing"],
                    total_net_export + total_net_packing,
                ),
            },
        }

        return {
            "declaration_number": number,
            "file_type": file_type,
            "label_file": label_path.name,
            "output_file": output_path.name,
            "counts": {
                "output_items": total_out_items,
                "ground_truth_items": total_gt_items,
                "ground_truth_groups": total_groups,
                "item_count_match": total_out_items == total_gt_items,
            },
            "feature_matching": feature_matching,
            "accuracy": {
                "overall_accuracy_pct": overall_acc_pct,
                "perfect_groups": perfect_groups,
                "perfect_items": perfect_items,
            },
            "mismatches": mismatches,
            "price_value": price_value,
            "price_currency": price_currency,
        }

    # ------------------------------------------------------------------ #
    #  Report builders                                                     #
    # ------------------------------------------------------------------ #

    def build_benchmark_report(self, results: List[Dict]) -> Dict:
        """Aggregate all file results into benchmark_report.json structure."""

        export_results = [r for r in results if r.get("file_type") == "export_declaration"]
        packing_results = [r for r in results if r.get("file_type") == "packing_list"]

        total_gt_items = sum(r["counts"]["ground_truth_items"] for r in results)
        total_out_items = sum(r["counts"]["output_items"] for r in results)
        total_groups = sum(r["counts"]["ground_truth_groups"] for r in results)
        total_perf_grps = sum(r["accuracy"]["perfect_groups"] for r in results)
        total_perf_itms = sum(r["accuracy"]["perfect_items"] for r in results)

        feature_keys = [
            "hs_code", "country_numeric", "country_alpha2",
            "gross_weight_export", "net_weight_export",
            "gross_weight_packing", "net_weight_packing",
            "gross_weight", "net_weight",
        ]
        feat_totals = {k: {"matched": 0, "total": 0} for k in feature_keys}

        for r in results:
            for k, v in r["feature_matching"].items():
                if k in feat_totals:
                    feat_totals[k]["matched"] += v["matched"]
                    feat_totals[k]["total"] += v["total"]

        def pct(n, d):
            return round(n / d * 100, 1) if d > 0 else 0.0

        feat_summary = {
            k: {
                "matched": v["matched"],
                "total": v["total"],
                "percentage": pct(v["matched"], v["total"]),
            }
            for k, v in feat_totals.items()
        }

        total_price = round(sum(r.get("price_value", 0.0) for r in results), 6)
        currency = results[0].get("price_currency", "USD") if results else "USD"

        # Clean results - remove extra accuracy fields
        clean_results = []
        for r in results:
            entry = {k: v for k, v in r.items() if k not in ("mismatches", "price_value", "price_currency")}
            entry["mismatch_count"] = len(r["mismatches"])
            entry["price"] = f"{round(r['price_value'], 6)} {r['price_currency']}"
            # Keep only overall_accuracy_pct in accuracy
            if "accuracy" in entry:
                entry["accuracy"] = {
                    "overall_accuracy_pct": entry["accuracy"]["overall_accuracy_pct"]
                }
            clean_results.append(entry)

        # Calculate total accuracy
        total_accuracy_item = pct(total_perf_itms, total_gt_items)

        return {
            "report_type": "BENCHMARK_REPORT",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "files_analyzed": len(results),
            "export_files_analyzed": len(export_results),
            "packing_files_analyzed": len(packing_results),
            "results": clean_results,
            "overall_statistics": {
                "total_ground_truth_items": total_gt_items,
                "total_output_items": total_out_items,
                "total_groups": total_groups,
                "total_gross_export_accuracy": feat_summary["gross_weight_export"]["percentage"],
                "total_net_export_accuracy": feat_summary["net_weight_export"]["percentage"],
                "total_gross_packing_accuracy": feat_summary["gross_weight_packing"]["percentage"],
                "total_net_packing_accuracy": feat_summary["net_weight_packing"]["percentage"],
                "total_hscode": feat_summary["hs_code"]["percentage"],
                "total_country_numeric": feat_summary["country_numeric"]["percentage"],
                "total_country_alpha2": feat_summary["country_alpha2"]["percentage"],
                "total_accuracy_item": total_accuracy_item,
                "total_price": f"{total_price} {currency}",
            },
        }
    def build_mismatch_report(self, results: List[Dict]) -> Dict:
        """Build mismatch_report.json from per-file mismatch lists."""

        files_with_mismatches = []
        summary_counts = {
            "total_hs_code_mismatches": 0,
            "total_country_numeric_mismatches": 0,
            "total_country_alpha2_mismatches": 0,
            "total_gross_weight_export_mismatches": 0,
            "total_net_weight_export_mismatches": 0,
            "total_gross_weight_packing_mismatches": 0,
            "total_net_weight_packing_mismatches": 0,
            "total_missing_items": 0,
            "total_extra_items": 0,
        }

        type_to_summary_key = {
            "hs_code": "total_hs_code_mismatches",
            "country_numeric": "total_country_numeric_mismatches",
            "country_alpha2": "total_country_alpha2_mismatches",
            "gross_weight_export": "total_gross_weight_export_mismatches",
            "net_weight_export": "total_net_weight_export_mismatches",
            "gross_weight_packing": "total_gross_weight_packing_mismatches",
            "net_weight_packing": "total_net_weight_packing_mismatches",
            "missing_item": "total_missing_items",
            "extra_item": "total_extra_items",
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
                "file_type": r.get("file_type", "unknown"),
                "label_file": r["label_file"],
                "output_file": r["output_file"],
                "item_count_match": r["counts"]["item_count_match"],
                "output_items": r["counts"]["output_items"],
                "ground_truth_items": r["counts"]["ground_truth_items"],
                "total_mismatches": len(mismatches),
                "mismatches": mismatches,
            })

        return {
            "report_type": "MISMATCH_REPORT",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "total_files_analyzed": len(results),
            "total_files_with_mismatches": len(files_with_mismatches),
            "summary": summary_counts,
            "files": files_with_mismatches,
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
        print(f"\nPacking-list file numbers (hardcoded): {sorted(PACKING_LIST_NUMBERS)}")
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
        mismatch_report = self.build_mismatch_report(all_results)

        benchmark_path = self.reports_dir / "benchmark_report.json"
        mismatch_path = self.reports_dir / "mismatch_report.json"

        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"  benchmark_report.json  -> {benchmark_path}")
        print(f"  mismatch_report.json   -> {mismatch_path}")
        ov = benchmark_report["overall_statistics"]
        print(f"\n  Files analyzed:              {benchmark_report['files_analyzed']}")
        print(f"    Export files:              {benchmark_report.get('export_files_analyzed', 0)}")
        print(f"    Packing files:             {benchmark_report.get('packing_files_analyzed', 0)}")
        print(f"  Total GT items:              {ov['total_ground_truth_items']}")
        print(f"\n  Overall Accuracy:")
        print(f"    Per item:                  {ov['total_accuracy_item']}%")
        print(f"\n  Weight Accuracy (Total):")
        print(f"    Export  Gross:  {ov['total_gross_export_accuracy']}%")
        print(f"    Export  Net:    {ov['total_net_export_accuracy']}%")
        print(f"    Packing Gross:  {ov['total_gross_packing_accuracy']}%")
        print(f"    Packing Net:    {ov['total_net_packing_accuracy']}%")
        print(f"\n  Files with mismatches:       {mismatch_report['total_files_with_mismatches']}")
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