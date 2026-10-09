"""
math_engine.py
=============================================================================
Aasaan Khata - Deterministic Math Validation Layer
=============================================================================
Post-processing engine to validate and audit VLM extraction results from
handwritten Urdu wholesale ledgers.

Handles:
1. Urdu / Arabic-Indic numerals conversion (۰۱۲۳۴۵۶۷۸۹ -> 0123456789).
2. Pakistani currency slashes and artifacts (e.g. '2500/-', '1000/=', 'Rs.').
3. Deterministic arithmetic checks:
   - Line-Item: Quantity * Unit Rate == Line Total
   - Wholesale unit conversions (e.g. Mann / 40kg, Bags).
   - Grand Total: Sum of line totals (+ expenses - discounts) == Grand Total.
4. Auto-correction suggestions (e.g. if 2 of 3 fields match).
5. HITL (Human-in-the-Loop) routing flags and audit logs.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Numeral translations
URDU_DIGITS = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

# Common wholesale unit conversion factors
UNIT_FACTORS = {
    'من': 40.0,      # 1 Mann = 40 kg
    'mann': 40.0,
    'mon': 40.0,
    'maund': 40.0,
}


def clean_numeric_string(raw: Any) -> Optional[float]:
    """
    Cleans raw OCR / VLM text string into a clean float.
    Removes currency symbols ('/-', '/=', 'Rs', 'PKR', 'روپے'), Urdu digits, commas.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    
    text = str(raw).strip()
    if not text:
        return None
    
    # Replace Urdu and Arabic digits with Western digits
    for u_digit, w_digit in URDU_DIGITS.items():
        text = text.replace(u_digit, w_digit)
    
    # Strip common Pakistani currency notation like '/-', '/=', 'Rs.', 'روپے'
    text = re.sub(r'[\/\\\-—=]+$', '', text)      # Trailing slashes or hyphens (e.g. 2500/-)
    text = re.sub(r'^(rs\.?|pkr|روپے)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*(rs\.?|pkr|روپے)$', '', text, flags=re.IGNORECASE)
    
    # Remove commas used as thousands separators
    text = text.replace(',', '')
    
    # Extract the first valid floating number found
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


@dataclass
class LineValidationResult:
    index: int
    item_name: str
    quantity: Optional[float]
    unit: str
    rate: Optional[float]
    extracted_total: Optional[float]
    calculated_total: Optional[float]
    is_valid: bool
    status: str  # 'VERIFIED', 'MISMATCH', 'MISSING_DATA', 'AUTO_CORRECTED'
    diff: float = 0.0
    suggested_correction: Optional[Dict[str, float]] = None
    warning_message: str = ""
    requires_hitl: bool = False


@dataclass
class DocumentValidationResult:
    is_valid: bool
    status: str  # 'ALL_VERIFIED', 'MATH_DISCREPANCY', 'PARTIAL_REVIEW_NEEDED'
    line_results: List[LineValidationResult] = field(default_factory=list)
    sum_of_lines: float = 0.0
    extracted_grand_total: Optional[float] = None
    grand_total_diff: float = 0.0
    grand_total_valid: bool = False
    requires_hitl: bool = False
    audit_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "sum_of_lines": self.sum_of_lines,
            "extracted_grand_total": self.extracted_grand_total,
            "grand_total_diff": self.grand_total_diff,
            "grand_total_valid": self.grand_total_valid,
            "requires_hitl": self.requires_hitl,
            "audit_summary": self.audit_summary,
            "line_results": [asdict(lr) for lr in self.line_results],
        }


class MathValidator:
    """
    Validates arithmetic in ledger data extracted by the VLM.
    """
    def __init__(self, tolerance: float = 1.0, relative_tolerance: float = 0.01):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance

    def validate_line(
        self,
        index: int,
        item_name: str,
        raw_quantity: Any,
        raw_rate: Any,
        raw_total: Any,
        unit: str = ""
    ) -> LineValidationResult:
        """
        Validates: Quantity * Rate == Line Total.
        Also checks wholesale unit factors (e.g. rate per 40kg).
        """
        qty = clean_numeric_string(raw_quantity)
        rate = clean_numeric_string(raw_rate)
        extracted_total = clean_numeric_string(raw_total)

        # Case 1: Any essential field missing
        if qty is None or rate is None or extracted_total is None:
            # Check if two fields exist and we can deduce the third
            suggested = None
            if qty is not None and rate is not None and extracted_total is None:
                suggested = {"line_total": round(qty * rate, 2)}
            elif extracted_total is not None and rate is not None and rate > 0 and qty is None:
                suggested = {"quantity": round(extracted_total / rate, 2)}
            elif extracted_total is not None and qty is not None and qty > 0 and rate is None:
                suggested = {"rate": round(extracted_total / qty, 2)}

            return LineValidationResult(
                index=index,
                item_name=item_name or "نامعلوم (Unknown)",
                quantity=qty,
                unit=unit,
                rate=rate,
                extracted_total=extracted_total,
                calculated_total=round(qty * rate, 2) if (qty is not None and rate is not None) else None,
                is_valid=False,
                status="MISSING_DATA",
                diff=0.0,
                suggested_correction=suggested,
                warning_message="Missing quantity, rate, or line total in entry.",
                requires_hitl=True
            )

        # Standard direct multiplication
        direct_calc = round(qty * rate, 2)
        diff = abs(direct_calc - extracted_total)

        # Check standard match within tolerance
        if diff <= self.tolerance or (extracted_total > 0 and (diff / extracted_total) <= self.relative_tolerance):
            return LineValidationResult(
                index=index,
                item_name=item_name,
                quantity=qty,
                unit=unit,
                rate=rate,
                extracted_total=extracted_total,
                calculated_total=direct_calc,
                is_valid=True,
                status="VERIFIED",
                diff=diff,
                requires_hitl=False
            )

        # Check if unit factor applies (e.g. Rate is per Mann / 40kg, but qty is in kg)
        unit_lower = unit.lower().strip()
        factor = UNIT_FACTORS.get(unit_lower, None)
        if factor:
            mann_calc = round((qty / factor) * rate, 2)
            mann_diff = abs(mann_calc - extracted_total)
            if mann_diff <= self.tolerance or (extracted_total > 0 and (mann_diff / extracted_total) <= self.relative_tolerance):
                return LineValidationResult(
                    index=index,
                    item_name=item_name,
                    quantity=qty,
                    unit=unit,
                    rate=rate,
                    extracted_total=extracted_total,
                    calculated_total=mann_calc,
                    is_valid=True,
                    status="VERIFIED",
                    diff=mann_diff,
                    warning_message=f"Verified using wholesale Mann (40kg) divisor.",
                    requires_hitl=False
                )

        # Arithmetic Mismatch detected
        # Formulate suggested corrections
        suggested = {}
        # Hypothesis 1: OCR misread total, Qty & Rate are correct
        suggested["suggested_total"] = direct_calc
        # Hypothesis 2: OCR misread Rate, Qty & Total are correct
        if qty > 0:
            suggested["suggested_rate"] = round(extracted_total / qty, 2)
        # Hypothesis 3: OCR misread Qty, Rate & Total are correct
        if rate > 0:
            suggested["suggested_qty"] = round(extracted_total / rate, 2)

        return LineValidationResult(
            index=index,
            item_name=item_name,
            quantity=qty,
            unit=unit,
            rate=rate,
            extracted_total=extracted_total,
            calculated_total=direct_calc,
            is_valid=False,
            status="MISMATCH",
            diff=round(diff, 2),
            suggested_correction=suggested,
            warning_message=f"Math error: {qty} × {rate} = {direct_calc}, but ledger reads {extracted_total} (Diff: {diff:.2f})",
            requires_hitl=True
        )

    def validate_document(
        self,
        extracted_data: Dict[str, Any]
    ) -> DocumentValidationResult:
        """
        Validates entire document dictionary containing items and grand total.
        Supports both schema formats:
        - {"items": [...], "grand_total": 1234}
        - {"ledger": {"transactions": [...], "total": 1234}}
        - {"entries": [...], "total": 1234}
        """
        # Normalize items list
        items = []
        if "items" in extracted_data and isinstance(extracted_data["items"], list):
            items = extracted_data["items"]
        elif "transactions" in extracted_data and isinstance(extracted_data["transactions"], list):
            items = extracted_data["transactions"]
        elif "entries" in extracted_data and isinstance(extracted_data["entries"], list):
            items = extracted_data["entries"]
        elif "ledger" in extracted_data and isinstance(extracted_data["ledger"], dict):
            sub = extracted_data["ledger"]
            items = sub.get("transactions", sub.get("items", sub.get("entries", [])))

        # Normalize grand total
        raw_grand_total = None
        for key in ["grand_total", "total", "net_amount", "balance", "total_amount"]:
            if key in extracted_data:
                raw_grand_total = extracted_data[key]
                break
        if raw_grand_total is None and "ledger" in extracted_data and isinstance(extracted_data["ledger"], dict):
            for key in ["grand_total", "total", "net_amount", "balance"]:
                if key in extracted_data["ledger"]:
                    raw_grand_total = extracted_data["ledger"][key]
                    break

        extracted_grand_total = clean_numeric_string(raw_grand_total)

        line_results: List[LineValidationResult] = []
        sum_of_lines = 0.0
        has_any_line_error = False

        for idx, entry in enumerate(items):
            if not isinstance(entry, dict):
                continue
            item_name = str(entry.get("item", entry.get("particulars", entry.get("description", entry.get("name", f"Item #{idx+1}")))))
            qty = entry.get("quantity", entry.get("qty", entry.get("count", None)))
            rate = entry.get("rate", entry.get("unit_rate", entry.get("price", None)))
            total = entry.get("line_total", entry.get("amount", entry.get("total", None)))
            unit = str(entry.get("unit", ""))

            res = self.validate_line(idx, item_name, qty, rate, total, unit)
            line_results.append(res)
            
            # Use calculated total if verified, else extracted
            if res.extracted_total is not None:
                sum_of_lines += res.extracted_total
            elif res.calculated_total is not None:
                sum_of_lines += res.calculated_total

            if not res.is_valid:
                has_any_line_error = True

        sum_of_lines = round(sum_of_lines, 2)

        # Grand total check
        grand_total_valid = False
        grand_total_diff = 0.0
        if extracted_grand_total is not None:
            grand_total_diff = abs(sum_of_lines - extracted_grand_total)
            if grand_total_diff <= self.tolerance or (extracted_grand_total > 0 and (grand_total_diff / extracted_grand_total) <= self.relative_tolerance):
                grand_total_valid = True
        else:
            grand_total_diff = 0.0

        # Overall status
        if not has_any_line_error and (grand_total_valid or extracted_grand_total is None):
            status = "ALL_VERIFIED"
            is_valid = True
            requires_hitl = False
            audit = f"All {len(line_results)} items passed math validation. Line sum: {sum_of_lines}"
        else:
            status = "MATH_DISCREPANCY"
            is_valid = False
            requires_hitl = True
            reasons = []
            if has_any_line_error:
                err_count = sum(1 for r in line_results if not r.is_valid)
                reasons.append(f"{err_count}/{len(line_results)} line items failed multiplication check")
            if extracted_grand_total is not None and not grand_total_valid:
                reasons.append(f"Sum of lines ({sum_of_lines}) != Grand Total ({extracted_grand_total}), diff: {grand_total_diff:.2f}")
            audit = " ; ".join(reasons)

        return DocumentValidationResult(
            is_valid=is_valid,
            status=status,
            line_results=line_results,
            sum_of_lines=sum_of_lines,
            extracted_grand_total=extracted_grand_total,
            grand_total_diff=round(grand_total_diff, 2),
            grand_total_valid=grand_total_valid,
            requires_hitl=requires_hitl,
            audit_summary=audit
        )
