"""Step 4: custom Guardrails AI validators for PII and JSON."""

from __future__ import annotations

import io
import json
import os
import re
from contextlib import redirect_stdout

from config import EVIDENCE_DIR, ensure_dirs

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("GUARDRAILS_DISABLE_TELEMETRY", "true")

try:
    from guardrails import Guard, OnFailAction
except ImportError:
    from guardrails import Guard
    from guardrails.validator_base import OnFailAction

try:
    from guardrails import Validator, register_validator
except ImportError:
    from guardrails.validator_base import Validator, register_validator

try:
    from guardrails.validators import FailResult, PassResult
except ImportError:
    from guardrails.validator_base import FailResult, PassResult


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detect and redact common PII with regular expressions."""

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted = redacted.replace(match, f"[{pii_type}_REDACTED]")
                found.append(pii_type)

        if found:
            return FailResult(
                error_message=f"Detected PII types: {', '.join(sorted(set(found)))}",
                fix_value=redacted,
            )
        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validate and repair simple JSON formatting problems."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as exc:
            fallback = json.dumps(
                {"error": "Invalid JSON after repair attempt", "raw": value, "detail": str(exc)}
            )
            return FailResult(error_message=str(exc), fix_value=fallback)


def demo_pii_guard() -> None:
    print("=" * 55)
    print("PII Detection Demo")
    print("=" * 55)
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    cases = [
        ("Clean", "No sensitive information in this text."),
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call support at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email alice@example.com or call 555-123-4567."),
    ]
    validator = PIIDetector(on_fail=OnFailAction.FIX)
    for label, text in cases:
        result = guard.validate(text)
        direct = validator.validate(text, {})
        output = getattr(direct, "fix_value", None) or getattr(direct, "value_override", None) or text
        print(f"\n[{label}]")
        print(f"Input : {text}")
        print(f"Passed: {result.validation_passed}")
        print(f"Output: {output}")


def demo_json_guard() -> None:
    print("=" * 55)
    print("JSON Formatting Demo")
    print("=" * 55)
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Broken", "This is not JSON at all: ??? {]"),
    ]
    validator = JSONFormatter(on_fail=OnFailAction.FIX)
    for label, text in cases:
        result = guard.validate(text)
        direct = validator.validate(text, {})
        output = getattr(direct, "fix_value", None) or getattr(direct, "value_override", None) or text
        print(f"\n[{label}]")
        print(f"Input : {text}")
        print(f"Passed: {result.validation_passed}")
        print(f"Output: {output}")


def _run_and_save(func, path):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func()
    output = buffer.getvalue()
    print(output, end="")
    path.write_text(output, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("=" * 55)
    print("Step 4: Guardrails AI Validators")
    print("=" * 55)
    _run_and_save(demo_pii_guard, EVIDENCE_DIR / "04_pii_demo_log.txt")
    _run_and_save(demo_json_guard, EVIDENCE_DIR / "04_json_demo_log.txt")
    print("Step 4 complete. Saved Guardrails logs in evidence/.")


if __name__ == "__main__":
    main()
