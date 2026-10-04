"""Reproducible input and result filters for structured corpora."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from .cluster import BackendError, normalise


@dataclass(frozen=True)
class FilterSpec:
    """A saved, reviewable set of transformations and exclusions."""

    json_fields: tuple[str, ...] = ()
    html_visible_text: bool = False
    exclude_context: tuple[re.Pattern[str], ...] = ()
    include_context: tuple[re.Pattern[str], ...] = ()
    exclude_terms: frozenset[str] = frozenset()
    source: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> "FilterSpec":
        path = Path(path)
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise BackendError(f"filter file could not be read: {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"{path}:{exc.lineno}: filter file is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise BackendError(f"{path}: filter must be a JSON object")
        if data.get("version", 1) != 1:
            raise BackendError(f"{path}: unsupported filter version {data.get('version')!r}")

        input_options = data.get("input", {})
        if not isinstance(input_options, dict):
            raise BackendError(f"{path}: input must be an object")
        fields = _string_list(input_options.get("json_fields", []), path, "input.json_fields")
        html_mode = input_options.get("html", "raw")
        if html_mode not in {"raw", "visible_text"}:
            raise BackendError(f"{path}: input.html must be 'raw' or 'visible_text'")

        exclude = data.get("exclude", {})
        include = data.get("include", {})
        if not isinstance(exclude, dict) or not isinstance(include, dict):
            raise BackendError(f"{path}: include and exclude must be objects")
        excluded_patterns = _patterns(exclude.get("context_regex", []), path, "exclude.context_regex")
        included_patterns = _patterns(include.get("context_regex", []), path, "include.context_regex")
        terms = frozenset(
            normalise(term)
            for term in _string_list(exclude.get("terms", []), path, "exclude.terms")
        )
        return cls(
            json_fields=tuple(fields),
            html_visible_text=html_mode == "visible_text",
            exclude_context=tuple(excluded_patterns),
            include_context=tuple(included_patterns),
            exclude_terms=terms,
            source=data,
        )

    def prepare(self, raw: str) -> str | None:
        """Extract searchable text and apply whole-record context filters."""
        text = raw
        if self.json_fields:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                return None
            values = [_json_value(record, field) for field in self.json_fields]
            text = " ".join(value for value in values if value)
        if self.html_visible_text:
            text = visible_text(text)
        if self.include_context and not any(pattern.search(text) for pattern in self.include_context):
            return None
        if any(pattern.search(text) for pattern in self.exclude_context):
            return None
        return text

    def keep_matches(self, matches: Iterable) -> list:
        if not self.exclude_terms:
            return list(matches)
        return [match for match in matches if normalise(match.term.text) not in self.exclude_terms]

    def as_dict(self) -> dict:
        if self.source is not None:
            return self.source
        return {"version": 1}


def _string_list(value, path: Path, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BackendError(f"{path}: {field} must be an array of strings")
    return value


def _patterns(value, path: Path, field: str) -> list[re.Pattern[str]]:
    patterns = []
    for expression in _string_list(value, path, field):
        try:
            patterns.append(re.compile(expression))
        except re.error as exc:
            raise BackendError(f"{path}: invalid regex in {field}: {exc}") from exc
    return patterns


def _json_value(record, dotted_path: str) -> str:
    value = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return ""
        value = value[part]
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "template"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "template"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def visible_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(value)
    parser.close()
    return " ".join(" ".join(parser.parts).split())
