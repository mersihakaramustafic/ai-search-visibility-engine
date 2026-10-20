"""Deterministic citation extraction from stored provider metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from visibility_extractor.config import CitationConfig
from visibility_extractor.matching import URL_RE, Brand

DOMAIN_RE = re.compile(r"^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}$")
GEMINI_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
TRAILING_PUNCTUATION = ".,;:!?*_"


@dataclass(frozen=True)
class Citation:
    citation_position: int
    url: str
    domain: str | None
    title: str | None
    brand_id: str | None
    citation_source: str  # 'provider_metadata' | 'response_text'
    provider_format: str  # 'openai_url_citation' | 'gemini_grounding_chunk' | 'text_url'
    metadata: dict[str, Any]
    is_target_brand_domain: bool | None


@dataclass
class CitationResult:
    citations: list[Citation] = field(default_factory=list)
    masked_spans: list[tuple[int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, **fields: Any) -> None:
        self.citations.append(Citation(citation_position=len(self.citations) + 1, **fields))


def normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    host = value.strip().lower().rstrip(".").removeprefix("www.")
    return host if DOMAIN_RE.match(host) else None


def split_url(url: str) -> tuple[str | None, str]:
    try:
        parts = urlsplit(url)
        return normalize_domain(parts.hostname), parts.path
    except ValueError:
        return None, ""


class DomainAttributor:
    def __init__(self, brands: list[Brand], config: CitationConfig):
        known = {b.brand_id for b in brands}
        unknown = sorted(set(config.brand_url_rules) - known)
        if unknown:
            raise ValueError(f"unknown brand_id(s) in citations.brand_url_rules: {unknown}")
        self.brands = brands
        self.rules = config.brand_url_rules
        self.target_domains = [b.canonical_domain for b in brands if b.is_primary_brand]

    @staticmethod
    def _on(domain: str, canonical: str) -> bool:
        return domain == canonical or domain.endswith("." + canonical)

    def brand_for(self, domain: str | None, path: str | None) -> tuple[str | None, dict[str, str]]:
        """(brand_id or None, metadata note explaining a skipped attribution)."""
        if not domain:
            return None, {}
        for b in self.brands:
            if not self._on(domain, b.canonical_domain):
                continue
            rules = self.rules.get(b.brand_id)
            if not rules:
                return b.brand_id, {}
            if path is None:
                return None, {"brand_attribution": f"skipped: {b.canonical_domain} is shared and no path is known"}
            if any(r in f"{domain}{path}".lower() for r in rules):
                return b.brand_id, {}
            return None, {"brand_attribution": f"skipped: URL does not match {b.brand_id} rules"}
        return None, {}

    def is_target(self, domain: str | None) -> bool | None:
        if not domain:
            return None
        return any(self._on(domain, t) for t in self.target_domains)


def _openai(meta: dict[str, Any], attributor: DomainAttributor, result: CitationResult) -> None:
    annotations = meta.get("url_citations")
    if annotations is None:
        return
    if not isinstance(annotations, list):
        result.warnings.append("openai url_citations is not a list; skipped")
        return
    for a in annotations:
        url = a.get("url") if isinstance(a, dict) else None
        if not url:
            result.warnings.append("openai url_citation without url; skipped")
            continue
        start, end = a.get("start_index"), a.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
            result.masked_spans.append((start, end))
        domain, path = split_url(url)
        brand_id, note = attributor.brand_for(domain, path)
        result.add(url=url, domain=domain, title=a.get("title"), brand_id=brand_id,
                   citation_source="provider_metadata", provider_format="openai_url_citation",
                   metadata={"start_index": start, "end_index": end, **note},
                   is_target_brand_domain=attributor.is_target(domain))


def _gemini(meta: dict[str, Any], attributor: DomainAttributor, result: CitationResult) -> None:
    grounding = meta.get("grounding_metadata")
    if grounding is None:
        return
    if not isinstance(grounding, dict):
        result.warnings.append("gemini grounding_metadata is not an object; skipped")
        return
    supports: dict[int, int] = {}
    for s in grounding.get("grounding_supports") or []:
        for i in (s.get("grounding_chunk_indices") or []) if isinstance(s, dict) else []:
            supports[i] = supports.get(i, 0) + 1
    for index, chunk in enumerate(grounding.get("grounding_chunks") or []):
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not isinstance(web, dict) or not web.get("uri"):
            result.warnings.append(f"gemini grounding chunk {index} has no web uri; skipped")
            continue
        uri, title = web["uri"], web.get("title")
        domain = normalize_domain(web.get("domain")) or normalize_domain(title)
        uri_host, uri_path = split_url(uri)
        is_redirect = uri_host == GEMINI_REDIRECT_HOST
        brand_id, note = attributor.brand_for(domain, None if is_redirect else uri_path)
        result.add(url=uri, domain=domain, title=title, brand_id=brand_id,
                   citation_source="provider_metadata", provider_format="gemini_grounding_chunk",
                   metadata={"chunk_index": index, "uri_is_redirect": is_redirect,
                             "supporting_segments": supports.get(index, 0), **note},
                   is_target_brand_domain=attributor.is_target(domain))


def _text_urls(text: str, attributor: DomainAttributor, result: CitationResult) -> None:
    seen = {c.url for c in result.citations}
    for m in URL_RE.finditer(text):
        if any(s <= m.start() < e for s, e in result.masked_spans):
            continue
        url = m.group(0).rstrip(TRAILING_PUNCTUATION)
        if url in seen:
            continue
        seen.add(url)
        domain, path = split_url(url)
        brand_id, note = attributor.brand_for(domain, path)
        result.add(url=url, domain=domain, title=None, brand_id=brand_id,
                   citation_source="response_text", provider_format="text_url",
                   metadata={"char_offset": m.start(), **note},
                   is_target_brand_domain=attributor.is_target(domain))


def extract_citations(provider: str, text: str, grounding_metadata: Any,
                      attributor: DomainAttributor) -> CitationResult:
    result = CitationResult()
    meta = grounding_metadata or {}
    if not isinstance(meta, dict):
        result.warnings.append("grounding_metadata is not an object; provider citations skipped")
        meta = {}
    if provider == "openai":
        _openai(meta, attributor, result)
    elif provider == "gemini":
        _gemini(meta, attributor, result)
    elif meta:
        result.warnings.append(f"no citation parser for provider {provider!r}; provider citations skipped")
    _text_urls(text, attributor, result)
    return result
