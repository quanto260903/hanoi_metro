"""
Module Ingestion Pipeline cho Hệ thống Hanoi Metro AI Agent.
Phân loại trang đa luồng (DIRECT_TEXT, STANDARD_OCR, VLM) và tổng hợp Markdown.
"""

from .classifier import PageClassifier, PageType, PageClassificationResult
from .parsers import DirectTextParser, StandardOCRParser, VLMParser, create_parser

__all__ = [
    "PageClassifier",
    "PageType",
    "PageClassificationResult",
    "DirectTextParser",
    "StandardOCRParser",
    "VLMParser",
    "create_parser",
    "IngestionPipeline",
]

def __getattr__(name: str):
    if name == "IngestionPipeline":
        from .pipeline import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

