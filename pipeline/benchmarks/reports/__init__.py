"""
Report generation for N'Ko and Manding language benchmark results.
"""

from .generator import ReportGenerator
from .manding_report import (
    MandingReportGenerator,
    ModelBenchmarkScore,
    LanguagePairScore,
    CurriculumScore,
    generate_manding_benchmark_report,
)

__all__ = [
    "ReportGenerator",
    "MandingReportGenerator",
    "ModelBenchmarkScore",
    "LanguagePairScore",
    "CurriculumScore",
    "generate_manding_benchmark_report",
]

