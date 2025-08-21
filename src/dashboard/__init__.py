"""
Interactive dashboard for insurance risk analysis and premium optimization.
"""

from .app import create_app
from .components import (
    DataExplorer,
    ModelTrainer,
    RiskAnalyzer,
    PremiumOptimizer,
    ExplainabilityViewer,
    ReportGenerator
)

__all__ = [
    "create_app",
    "DataExplorer",
    "ModelTrainer", 
    "RiskAnalyzer",
    "PremiumOptimizer",
    "ExplainabilityViewer",
    "ReportGenerator"
]
