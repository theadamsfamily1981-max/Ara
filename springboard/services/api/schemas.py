"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class AnalysisRequest(BaseModel):
    """Request for analysis."""
    target_col: str
    client_name: str = "Client"
    title: Optional[str] = None


class PatternResult(BaseModel):
    """A single pattern finding."""
    type: str
    feature: str
    strength: float
    note: str


class ExperimentResult(BaseModel):
    """A suggested experiment."""
    title: str
    hypothesis: str
    variant_a: str
    variant_b: str
    metric_to_watch: str
    expected_lift: str
    confidence: str
    rationale: str


class AnalysisResponse(BaseModel):
    """Response from analysis endpoint."""
    success: bool
    report_markdown: str
    summary: str
    patterns: List[Dict[str, Any]]
    experiments: List[Dict[str, Any]]
    extras: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "report_markdown": "# Email Subject Line Analysis\n...",
                "summary": "Here's what I noticed...",
                "patterns": [
                    {
                        "type": "numeric",
                        "feature": "subject_length_chars",
                        "strength": 0.35,
                        "note": "Shorter subjects perform better",
                    }
                ],
                "experiments": [
                    {
                        "title": "Test shorter subjects",
                        "hypothesis": "Shorter subjects increase opens",
                        "variant_a": "Current length",
                        "variant_b": "Under 40 characters",
                        "metric_to_watch": "open_rate",
                        "expected_lift": "5-10%",
                        "confidence": "medium",
                        "rationale": "Strong correlation found",
                    }
                ],
            }
        }
