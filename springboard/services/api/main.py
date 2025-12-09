"""
Ara Springboard API - FastAPI application.

Run with:
    uvicorn services.api.main:app --reload
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
from typing import Optional

from .schemas import AnalysisRequest, AnalysisResponse

app = FastAPI(
    title="Ara Springboard API",
    description="Pattern analysis for small data",
    version="0.1.0",
)

# CORS - adjust for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "name": "Ara Springboard API",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": [
            "/analyze/emails",
            "/analyze/pricing",
            "/analyze/kpis",
        ],
    }


@app.post("/analyze/emails", response_model=AnalysisResponse)
async def analyze_emails(
    file: UploadFile,
    target_col: str = "open_rate",
    client_name: str = "Client",
):
    """
    Analyze email campaign data for subject line patterns.

    Upload a CSV with columns including:
    - subject: The email subject line
    - open_rate: Open rate (or other target metric)
    - Optionally: send_date, click_rate, etc.
    """
    from ara_consult.workflows.email_subjects import analyze_email_subjects

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_email_subjects(
            tmp_path,
            target_col=target_col,
            client_name=client_name,
        )
        return AnalysisResponse(
            success=True,
            report_markdown=result["report_markdown"],
            summary=result["summary"],
            patterns=result["patterns"]["top_patterns"],
            experiments=result["experiments"],
            extras={"subject_suggestions": result["subject_suggestions"]},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/analyze/pricing", response_model=AnalysisResponse)
async def analyze_pricing(
    file: UploadFile,
    target_col: str = "conversion_rate",
    price_col: str = "price",
    client_name: str = "Client",
):
    """
    Analyze pricing data for patterns.

    Upload a CSV with columns including:
    - price: The price point
    - conversion_rate: Conversion rate (or other target metric)
    """
    from ara_consult.workflows.pricing import analyze_pricing as run_analysis

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_analysis(
            tmp_path,
            target_col=target_col,
            price_col=price_col,
            client_name=client_name,
        )
        return AnalysisResponse(
            success=True,
            report_markdown=result["report_markdown"],
            summary=result["summary"],
            patterns=result["patterns"]["top_patterns"],
            experiments=result["experiments"],
            extras={"pricing_suggestions": result["pricing_suggestions"]},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/analyze/kpis", response_model=AnalysisResponse)
async def analyze_kpis(
    file: UploadFile,
    target_col: str,
    client_name: str = "Client",
    title: Optional[str] = None,
):
    """
    Analyze any KPI data for patterns.

    Upload a CSV with:
    - A target metric column (specified by target_col)
    - Various feature columns
    """
    from ara_consult.workflows.generic_kpis import analyze_kpis as run_analysis

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_analysis(
            tmp_path,
            target_col=target_col,
            client_name=client_name,
            title=title,
        )
        return AnalysisResponse(
            success=True,
            report_markdown=result["report_markdown"],
            summary=result["summary"],
            patterns=result["patterns"]["top_patterns"],
            experiments=result["experiments"],
            extras={
                "segments": result["segments"],
                "data_overview": result["data_overview"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
