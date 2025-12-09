"""
Ara SaaS API
=============

FastAPI application for the Ara Memory Foundry.

Endpoints:
- /api/packs       - List/fetch Memory Packs
- /api/route       - Blind router for encrypted envelopes
- /api/memory/sync - Sync encrypted memory records
- /api/health      - Health check
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .wire_protocol import (
    OuterEnvelope,
    ResponseEnvelope,
    MessageType,
)
from .blind_router import BlindRouter, create_router_with_stubs

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models (for FastAPI validation)
# =============================================================================

if HAS_FASTAPI:

    class PackMetadata(BaseModel):
        pack_id: str
        version: str
        domain: str
        capabilities: List[str]
        size_bytes: Optional[int] = None

    class PackContent(BaseModel):
        manifest: Dict[str, Any]
        content_encrypted: Optional[str] = None  # base64 for encrypted packs
        content: Optional[List[Dict[str, Any]]] = None  # plaintext for dev

    class EnvelopeRequest(BaseModel):
        v: int = 1
        envelope_id: str
        route_to: str
        message_type: str
        priority: float = 0.5
        client_hint: str = ""
        service_hint: str = ""
        payload_e2e: Optional[Dict[str, Any]] = None

    class MemorySyncRequest(BaseModel):
        user_id: str
        device_id: str
        records: List[Dict[str, Any]]


# =============================================================================
# Application Factory
# =============================================================================

def create_app(
    packs_dir: Optional[Path] = None,
    memory_db_path: Optional[Path] = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        packs_dir: Directory containing Memory Packs
        memory_db_path: Path to memory fabric database
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Ara Memory Foundry",
        description="Memory Packs SaaS - Encrypted knowledge for any LLM",
        version="0.1.0",
    )

    # Initialize components
    packs_dir = packs_dir or Path("memory_packs")
    memory_db_path = memory_db_path or Path("data/memory_fabric.sqlite")

    # Blind router
    router = create_router_with_stubs()

    # Memory fabric (lazy init)
    memory_fabric = None

    # =========================================================================
    # Pack Endpoints
    # =========================================================================

    @app.get("/api/packs", response_model=List[PackMetadata])
    async def list_packs(domain: Optional[str] = None):
        """List available Memory Packs."""
        packs = _scan_packs(packs_dir)

        if domain:
            packs = [p for p in packs if p.get("domain") == domain]

        return packs

    @app.get("/api/packs/{pack_id}/{version}")
    async def get_pack(pack_id: str, version: str):
        """
        Fetch a specific Memory Pack.

        Returns manifest + encrypted content.
        """
        pack_path = packs_dir / pack_id / version

        # Check if pack exists
        manifest_path = pack_path / "manifest.json"
        if not manifest_path.exists():
            # Return stub pack for dev
            return _stub_pack(pack_id, version)

        # Load manifest
        manifest = json.loads(manifest_path.read_text())

        # Load content
        content_path = pack_path / "content.jsonl"
        content = []
        if content_path.exists():
            with open(content_path) as f:
                for line in f:
                    if line.strip():
                        content.append(json.loads(line))

        return {
            "manifest": manifest,
            "content": content,  # In production, this would be encrypted
        }

    # =========================================================================
    # Router Endpoints
    # =========================================================================

    @app.post("/api/route")
    async def route_envelope(request: EnvelopeRequest):
        """
        Route an encrypted envelope to a service.

        The router does NOT decrypt payload_e2e - it just forwards
        based on route_to and message_type.
        """
        from .wire_protocol import InnerPayload

        # Convert to internal envelope
        payload = None
        if request.payload_e2e:
            payload = InnerPayload.from_dict(request.payload_e2e)

        envelope = OuterEnvelope(
            envelope_id=request.envelope_id,
            route_to=request.route_to,
            message_type=request.message_type,
            priority=request.priority,
            client_hint=request.client_hint,
            service_hint=request.service_hint,
            payload_e2e=payload,
        )

        # Route
        response = router.route(envelope)

        return response.to_dict()

    @app.get("/api/route/services")
    async def list_services():
        """List registered services."""
        return {
            "services": router.list_services(),
            "metrics": router.get_metrics(),
        }

    # =========================================================================
    # Memory Endpoints
    # =========================================================================

    @app.post("/api/memory/sync")
    async def sync_memory(request: MemorySyncRequest):
        """
        Sync encrypted memory records from client.

        The server stores encrypted blobs - it cannot read the content.
        """
        nonlocal memory_fabric

        # Lazy init
        if memory_fabric is None:
            from ..storage.memory_fabric import MemoryFabric
            memory_fabric = MemoryFabric(memory_db_path)

        # Store records
        from .wire_protocol import MemoryRecord

        count = 0
        for rec_data in request.records:
            record = MemoryRecord(
                record_id=rec_data.get("record_id", ""),
                kind=rec_data.get("kind", "episode"),
                scope=rec_data.get("scope", "private"),
                tags=rec_data.get("tags", []),
                dek_wrapped=rec_data.get("dek_wrapped", ""),
                ciphertext=rec_data.get("ciphertext", ""),
                metadata=rec_data.get("metadata", {}),
            )
            memory_fabric.store_record(
                user_id=request.user_id,
                device_id=request.device_id,
                record=record,
            )
            count += 1

        return {"status": "ok", "records_synced": count}

    @app.get("/api/memory/list")
    async def list_memory(
        user_id: str,
        kind: Optional[str] = None,
        limit: int = 100,
    ):
        """
        List memory record metadata for a user.

        Returns metadata ONLY - not encrypted content.
        """
        nonlocal memory_fabric

        if memory_fabric is None:
            from ..storage.memory_fabric import MemoryFabric
            memory_fabric = MemoryFabric(memory_db_path)

        records = memory_fabric.list_records(
            user_id=user_id,
            kind=kind,
            limit=limit,
        )

        return {"records": records}

    # =========================================================================
    # Health & Info
    # =========================================================================

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "0.1.0",
        }

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Ara Memory Foundry",
            "version": "0.1.0",
            "endpoints": {
                "packs": "/api/packs",
                "route": "/api/route",
                "memory": "/api/memory/sync",
                "health": "/api/health",
            },
        }

    return app


# =============================================================================
# Helpers
# =============================================================================

def _scan_packs(packs_dir: Path) -> List[Dict[str, Any]]:
    """Scan directory for available packs."""
    packs = []

    if not packs_dir.exists():
        # Return stub packs for dev
        return [
            {
                "pack_id": "indie_publishing",
                "version": "1",
                "domain": "publishing",
                "capabilities": ["kdp_workflow", "blurb_writing", "launch_checklist"],
                "size_bytes": 10240,
            },
            {
                "pack_id": "coding_patterns",
                "version": "1",
                "domain": "coding",
                "capabilities": ["python_best_practices", "testing_patterns"],
                "size_bytes": 20480,
            },
        ]

    for pack_path in packs_dir.iterdir():
        if not pack_path.is_dir():
            continue

        for version_path in pack_path.iterdir():
            if not version_path.is_dir():
                continue

            manifest_path = version_path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                packs.append({
                    "pack_id": manifest.get("pack_id", pack_path.name),
                    "version": manifest.get("version", version_path.name),
                    "domain": manifest.get("domain", ""),
                    "capabilities": manifest.get("capabilities", []),
                })

    return packs


def _stub_pack(pack_id: str, version: str) -> Dict[str, Any]:
    """Return a stub pack for development."""
    if pack_id == "indie_publishing":
        return {
            "manifest": {
                "pack_id": "indie_publishing",
                "version": version,
                "domain": "publishing",
                "capabilities": ["kdp_workflow", "blurb_writing", "launch_checklist"],
                "created_at": "2025-12-09T00:00:00Z",
                "encryption": {"algorithm": "none"},
            },
            "content": [
                {
                    "id": "ep001",
                    "type": "checklist",
                    "tags": ["kdp", "setup"],
                    "text": "KDP Account Setup Checklist:\n1. Go to kdp.amazon.com\n2. Sign in with your Amazon account\n3. Complete tax interview (W-9 for US)\n4. Set up payment method\n5. Verify your email address",
                },
                {
                    "id": "ep002",
                    "type": "procedure",
                    "tags": ["launch", "marketing"],
                    "text": "Book Launch Sequence:\n- T-30: Announce to email list, cover reveal\n- T-21: Send ARCs to top reviewers\n- T-14: Set up pre-order on Amazon\n- T-7: Final marketing push, countdown\n- T-1: Last email, social media blitz\n- Launch: Go live, celebrate!",
                },
                {
                    "id": "ep003",
                    "type": "prompt_example",
                    "tags": ["blurb", "romance"],
                    "text": "Blurb Writing Prompt:\nWrite a compelling 150-word blurb for [TITLE]. Include:\n- Hook in first line\n- Stakes and conflict\n- Emotional promise\n- Call to action\n\nExample structure:\n[Hook] - grab attention\n[Setup] - introduce protagonist\n[Conflict] - what's at stake\n[Twist] - unexpected element\n[Promise] - emotional payoff hint",
                },
            ],
        }

    # Generic stub
    return {
        "manifest": {
            "pack_id": pack_id,
            "version": version,
            "domain": "general",
            "capabilities": [],
            "created_at": "2025-12-09T00:00:00Z",
        },
        "content": [],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Memory Foundry API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--packs-dir", type=Path, default=Path("memory_packs"))
    parser.add_argument("--memory-db", type=Path, default=Path("data/memory_fabric.sqlite"))

    args = parser.parse_args()

    app = create_app(
        packs_dir=args.packs_dir,
        memory_db_path=args.memory_db,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
