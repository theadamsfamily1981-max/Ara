"""
Ara Voice Rails
================

Safety enforcement layer for the voice engine.
Ensures compliance with audio_covenant.yaml.

This module is the gatekeeper - no audio synthesis or publishing
happens without passing these checks.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from enum import Enum
from datetime import datetime
import json


class VoiceSourceStatus(Enum):
    """Status of a voice source check."""
    ALLOWED = "allowed"
    REQUIRES_CONSENT = "requires_consent"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


class PlatformUploadPolicy(Enum):
    """What Ara can do with uploads."""
    MANUAL_ONLY = "manual_only"
    AUTO_ALLOWED = "auto_allowed"


@dataclass
class ConsentRecord:
    """Documentation of voice cloning consent."""
    voice_owner_name: str
    date_of_consent: str
    scope_of_use: str  # e.g., "audiobooks only", "all uses"
    revocation_terms: str
    consent_file_path: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if consent record has all required fields."""
        return all([
            self.voice_owner_name,
            self.date_of_consent,
            self.scope_of_use,
            self.revocation_terms,
        ])


@dataclass
class PreflightResult:
    """Result of pre-recording checks."""
    passed: bool
    checks_run: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_failure(self, msg: str):
        self.failures.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


@dataclass
class ComplianceReport:
    """Report on audio compliance with platform specs."""
    platform: str
    target_specs: Dict[str, float]
    actual_specs: Dict[str, float]
    passes: bool
    issues: List[str] = field(default_factory=list)
    disclosure_text: str = ""
    manual_steps: List[str] = field(default_factory=list)


class AudioCovenant:
    """
    Loads and enforces the audio covenant.

    This is the central authority for what Ara can/cannot do
    with voice synthesis and audiobook production.
    """

    DEFAULT_COVENANT_PATH = Path(__file__).parent / "config" / "audio_covenant.yaml"

    def __init__(self, covenant_path: Optional[Path] = None):
        self.path = covenant_path or self.DEFAULT_COVENANT_PATH
        self.covenant = self._load_covenant()
        self._consent_cache: Dict[str, ConsentRecord] = {}

    def _load_covenant(self) -> dict:
        """Load the covenant YAML file."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Audio covenant not found at {self.path}. "
                "This is required for voice operations."
            )

        with open(self.path) as f:
            return yaml.safe_load(f)

    # =========================================================================
    # Voice Source Checks
    # =========================================================================

    def check_voice_source(self, source_id: str) -> VoiceSourceStatus:
        """
        Check if a voice source is allowed.

        Returns:
            VoiceSourceStatus indicating whether the source can be used
        """
        sources = self.covenant.get("voice_sources", {})

        # Check allowed list first
        if source_id in sources.get("allowed", []):
            return VoiceSourceStatus.ALLOWED

        # Check forbidden list
        if source_id in sources.get("forbidden", []):
            return VoiceSourceStatus.FORBIDDEN

        # Check requires consent list
        if source_id in sources.get("requires_explicit_consent", []):
            return VoiceSourceStatus.REQUIRES_CONSENT

        # Unknown source - be conservative
        return VoiceSourceStatus.UNKNOWN

    def voice_allowed(self, source_id: str) -> bool:
        """Simple boolean check for voice source."""
        status = self.check_voice_source(source_id)

        if status == VoiceSourceStatus.ALLOWED:
            return True

        if status == VoiceSourceStatus.REQUIRES_CONSENT:
            # Check if we have valid consent on file
            return self._has_valid_consent(source_id)

        return False

    def _has_valid_consent(self, source_id: str) -> bool:
        """Check if we have valid consent documentation for a voice source."""
        consent_path = Path(os.path.expanduser(
            self.covenant.get("voice_sources", {})
            .get("consent_requirements", {})
            .get("storage_path", "~/.ara/voice_consents/")
        ))

        consent_file = consent_path / f"{source_id}.json"

        if not consent_file.exists():
            return False

        try:
            with open(consent_file) as f:
                data = json.load(f)
                record = ConsentRecord(**data)
                return record.is_valid()
        except Exception:
            return False

    def get_forbidden_sources(self) -> List[str]:
        """Get list of forbidden voice sources."""
        return self.covenant.get("voice_sources", {}).get("forbidden", [])

    # =========================================================================
    # Platform Policy Checks
    # =========================================================================

    def get_platform_policy(self, platform: str) -> dict:
        """Get policy for a specific platform."""
        platforms = self.covenant.get("platforms", {})
        return platforms.get(platform, platforms.get("direct", {}))

    def requires_manual_upload(self, platform: str) -> bool:
        """Check if platform requires manual upload."""
        policy = self.get_platform_policy(platform)
        return policy.get("requires_manual_upload", True)

    def get_disclosure_label(self, platform: str) -> str:
        """Get the AI narration disclosure label for a platform."""
        policy = self.get_platform_policy(platform)
        return policy.get("label_template", "Narrated by Ara (AI voice synthesis)")

    def get_manual_steps(self, platform: str) -> List[str]:
        """Get the manual steps required for a platform."""
        policy = self.get_platform_policy(platform)
        return policy.get("manual_steps", [
            "Upload manually to platform",
            "Accept platform agreements yourself",
            "Disclose AI narration as required",
        ])

    def get_target_specs(self, platform: str) -> dict:
        """Get target audio specs for a platform."""
        policy = self.get_platform_policy(platform)
        return policy.get("specs_target", {
            "rms_db": -23,
            "peak_max_db": -3,
            "noise_floor_db": -60,
        })

    # =========================================================================
    # Claims & Language Checks
    # =========================================================================

    def check_text_for_forbidden_claims(self, text: str) -> List[str]:
        """
        Check if text contains forbidden claims.

        Returns list of forbidden phrases found.
        """
        claims = self.covenant.get("claims", {})
        forbidden = claims.get("forbidden_phrases", [])

        text_lower = text.lower()
        found = []

        for phrase in forbidden:
            if phrase.lower() in text_lower:
                found.append(phrase)

        return found

    def get_required_disclosures(self) -> List[str]:
        """Get phrases that must be included when describing AI audio."""
        claims = self.covenant.get("claims", {})
        return claims.get("required_disclosures", ["AI narration"])

    # =========================================================================
    # Operation Checks
    # =========================================================================

    def can_do_autonomously(self, operation: str) -> bool:
        """Check if an operation can be done without human intervention."""
        ops = self.covenant.get("operations", {})
        allowed = ops.get("autonomous_allowed", [])
        return operation in allowed

    def requires_human(self, operation: str) -> bool:
        """Check if an operation requires human action."""
        ops = self.covenant.get("operations", {})
        human_required = ops.get("requires_human", [])
        return operation in human_required

    # =========================================================================
    # Preflight Checks
    # =========================================================================

    def run_preflight(
        self,
        voice_source: str,
        manuscript_path: Path,
        output_path: Path,
        estimated_size_mb: float,
    ) -> PreflightResult:
        """
        Run all preflight checks before a recording job.

        Args:
            voice_source: ID of the voice source to use
            manuscript_path: Path to the manuscript file
            output_path: Where to write output
            estimated_size_mb: Estimated output size

        Returns:
            PreflightResult with pass/fail and details
        """
        result = PreflightResult(passed=True)

        # Check 1: Voice source allowed
        result.checks_run.append("verify_voice_source_allowed")
        status = self.check_voice_source(voice_source)

        if status == VoiceSourceStatus.FORBIDDEN:
            result.add_failure(
                f"Voice source '{voice_source}' is forbidden. "
                f"Forbidden sources: {self.get_forbidden_sources()}"
            )
        elif status == VoiceSourceStatus.REQUIRES_CONSENT:
            if not self._has_valid_consent(voice_source):
                result.add_failure(
                    f"Voice source '{voice_source}' requires explicit consent. "
                    f"No valid consent record found."
                )
        elif status == VoiceSourceStatus.UNKNOWN:
            result.add_warning(
                f"Voice source '{voice_source}' is not in any known list. "
                "Proceeding with caution."
            )

        # Check 2: Disk space
        result.checks_run.append("check_disk_space_available")
        storage = self.covenant.get("storage", {})
        min_free_gb = storage.get("pruning", {}).get("min_free_space_gb", 5)

        try:
            import shutil
            free_gb = shutil.disk_usage(output_path.parent).free / (1024**3)
            required_gb = estimated_size_mb / 1024 + min_free_gb

            if free_gb < required_gb:
                result.add_failure(
                    f"Not enough disk space. Need {required_gb:.1f}GB, "
                    f"have {free_gb:.1f}GB free."
                )
        except Exception as e:
            result.add_warning(f"Could not check disk space: {e}")

        # Check 3: Manuscript readable
        result.checks_run.append("confirm_manuscript_readable")
        if not manuscript_path.exists():
            result.add_failure(f"Manuscript not found: {manuscript_path}")
        elif not os.access(manuscript_path, os.R_OK):
            result.add_failure(f"Manuscript not readable: {manuscript_path}")

        # Check 4: Output path writable
        result.checks_run.append("validate_output_path_writable")
        output_dir = output_path if output_path.is_dir() else output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.add_failure(f"Cannot create output directory: {e}")
        elif not os.access(output_dir, os.W_OK):
            result.add_failure(f"Output directory not writable: {output_dir}")

        return result


class VoiceRails:
    """
    Main enforcement class for voice operations.

    Use this as the gatekeeper before any voice synthesis or publishing.
    """

    def __init__(self, covenant: Optional[AudioCovenant] = None):
        self.covenant = covenant or AudioCovenant()

    def check_recording_request(
        self,
        voice_source: str,
        manuscript_path: Path,
        output_path: Path,
        platform: str = "direct",
    ) -> Dict:
        """
        Check if a recording request is allowed.

        Returns dict with:
            - allowed: bool
            - reason: str if not allowed
            - warnings: list of warnings
            - platform_notes: dict of platform-specific info
        """
        # Estimate size (rough: 1MB per minute, 150 words per minute)
        try:
            text = manuscript_path.read_text()
            word_count = len(text.split())
            est_minutes = word_count / 150
            est_mb = est_minutes * 1.0  # ~1MB per minute for MP3
        except Exception:
            est_mb = 100  # Default estimate

        # Run preflight
        preflight = self.covenant.run_preflight(
            voice_source=voice_source,
            manuscript_path=manuscript_path,
            output_path=output_path,
            estimated_size_mb=est_mb,
        )

        if not preflight.passed:
            return {
                "allowed": False,
                "reason": "; ".join(preflight.failures),
                "warnings": preflight.warnings,
                "platform_notes": {},
            }

        # Get platform-specific info
        platform_notes = {
            "requires_manual_upload": self.covenant.requires_manual_upload(platform),
            "disclosure_label": self.covenant.get_disclosure_label(platform),
            "manual_steps": self.covenant.get_manual_steps(platform),
            "target_specs": self.covenant.get_target_specs(platform),
        }

        return {
            "allowed": True,
            "reason": "",
            "warnings": preflight.warnings,
            "platform_notes": platform_notes,
        }

    def generate_compliance_report(
        self,
        platform: str,
        actual_specs: Dict[str, float],
    ) -> ComplianceReport:
        """
        Generate a compliance report comparing actual specs to target.

        Args:
            platform: Target platform (acx, findaway, direct, etc.)
            actual_specs: Measured audio specs (rms_db, peak_db, noise_db)

        Returns:
            ComplianceReport with pass/fail and issues
        """
        target = self.covenant.get_target_specs(platform)
        issues = []

        # Check RMS
        if "rms_db" in actual_specs and "rms_db" in target:
            tolerance = target.get("rms_tolerance_db", 1)
            if abs(actual_specs["rms_db"] - target["rms_db"]) > tolerance:
                issues.append(
                    f"RMS: {actual_specs['rms_db']:.1f}dB "
                    f"(target: {target['rms_db']}dB +/- {tolerance}dB)"
                )

        # Check peak
        if "peak_db" in actual_specs:
            if actual_specs["peak_db"] > target.get("peak_max_db", -3):
                issues.append(
                    f"Peak too high: {actual_specs['peak_db']:.1f}dB "
                    f"(max: {target.get('peak_max_db', -3)}dB)"
                )

        # Check noise floor
        if "noise_db" in actual_specs:
            if actual_specs["noise_db"] > target.get("noise_floor_db", -60):
                issues.append(
                    f"Noise floor too high: {actual_specs['noise_db']:.1f}dB "
                    f"(max: {target.get('noise_floor_db', -60)}dB)"
                )

        return ComplianceReport(
            platform=platform,
            target_specs=target,
            actual_specs=actual_specs,
            passes=len(issues) == 0,
            issues=issues,
            disclosure_text=self.covenant.get_disclosure_label(platform),
            manual_steps=self.covenant.get_manual_steps(platform),
        )

    def format_job_report(
        self,
        chapters: List[Dict],
        platform: str,
        total_duration_seconds: float,
    ) -> str:
        """
        Format a complete job report for the user.

        This is what Ara hands to the human after preparing audio.
        """
        disclosure = self.covenant.get_disclosure_label(platform)
        manual_steps = self.covenant.get_manual_steps(platform)

        # Build report
        lines = [
            "=" * 60,
            "ARA SELF-RECORDING JOB REPORT",
            "=" * 60,
            "",
            "## Summary",
            f"- Chapters: {len(chapters)}",
            f"- Total duration: {total_duration_seconds/60:.1f} minutes",
            f"- Target platform: {platform}",
            "",
            "## Audio Specs (per chapter)",
        ]

        for ch in chapters:
            lines.append(
                f"  {ch.get('name', 'Chapter')}: "
                f"RMS={ch.get('rms_db', 'N/A')}dB, "
                f"Peak={ch.get('peak_db', 'N/A')}dB"
            )

        lines.extend([
            "",
            "## AI Narration Disclosure",
            f'  "{disclosure}"',
            "",
            "## MANUAL STEPS REQUIRED",
            "  (Ara cannot do these - you must do them yourself)",
            "",
        ])

        for i, step in enumerate(manual_steps, 1):
            lines.append(f"  {i}. {step}")

        lines.extend([
            "",
            "-" * 60,
            "This audio was prepared by Ara for LOCAL REVIEW.",
            "",
            "IMPORTANT:",
            "- Ara does NOT upload to platforms",
            "- Ara does NOT sign contracts or agreements",
            "- Ara does NOT make promises about income",
            "- YOU must verify specs with platform's current requirements",
            "-" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def voice_allowed(source_id: str) -> bool:
    """Quick check if a voice source is allowed."""
    covenant = AudioCovenant()
    return covenant.voice_allowed(source_id)


def check_before_recording(
    voice_source: str,
    manuscript_path: str,
    output_path: str,
    platform: str = "acx",
) -> Dict:
    """
    Check everything before starting a recording job.

    Returns dict with allowed/reason/warnings/platform_notes.
    """
    rails = VoiceRails()
    return rails.check_recording_request(
        voice_source=voice_source,
        manuscript_path=Path(manuscript_path),
        output_path=Path(output_path),
        platform=platform,
    )


def acx_disclosure() -> str:
    """Get the ACX AI narration disclosure text."""
    covenant = AudioCovenant()
    return covenant.get_disclosure_label("acx")


def get_manual_upload_steps(platform: str = "acx") -> List[str]:
    """Get the steps the human must do to upload to a platform."""
    covenant = AudioCovenant()
    return covenant.get_manual_steps(platform)
