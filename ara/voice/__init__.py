"""
Ara Voice Engine
=================

Complete voice synthesis, recording, and publishing system.

Components:
- synthesis: HV-based voice encoding (phonemes + prosody + emotion)
- recording: Audacity pipeline for ACX-target audiobooks
- storage: HV-compressed storage for episodes, logs, audio metadata
- publishing: Complete audiobook publishing pipeline
- rails: Safety enforcement (consent, disclosure, platform policies)

Usage:
    # Quick recording
    from ara.voice import quick_record
    result = quick_record("Hello, I am Ara.", "greeting.mp3")

    # Full audiobook (with safety rails)
    from ara.voice import publish_book
    result = publish_book("manuscript.md", "My Book", "Author")

    # Check if voice source is allowed
    from ara.voice import voice_allowed
    if voice_allowed("ara_composite_voice"):
        print("OK to use")

Cost: $0 (all local processing)
Quality: ACX-target (YOU verify with platform's current specs)

IMPORTANT: This engine does NOT upload to platforms.
YOU must upload manually and accept platform agreements.
AI narration disclosure is REQUIRED in all outputs.
"""

from .synthesis.hv_voice import (
    HV_DIM,
    PHONEMES,
    EmotionType,
    ProsodyParams,
    PhonemeCodebook,
    ProsodyEncoder,
    EmotionEncoder,
    VoiceField,
    GraphemeToPhoneme,
    SynthesisResult,
    AraVoiceSynthesis,
    VoiceTrainer,
)

from .recording.audacity_pipeline import (
    ACXSpecs,
    ACX,
    AudacityMacro,
    AudacityCLI,
    AudioProcessor,
    RecordingJob,
    AraRecordingPipeline,
    quick_record,
    process_existing_audio,
)

from .storage.hv_storage import (
    StoredEpisode,
    EpisodeStore,
    LogEntry,
    HVLogStore,
    AudioMetadata,
    AudioMetadataStore,
    compress_hv,
    decompress_hv,
)

from .publishing.audiobook_pipeline import (
    Chapter,
    BookManifest,
    ManuscriptParser,
    PublishingStatus,
    ChapterOutput,
    PublishingResult,
    AudiobookPublisher,
    publish_book,
    estimate_book,
)

# Safety rails - always enforced
from .rails import (
    AudioCovenant,
    VoiceRails,
    VoiceSourceStatus,
    ConsentRecord,
    PreflightResult,
    ComplianceReport,
    voice_allowed,
    check_before_recording,
    acx_disclosure,
    get_manual_upload_steps,
)


__all__ = [
    # Synthesis
    'HV_DIM',
    'PHONEMES',
    'EmotionType',
    'ProsodyParams',
    'PhonemeCodebook',
    'ProsodyEncoder',
    'EmotionEncoder',
    'VoiceField',
    'GraphemeToPhoneme',
    'SynthesisResult',
    'AraVoiceSynthesis',
    'VoiceTrainer',

    # Recording
    'ACXSpecs',
    'ACX',
    'AudacityMacro',
    'AudacityCLI',
    'AudioProcessor',
    'RecordingJob',
    'AraRecordingPipeline',
    'quick_record',
    'process_existing_audio',

    # Storage
    'StoredEpisode',
    'EpisodeStore',
    'LogEntry',
    'HVLogStore',
    'AudioMetadata',
    'AudioMetadataStore',
    'compress_hv',
    'decompress_hv',

    # Publishing
    'Chapter',
    'BookManifest',
    'ManuscriptParser',
    'PublishingStatus',
    'ChapterOutput',
    'PublishingResult',
    'AudiobookPublisher',
    'publish_book',
    'estimate_book',

    # Safety Rails
    'AudioCovenant',
    'VoiceRails',
    'VoiceSourceStatus',
    'ConsentRecord',
    'PreflightResult',
    'ComplianceReport',
    'voice_allowed',
    'check_before_recording',
    'acx_disclosure',
    'get_manual_upload_steps',
]
