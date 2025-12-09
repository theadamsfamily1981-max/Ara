"""
Ara Voice Engine
=================

Complete voice synthesis, recording, and publishing system.

Components:
- synthesis: HV-based voice encoding (phonemes + prosody + emotion)
- recording: Audacity pipeline for ACX-compliant audiobooks
- storage: HV-compressed storage for episodes, logs, audio metadata
- publishing: Complete audiobook publishing pipeline

Usage:
    # Quick recording
    from ara.voice import quick_record
    result = quick_record("Hello, I am Ara.", "greeting.mp3")

    # Full audiobook
    from ara.voice import publish_book
    result = publish_book("manuscript.md", "My Book", "Author")

Cost: $0 (all local processing)
Quality: ACX perfect (-23dB RMS, 192kbps MP3)
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
]
