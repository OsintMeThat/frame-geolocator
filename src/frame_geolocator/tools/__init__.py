"""Tool catalogue for FrameGeolocator.

Each tool is a small, deterministic-where-possible, independently-testable unit with a
typed contract. Tools are grouped by role in ``docs/architecture.md`` (media, scene,
region restriction, terrain, geospatial matching, geometric fusion).
"""

from frame_geolocator.tools.classify_audio_events import (
    AudioEvent,
    ClassifyAudioConfig,
    ClassifyAudioResult,
    classify_audio_events,
)
from frame_geolocator.tools.decompose_video import (
    DecomposeConfig,
    DecomposeResult,
    SelectedFrame,
    decompose_video,
)
from frame_geolocator.tools.extract_audio import (
    ExtractAudioConfig,
    ExtractAudioResult,
    extract_audio,
)
from frame_geolocator.tools.flash_to_bang_range import (
    FlashToBangConfig,
    FlashToBangResult,
    RangedEvent,
    flash_to_bang_range,
)
from frame_geolocator.tools.identify_spoken_language import (
    Segment,
    SpokenLanguageConfig,
    SpokenLanguageResult,
    identify_spoken_language,
)
from frame_geolocator.tools.read_text_ocr import (
    OCRConfig,
    OCRResult,
    TextDetection,
    read_text_ocr,
)

__all__ = [
    "AudioEvent",
    "ClassifyAudioConfig",
    "ClassifyAudioResult",
    "classify_audio_events",
    "DecomposeConfig",
    "DecomposeResult",
    "SelectedFrame",
    "decompose_video",
    "OCRConfig",
    "OCRResult",
    "TextDetection",
    "read_text_ocr",
    "ExtractAudioConfig",
    "ExtractAudioResult",
    "extract_audio",
    "FlashToBangConfig",
    "FlashToBangResult",
    "RangedEvent",
    "flash_to_bang_range",
    "Segment",
    "SpokenLanguageConfig",
    "SpokenLanguageResult",
    "identify_spoken_language",
]
