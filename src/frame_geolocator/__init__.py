"""FrameGeolocator — agentic OSINT geolocation of images and videos.

The package is organized as a catalogue of small, independently-testable tools (see
``frame_geolocator.tools``) that an orchestrating LLM calls and interprets. The hard
numeric work lives in the tools; the LLM only routes and reasons.

See ``docs/architecture.md`` for the full design.
"""

__version__ = "0.0.1"
