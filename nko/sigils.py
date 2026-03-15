"""nko.sigils — N'Ko Sound Sigils public API

Thin wrapper around the tools/sound-sigils implementation.
Provides the SoundSigils engine, definitions, and models
as part of the main nko package.

Usage::

    from nko.sigils import SoundSigils, SIGIL_DEFINITIONS

    ss = SoundSigils()
    ss.play("ߛ")                    # play by character
    ss.play("stabilization")        # play by name
    print(ss.list_sigils())          # formatted table
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def _bootstrap_sound_sigils() -> bool:
    """Register tools/sound-sigils as the 'sound_sigils' package.

    The directory is named ``sound-sigils`` (with a hyphen) but internal
    imports reference ``sound_sigils`` (underscore).  We use importlib to
    bridge the gap without requiring installation or symlinks.
    """
    if "sound_sigils" in sys.modules:
        return True

    pkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "sound-sigils",
    )
    init_path = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init_path):
        return False

    spec = importlib.util.spec_from_file_location(
        "sound_sigils",
        init_path,
        submodule_search_locations=[pkg_dir],
    )
    if spec is None or spec.loader is None:
        return False

    mod = importlib.util.module_from_spec(spec)
    sys.modules["sound_sigils"] = mod
    spec.loader.exec_module(mod)
    return True


try:
    _bootstrap_sound_sigils()
    from sound_sigils.engine import SoundSigils
    from sound_sigils.models import SigilDefinition, SigilSound
    from sound_sigils.definitions import (
        SIGIL_DEFINITIONS,
        get_definition_by_char,
        get_definition_by_name,
    )

    __all__ = [
        "SoundSigils",
        "SigilDefinition",
        "SigilSound",
        "SIGIL_DEFINITIONS",
        "get_definition_by_char",
        "get_definition_by_name",
    ]
except ImportError:
    # Graceful degradation — sigils available as static definitions only
    SoundSigils = None  # type: ignore[assignment, misc]
    SigilDefinition = None  # type: ignore[assignment, misc]
    SigilSound = None  # type: ignore[assignment, misc]
    SIGIL_DEFINITIONS = []  # type: ignore[assignment]

    def get_definition_by_char(char: str):  # type: ignore[misc]
        return None

    def get_definition_by_name(name: str):  # type: ignore[misc]
        return None

    __all__ = [
        "SoundSigils",
        "SigilDefinition",
        "SigilSound",
        "SIGIL_DEFINITIONS",
        "get_definition_by_char",
        "get_definition_by_name",
    ]
