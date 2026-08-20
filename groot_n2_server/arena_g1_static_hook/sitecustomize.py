"""Register the Arena-G1-Static embodiment config under the GR00T 2 runtime.

Same process-local shim as the 405 hook: the embodiment config is written
against the pre-rename ``gr00t`` import paths and the former
``register_modality_config(config, embodiment_tag=...)`` signature. This maps
those onto the current ``groot`` runtime, then runs the config so
``new_embodiment`` is registered. It is intentionally not installed into the
package -- it only takes effect when this directory is on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import runpy
import sys
import types

from groot.core.data import types as groot_types
from groot.core.data.schema import embodiment_tags
from groot.core.training.configs.data import embodiment_configs


def _register_legacy(config: dict, *, embodiment_tag: object) -> None:
    embodiment_configs.register_modality_config(
        getattr(embodiment_tag, "value", embodiment_tag), config
    )


legacy_root = types.ModuleType("gr00t")
legacy_configs = types.ModuleType("gr00t.configs")
legacy_configs_data = types.ModuleType("gr00t.configs.data")
legacy_embodiments = types.ModuleType("gr00t.configs.data.embodiment_configs")
legacy_embodiments.MODALITY_CONFIGS = embodiment_configs.MODALITY_CONFIGS
legacy_embodiments.register_modality_config = _register_legacy
legacy_data = types.ModuleType("gr00t.data")

sys.modules.update(
    {
        "gr00t": legacy_root,
        "gr00t.configs": legacy_configs,
        "gr00t.configs.data": legacy_configs_data,
        "gr00t.configs.data.embodiment_configs": legacy_embodiments,
        "gr00t.data": legacy_data,
        "gr00t.data.embodiment_tags": embodiment_tags,
        "gr00t.data.types": groot_types,
    }
)

_CONFIG = os.path.join(os.path.dirname(__file__), "arena_g1_static_config.py")
runpy.run_path(_CONFIG)
