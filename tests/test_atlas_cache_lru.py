"""Regression tests for the atlas volume LRU cache in DeepSliceAppState.

The cache used to be an unbounded dict that could hold both the mouse
(~74 MB) and rat (~200+ MB) 3D volumes at once. It is now an
OrderedDict capped at DeepSliceAppState._ATLAS_CACHE_MAX_ENTRIES so
users who toggle species don't keep both resident."""
from __future__ import annotations

import pathlib
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.gui.state import DeepSliceAppState


def test_atlas_cache_uses_ordered_dict_by_default():
    state = DeepSliceAppState(species="mouse")
    assert isinstance(state._atlas_cache, OrderedDict)


def test_atlas_cache_evicts_oldest_when_max_reached():
    state = DeepSliceAppState(species="mouse")
    # Simulate two volumes being cached; the cap is 1 by default so the
    # first entry must be evicted when the second is inserted.
    state._atlas_cache["mouse:nissl"] = np.zeros((4, 4, 4), dtype=np.float32)
    state._atlas_cache["rat:MRI"] = np.ones((4, 4, 4), dtype=np.float32)
    # Manually apply the same eviction the loader performs.
    while len(state._atlas_cache) > state._ATLAS_CACHE_MAX_ENTRIES:
        state._atlas_cache.popitem(last=False)
    assert list(state._atlas_cache.keys()) == ["rat:MRI"]


def test_set_species_clear_preserves_ordered_dict_type():
    """Changing species used to reset the cache to a plain dict, which broke
    the subsequent LRU eviction path (dict has no move_to_end/popitem(last=)).
    Now the cache is cleared in place."""
    state = DeepSliceAppState(species="mouse")
    original_cache_id = id(state._atlas_cache)
    state._atlas_cache["mouse:nissl"] = np.zeros((2, 2, 2), dtype=np.float32)
    state.set_species("rat")
    assert isinstance(state._atlas_cache, OrderedDict)
    assert id(state._atlas_cache) == original_cache_id
    assert len(state._atlas_cache) == 0
