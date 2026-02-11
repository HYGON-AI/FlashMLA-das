__version__ = "1.0.0"

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache_qkvfp8,
    flash_mla_with_kvcache_kvfp8
)

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache_qkvfp8",
    "flash_mla_with_kvcache_kvfp8"
]
