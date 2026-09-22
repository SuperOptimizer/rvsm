"""rvsm: raw CT + umbilicus in, a self-distilled recto/verso surface model out, on one machine."""
import os as _os

# Set before anything initialises CUDA (importing torch does not): the trainer's 256^3 backward
# allocates multi-GB buffers of varying shapes, and with the default allocator ~14 GB of a 61 GB budget
# sat reserved-but-unusable when it OOMed on tnr-0. Expandable segments map memory in place instead.
# A caller that set its own PYTORCH_CUDA_ALLOC_CONF keeps it.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
