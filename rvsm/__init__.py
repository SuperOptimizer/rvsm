"""rvsm: raw CT + umbilicus in, a self-distilled recto/verso surface model out, on one machine."""
# NOT set here: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. It cut the trainer's fragmentation,
# but on Thunder's virtualised A100 (tnr-0) the trainer then hung inside backward() -- twice, all
# threads parked on futexes -- which never happened without it. Set it in the environment on a host
# where the CUDA virtual-memory APIs are real.
__version__ = "0.1.0"
