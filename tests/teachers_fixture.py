"""A teacher small enough to run on a laptop CPU, registered through the real door.

The two real teachers are 100-400 MB checkpoints on another machine, so every test that needs "a
teacher" builds this one instead: a two-stage `VesuviusUNet` with a `surface` task decoder, saved in
villa's own checkpoint shape (`{"model": state_dict}`) and registered as `TEACHERS["fake"]` with a 32^3
patch. It goes through `load_teacher` -- strict state-dict parity, architecture inferred from shapes --
exactly as `recto` does, so the loader and the region runner are exercised, not mocked.
"""
import numpy as np
import pytest
import torch

from rvsm import teachers

PATCH = 32


def fake_spec():
    """The arch of the tiny teacher: 2 stages, features 4/8, one `surface` decoder with 2 classes."""
    return teachers.ArchSpec(
        in_channels=1, features_per_stage=[4, 8], n_blocks_per_stage=[1, 1], strides=[1, 2],
        kernel_size=3, conv_bias=True, norm="instance",
        task_decoders={"surface": teachers.DecoderSpec(out_channels=2, n_conv_per_stage=[1])})


def build_fake(seed=0):
    """The tiny net, deterministically initialised."""
    torch.manual_seed(int(seed))
    return teachers.VesuviusUNet(fake_spec()).eval()


def write_fake_ckpt(path, seed=0):
    """Save the tiny net the way villa saves the recto teacher, and return (path, state dict)."""
    net = build_fake(seed)
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    torch.save({"model": sd}, str(path))
    return str(path), sd


@pytest.fixture
def fake_teacher(tmp_path):
    """Registers `TEACHERS["fake"]` and yields a namespace with `.name`, `.ckpt`, `.spec`, `.patch`.

    The registration is undone afterwards so a test that lists the teachers still sees the two real
    ones and nothing else."""
    import types
    ckpt, _ = write_fake_ckpt(tmp_path / "fake_teacher.pth")
    spec = teachers.TeacherSpec(
        name="fake", state_key="model", kind="vesuvius", patch=(PATCH,) * 3,
        normalizer=teachers.Normalizer("zscore_instance"), activation="softmax", fg_channel=1,
        voxel_um=2.4, level=0, target="surface")
    teachers.register("fake", spec)
    try:
        yield types.SimpleNamespace(name="fake", ckpt=ckpt, spec=spec, patch=PATCH)
    finally:
        teachers.TEACHERS.pop("fake", None)


def slab_block(shape=(64, 64, 64), half=8, centre=None):
    """A bright slab at constant y in a block of air, in the uint8 the CT fixture uses.

    `centre` defaults to the middle of the block; putting it near a face is how a test gets windows
    that are pure air (the ones the region runner must never forward)."""
    Z, Y, X = shape
    c = Y / 2.0 if centre is None else float(centre)
    y = np.arange(Y)[None, :, None]
    v = (np.abs(y - c) < half).astype(np.uint8) * np.uint8(255)
    return np.broadcast_to(v, (Z, Y, X)).copy()
