"""Write reduced-width DRUNet state dicts in the LEGACY torch.save layout, without torch.

    python dpir_tiny_pth.py            # writes dpir_tiny_gray.pth and dpir_tiny_color.pth beside itself

The container mimics torch/serialization.py `_legacy_save` byte for byte in the parts the
reader in crates/codec/src/filter/dpir/pth.rs cares about: the magic-number / protocol /
sys_info pickles, tensors as `torch._utils._rebuild_tensor_v2(<persistent storage id>, ...)`,
an `OrderedDict` state dict carrying `_metadata` in its BUILD state, storage keys in the
lexicographic order torch writes them ('0', '1', '10', ..., '2'), then each storage as
int64 numel + raw f32 LE. The fake `torch` modules exist only so pickle can emit the same
GLOBAL references the real files hold.

Weights are deterministic (LCG) and Kaiming-scaled so activations stay O(1) through the net.
Architecture: UNetRes(in_nc, out_nc, nc=[2,4,8,16], nb=1) - the same names and shapes as the
release model (`nc=[64,128,256,512]`, `nb=4`), only narrower and shallower.
"""
import collections
import os
import pickle
import struct
import sys
import types

# Fake torch modules so pickle can emit `ctorch\nFloatStorage` / `ctorch._utils\n_rebuild_tensor_v2`.
torch = types.ModuleType("torch")
torch_utils = types.ModuleType("torch._utils")


class FloatStorage:
    pass


def _rebuild_tensor_v2(*args):
    raise NotImplementedError


FloatStorage.__module__ = "torch"
_rebuild_tensor_v2.__module__ = "torch._utils"
torch.FloatStorage = FloatStorage
torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
sys.modules["torch"] = torch
sys.modules["torch._utils"] = torch_utils


class Storage:
    def __init__(self, key, data):
        self.key, self.data = key, data


class Tensor:
    def __init__(self, storage, shape):
        self.storage, self.shape = storage, tuple(shape)

    def __reduce_ex__(self, protocol):
        stride, acc = [], 1
        for d in reversed(self.shape):
            stride.append(acc)
            acc *= d
        return (_rebuild_tensor_v2, (self.storage, 0, self.shape, tuple(reversed(stride)), False, collections.OrderedDict()))


class Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ("storage", FloatStorage, obj.key, "cpu", len(obj.data))
        return None


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def uniform(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        return (self.s >> 11) / float(1 << 53)


def conv_shapes(in_nc, out_nc, nc=(2, 4, 8, 16), nb=1):
    """(name, shape) in state-dict order, mirroring KAIR's UNetRes module tree."""
    out = [("m_head.weight", (nc[0], in_nc, 3, 3))]
    for i, stage in enumerate(["m_down1", "m_down2", "m_down3"]):
        for j in range(nb):
            out.append((f"{stage}.{j}.res.0.weight", (nc[i], nc[i], 3, 3)))
            out.append((f"{stage}.{j}.res.2.weight", (nc[i], nc[i], 3, 3)))
        out.append((f"{stage}.{nb}.weight", (nc[i + 1], nc[i], 2, 2)))
    for j in range(nb):
        out.append((f"m_body.{j}.res.0.weight", (nc[3], nc[3], 3, 3)))
        out.append((f"m_body.{j}.res.2.weight", (nc[3], nc[3], 3, 3)))
    for k, stage in enumerate(["m_up3", "m_up2", "m_up1"]):
        i = 3 - k
        out.append((f"{stage}.0.weight", (nc[i], nc[i - 1], 2, 2)))  # ConvTranspose2d: (in, out, k, k)
        for j in range(1, nb + 1):
            out.append((f"{stage}.{j}.res.0.weight", (nc[i - 1], nc[i - 1], 3, 3)))
            out.append((f"{stage}.{j}.res.2.weight", (nc[i - 1], nc[i - 1], 3, 3)))
    out.append(("m_tail.weight", (out_nc, nc[0], 3, 3)))
    return out


def write(path, in_nc, out_nc, seed):
    rng = Lcg(seed)
    sd = collections.OrderedDict()
    sd._metadata = collections.OrderedDict([("", {"version": 1}), ("m_head", {"version": 1})])
    storages = {}
    for idx, (name, shape) in enumerate(conv_shapes(in_nc, out_nc)):
        n = 1
        for d in shape:
            n *= d
        fan_in = shape[1] * shape[2] * shape[3]
        a = (6.0 / fan_in) ** 0.5  # Kaiming-uniform bound
        data = [(rng.uniform() * 2 - 1) * a for _ in range(n)]
        # residual blocks' second conv scaled down so the residual sum stays tame
        if name.endswith("res.2.weight"):
            data = [v * 0.25 for v in data]
        st = Storage(str(idx), data)
        storages[st.key] = st
        sd[name] = Tensor(st, shape)
    keys = sorted(storages.keys())  # torch: sorted(serialized_storages.keys()) - lexicographic
    with open(path, "wb") as f:
        f.write(pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2))
        f.write(pickle.dumps(1001, protocol=2))
        f.write(pickle.dumps({"protocol_version": 1001, "little_endian": True, "type_sizes": {"short": 2, "int": 4, "long": 4}}, protocol=2))
        Pickler(f, protocol=2).dump(sd)
        f.write(pickle.dumps(keys, protocol=2))
        for k in keys:
            d = storages[k].data
            f.write(struct.pack("<q", len(d)))
            f.write(struct.pack(f"<{len(d)}f", *d))
    print(path, os.path.getsize(path), "bytes", len(sd), "tensors")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    write(os.path.join(here, "dpir_tiny_gray.pth"), 2, 1, seed=1)
    write(os.path.join(here, "dpir_tiny_color.pth"), 4, 3, seed=2)
