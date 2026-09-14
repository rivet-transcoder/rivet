//! Reader for the **legacy** `torch.save` container — the format the DRUNet
//! release files (`drunet_gray.pth` / `drunet_color.pth`) are in.
//!
//! PyTorch has two on-disk formats. Checkpoints written since 1.6 are zip
//! archives (`data.pkl` + one file per storage), which candle's own
//! [`candle_core::pickle`] reads. Older files — and cszn/KAIR's 2020 release
//! assets — are the *legacy* layout ([`torch/serialization.py`,
//! `_legacy_save`](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/serialization.py)):
//!
//! ```text
//! pickle(MAGIC_NUMBER)      # 0x1950a86a20f9469cfc6c, a 10-byte LONG1
//! pickle(PROTOCOL_VERSION)  # 1001
//! pickle(sys_info)          # {protocol_version, little_endian, type_sizes}
//! pickle(state_dict)        # tensors are _rebuild_tensor_v2(<persistent storage id>, …)
//! pickle(storage_keys)      # list of str, the order the storages follow in
//! for each key:  int64 numel, then numel raw little-endian elements
//! ```
//!
//! The four pickles are decoded with candle's pickle VM (its `Stack` is public
//! and already understands `_rebuild_tensor_v2` argument tuples and persistent
//! ids); only the first one is matched by hand, because a 10-byte integer
//! overflows an `i64` and the VM's LONG1 decoder. The storages that follow
//! are plain `f32` runs that this module slices into the tensors the state
//! dict describes. Nothing here is DRUNet-specific — it is "a legacy `.pth`
//! holding a `str → contiguous float32 tensor` state dict".

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::pickle::{Object, Stack};

/// One tensor of a state dict: its name, shape, and row-major `f32` samples.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PthTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// `0x1950a86a20f9469cfc6c`, little-endian — `torch.serialization.MAGIC_NUMBER`.
const MAGIC: [u8; 10] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];
/// `torch.serialization.PROTOCOL_VERSION` for the legacy layout.
const PROTOCOL_VERSION: i32 = 1001;

/// Read a legacy `.pth` file into its tensors (see the module docs).
pub(super) fn read_legacy_pth_file(path: &Path) -> Result<Vec<PthTensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    read_legacy_pth(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Decode one pickle from the cursor (up to and including its STOP opcode).
fn read_pickle(cur: &mut Cursor<&[u8]>, what: &str) -> Result<Object> {
    let mut stack = Stack::empty();
    stack.read_loop(cur).with_context(|| format!("decoding the {what} pickle"))?;
    stack.finalize().with_context(|| format!("decoding the {what} pickle"))
}

/// The pickle VM's accessors return the offending object on mismatch; name it.
fn expect<T>(r: std::result::Result<T, Object>, what: &str) -> Result<T> {
    r.map_err(|o| anyhow!("expected {what}, found {o:?}"))
}

/// A tensor as the state dict describes it, before its storage is read.
struct TensorRef {
    name: String,
    storage_key: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// Parse one `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad,
/// backward_hooks)` value. `storage` is a persistent id
/// `('storage', <StorageClass>, key, location, numel[, view_metadata])`.
fn tensor_ref(name: String, value: Object) -> Result<TensorRef> {
    let (callable, args) = expect(value.reduce(), "a rebuilt tensor")?;
    let (module, class) = expect(callable.class(), "a rebuild function")?;
    if module != "torch._utils" || class != "_rebuild_tensor_v2" {
        bail!("tensor '{name}' is built by {module}.{class}; only torch._utils._rebuild_tensor_v2 is supported");
    }
    let mut args = expect(args.tuple(), "rebuild arguments")?.into_iter();
    let mut next = |what: &str| args.next().ok_or_else(|| anyhow!("tensor '{name}': missing {what}"));
    let storage = expect(next("storage")?.persistent_load(), "a persistent storage id")?;
    let pid = expect(storage.tuple(), "a storage id tuple")?;
    if pid.len() < 5 {
        bail!("tensor '{name}': storage id has {} fields, expected 5 or 6", pid.len());
    }
    let mut pid = pid.into_iter();
    let tag = expect(pid.next().unwrap().unicode(), "the 'storage' tag")?;
    if tag != "storage" {
        bail!("tensor '{name}': persistent id tag '{tag}' is not 'storage'");
    }
    let (_, storage_class) = expect(pid.next().unwrap().class(), "a storage class")?;
    if storage_class != "FloatStorage" {
        bail!("tensor '{name}' is stored as torch.{storage_class}; only FloatStorage (f32) is supported");
    }
    let storage_key = expect(pid.next().unwrap().unicode(), "a storage key")?;
    let offset = expect(next("offset")?.int_or_long(), "the storage offset")?;
    let dims = |o: Object, what: &str| -> Result<Vec<usize>> {
        expect(o.tuple(), what)?
            .into_iter()
            .map(|d| Ok(expect(d.int_or_long(), what)? as usize))
            .collect()
    };
    let shape = dims(next("size")?, "the size tuple")?;
    let stride = dims(next("stride")?, "the stride tuple")?;
    Ok(TensorRef { name, storage_key, offset: offset as usize, shape, stride })
}

/// Parse the whole legacy container. Errors name what was found, so a zip-format
/// checkpoint or a truncated download reads as that rather than as garbage.
pub(super) fn read_legacy_pth(bytes: &[u8]) -> Result<Vec<PthTensor>> {
    let mut cur = Cursor::new(bytes);
    // First pickle, matched by hand: PROTO 2, LONG1 of 10 bytes, STOP.
    let mut head = [0u8; 15];
    cur.read_exact(&mut head).context("reading the magic number")?;
    if head[..4] != [0x80, 2, 0x8a, 10] || head[4..14] != MAGIC || head[14] != b'.' {
        if bytes.starts_with(b"PK") {
            bail!("this is a zip-format torch checkpoint, not the legacy layout this reader handles");
        }
        bail!("not a torch.save file (bad magic number)");
    }
    let protocol = read_pickle(&mut cur, "protocol version")?;
    match protocol {
        Object::Int(PROTOCOL_VERSION) => {}
        Object::Long(v) if v == PROTOCOL_VERSION as i64 => {}
        o => bail!("unsupported torch serialization protocol {o:?} (want {PROTOCOL_VERSION})"),
    }
    let sys_info = read_pickle(&mut cur, "sys_info")?;
    for (k, v) in expect(sys_info.dict(), "the sys_info dict")? {
        if k == Object::Unicode("little_endian".into()) && v != Object::Bool(true) {
            bail!("big-endian checkpoints are not supported");
        }
    }
    let state = read_pickle(&mut cur, "state dict")?;
    let mut refs = Vec::new();
    for (k, v) in expect(state.dict(), "the state dict")? {
        let name = expect(k.unicode(), "a tensor name")?;
        // `OrderedDict._metadata` (per-module version tags) rides along in the
        // dict's BUILD state; it is not a tensor.
        if name == "_metadata" {
            continue;
        }
        refs.push(tensor_ref(name, v)?);
    }
    let keys = match read_pickle(&mut cur, "storage keys")? {
        Object::List(keys) => keys,
        o => bail!("expected the storage key list, found {o:?}"),
    };
    let mut storages: HashMap<String, Vec<f32>> = HashMap::with_capacity(keys.len());
    for key in keys {
        let key = expect(key.unicode(), "a storage key")?;
        let mut n = [0u8; 8];
        cur.read_exact(&mut n).with_context(|| format!("reading the length of storage {key}"))?;
        let numel = i64::from_le_bytes(n);
        if numel < 0 {
            bail!("storage {key} has negative length {numel}");
        }
        let start = cur.position() as usize;
        let end = start + numel as usize * 4;
        if end > bytes.len() {
            bail!("truncated: storage {key} needs {} bytes, {} remain (incomplete download?)", end - start, bytes.len() - start);
        }
        let data = bytes[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        cur.set_position(end as u64);
        storages.insert(key, data);
    }
    if (cur.position() as usize) != bytes.len() {
        bail!("{} trailing bytes after the last storage", bytes.len() - cur.position() as usize);
    }
    let mut out = Vec::with_capacity(refs.len());
    for r in refs {
        let storage = storages
            .get(&r.storage_key)
            .ok_or_else(|| anyhow!("tensor '{}' refers to missing storage {}", r.name, r.storage_key))?;
        let numel: usize = r.shape.iter().product();
        // Only contiguous (row-major) views: the stride of each dim must be the
        // product of the dims after it. A transposed or sliced view would need
        // a gather this reader does not do — and no state dict has one.
        let mut expected = 1usize;
        for (&d, &s) in r.shape.iter().zip(&r.stride).rev() {
            if d != 1 && s != expected {
                bail!("tensor '{}' is not contiguous (shape {:?}, stride {:?})", r.name, r.shape, r.stride);
            }
            expected *= d;
        }
        let end = r.offset + numel;
        if end > storage.len() {
            bail!("tensor '{}' needs {}..{} of storage {} which has {} elements", r.name, r.offset, end, r.storage_key, storage.len());
        }
        out.push(PthTensor { name: r.name, shape: r.shape, data: storage[r.offset..end].to_vec() });
    }
    Ok(out)
}
