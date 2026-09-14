//! DRUNet — cszn/DPIR's `UNetRes` — on candle.
//!
//! The architecture, transcribed from KAIR's `models/network_unet.py`
//! (`UNetRes(in_nc, out_nc, nc=[64,128,256,512], nb=4, act_mode='R',
//! downsample_mode='strideconv', upsample_mode='convtranspose')`):
//!
//! ```text
//! head   conv3x3(in_nc → nc0)                                x1
//! down1  nb × ResBlock(nc0), conv2x2 stride 2 (nc0 → nc1)    x2
//! down2  nb × ResBlock(nc1), conv2x2 stride 2 (nc1 → nc2)    x3
//! down3  nb × ResBlock(nc2), conv2x2 stride 2 (nc2 → nc3)    x4
//! body   nb × ResBlock(nc3)                                  x
//! up3    convT2x2 stride 2 (nc3 → nc2) of (x + x4), nb × ResBlock(nc2)
//! up2    convT2x2 stride 2 (nc2 → nc1) of (x + x3), nb × ResBlock(nc1)
//! up1    convT2x2 stride 2 (nc1 → nc0) of (x + x2), nb × ResBlock(nc0)
//! tail   conv3x3(nc0 → out_nc) of (x + x1)
//! ResBlock(c) = x + conv3x3(relu(conv3x3(x)))
//! ```
//!
//! No convolution has a bias and there is no normalisation layer. The input
//! is the image (`in_nc − 1` channels in `[0, 1]`) plus one constant channel
//! holding the noise level, and the three stride-2 stages mean the spatial
//! size must be a multiple of [`ALIGN`].
//!
//! Parameter names are the state dict's (`m_down1.0.res.0.weight` …), so the
//! release weights load without renaming; the shape is *inferred* from them
//! ([`Arch::infer`]) so a reduced-width network serves the unit tests.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder};

/// Spatial alignment the network needs: three stride-2 stages.
pub(super) const ALIGN: usize = 8;

/// The network's shape, read off a state dict's tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Arch {
    /// Input channels: image channels + 1 noise-level channel (2 gray, 4 colour).
    pub in_nc: usize,
    /// Output channels: 1 gray, 3 colour.
    pub out_nc: usize,
    /// Channel width per scale.
    pub nc: [usize; 4],
    /// Residual blocks per stage.
    pub nb: usize,
}

impl Arch {
    /// Infer the architecture from `{name: shape}`: `m_head.weight` gives the
    /// channel counts at the top, each `m_downN.<nb>.weight` (the stride-2
    /// conv, which follows the `nb` ResBlocks) the width of the next scale.
    pub(super) fn infer(shapes: &HashMap<String, Vec<usize>>) -> Result<Self> {
        let shape = |n: &str| shapes.get(n).ok_or_else(|| anyhow::anyhow!("state dict has no '{n}' (not a DRUNet checkpoint?)"));
        let head = shape("m_head.weight")?;
        let tail = shape("m_tail.weight")?;
        if head.len() != 4 || tail.len() != 4 {
            bail!("m_head/m_tail weights are not 4-D conv kernels");
        }
        let (nc0, in_nc, out_nc) = (head[0], head[1], tail[0]);
        let nb = (0..)
            .take_while(|j| shapes.contains_key(&format!("m_down1.{j}.res.0.weight")))
            .count();
        if nb == 0 {
            bail!("state dict has no 'm_down1.0.res.0.weight' (not a DRUNet checkpoint?)");
        }
        let nc1 = shape(&format!("m_down1.{nb}.weight"))?[0];
        let nc2 = shape(&format!("m_down2.{nb}.weight"))?[0];
        let nc3 = shape(&format!("m_down3.{nb}.weight"))?[0];
        Ok(Self { in_nc, out_nc, nc: [nc0, nc1, nc2, nc3], nb })
    }
}

struct ResBlock {
    c1: Conv2d,
    c2: Conv2d,
}

impl ResBlock {
    fn new(nc: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: candle_nn::conv2d_no_bias(nc, nc, 3, cfg, vb.pp("res.0"))?,
            c2: candle_nn::conv2d_no_bias(nc, nc, 3, cfg, vb.pp("res.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.c1.forward(x)?.relu()?;
        let y = self.c2.forward(&y)?;
        Ok((x + y)?)
    }
}

struct Down {
    blocks: Vec<ResBlock>,
    down: Conv2d,
}

struct Up {
    up: ConvTranspose2d,
    blocks: Vec<ResBlock>,
}

/// A loaded DRUNet, ready to run.
pub(super) struct DrUnet {
    arch: Arch,
    device: Device,
    head: Conv2d,
    down: Vec<Down>,
    body: Vec<ResBlock>,
    up: Vec<Up>,
    tail: Conv2d,
}

impl DrUnet {
    /// Build from a `{name: tensor}` state dict (any device / dtype: the
    /// tensors are moved to `device` as `f32`).
    pub(super) fn from_tensors(tensors: HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let shapes = tensors.iter().map(|(k, t)| (k.clone(), t.dims().to_vec())).collect();
        let arch = Arch::infer(&shapes)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::new(arch, vb, device.clone()).context("building DRUNet from the state dict")
    }

    fn new(arch: Arch, vb: VarBuilder, device: Device) -> Result<Self> {
        let Arch { in_nc, out_nc, nc, nb } = arch;
        let c3 = Conv2dConfig { padding: 1, ..Default::default() };
        let head = candle_nn::conv2d_no_bias(in_nc, nc[0], 3, c3, vb.pp("m_head"))?;
        let mut down = Vec::with_capacity(3);
        for (i, name) in ["m_down1", "m_down2", "m_down3"].iter().enumerate() {
            let v = vb.pp(name);
            let blocks = (0..nb).map(|j| ResBlock::new(nc[i], v.pp(j.to_string()))).collect::<Result<Vec<_>>>()?;
            let cfg = Conv2dConfig { stride: 2, ..Default::default() };
            let d = candle_nn::conv2d_no_bias(nc[i], nc[i + 1], 2, cfg, v.pp(nb.to_string()))?;
            down.push(Down { blocks, down: d });
        }
        let body = (0..nb).map(|j| ResBlock::new(nc[3], vb.pp("m_body").pp(j.to_string()))).collect::<Result<Vec<_>>>()?;
        let mut up = Vec::with_capacity(3);
        for (k, name) in ["m_up3", "m_up2", "m_up1"].iter().enumerate() {
            let i = 3 - k; // this stage maps nc[i] → nc[i - 1]
            let v = vb.pp(name);
            let cfg = ConvTranspose2dConfig { stride: 2, ..Default::default() };
            let u = candle_nn::conv_transpose2d_no_bias(nc[i], nc[i - 1], 2, cfg, v.pp("0"))?;
            let blocks = (1..=nb).map(|j| ResBlock::new(nc[i - 1], v.pp(j.to_string()))).collect::<Result<Vec<_>>>()?;
            up.push(Up { up: u, blocks });
        }
        let tail = candle_nn::conv2d_no_bias(nc[0], out_nc, 3, c3, vb.pp("m_tail"))?;
        Ok(Self { arch, device, head, down, body, up, tail })
    }

    pub(super) fn arch(&self) -> Arch {
        self.arch
    }

    /// The device the weights live on; inputs must be built there.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// Run `x` of shape `[1, in_nc, H, W]` (`H`, `W` multiples of [`ALIGN`]) to
    /// `[1, out_nc, H, W]`.
    pub(super) fn forward(&self, x0: &Tensor) -> Result<Tensor> {
        let dims = x0.dims4().context("DRUNet input must be [1, C, H, W]")?;
        if dims.1 != self.arch.in_nc || dims.2 % ALIGN != 0 || dims.3 % ALIGN != 0 {
            bail!("DRUNet input {dims:?}: want {} channels and H, W multiples of {ALIGN}", self.arch.in_nc);
        }
        let x1 = self.head.forward(x0)?;
        let mut skips = vec![x1.clone()];
        let mut x = x1;
        for d in &self.down {
            for b in &d.blocks {
                x = b.forward(&x)?;
            }
            x = d.down.forward(&x)?;
            skips.push(x.clone());
        }
        for b in &self.body {
            x = b.forward(&x)?;
        }
        for u in &self.up {
            x = (x + skips.pop().expect("one skip per stage"))?;
            x = u.up.forward(&x)?;
            for b in &u.blocks {
                x = b.forward(&x)?;
            }
        }
        x = (x + skips.pop().expect("the head skip"))?;
        Ok(self.tail.forward(&x)?)
    }
}
