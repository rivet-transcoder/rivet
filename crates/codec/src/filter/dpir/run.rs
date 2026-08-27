//! The prepared `denoise=dpir` step: model loading, device selection, and the
//! tiled per-frame inference. Everything here needs candle (`dpir` feature);
//! the pure helpers it uses (tiling, levels, colour) live in the parent.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

use super::net::{ALIGN, DrUnet};
use super::{
    DEFAULT_TILE, DpirModel, ENV_DEVICE, ENV_TILE, Levels, TILE_OVERLAP, align_up, f32_to_plane, model_path, plane_to_f32, pth,
    rgb_to_yuv420, sigma_channel, tiles, validate_sigma, yuv420_to_rgb,
};
use crate::filter::{assemble, bps, planes};
use crate::frame::VideoFrame;

/// A loaded DRUNet with its σ, device and tiling, applied per frame by
/// [`FilterChain::apply`](crate::filter::FilterChain::apply).
pub struct PreparedDpir {
    sigma: f32,
    model: DpirModel,
    net: Net,
    device: Device,
    tile: usize,
    overlap: usize,
}

/// One tile's inference request: `[1, in_nc, h, w]` samples in, `[1, out_nc, h, w]` out.
struct Job {
    buf: Vec<f32>,
    shape: (usize, usize, usize, usize),
    reply: mpsc::Sender<Result<Vec<f32>>>,
}

/// Where the network lives. On the CPU it is called in place. On CUDA it is
/// owned by **one dedicated thread that never exits**: candle keeps its cuDNN
/// handle (and CUDA stream state) in thread-locals, and on Windows the
/// thread-local destructor that tears the handle down runs during
/// `LdrShutdownThread`, after the CUDA DLLs have already detached from the
/// exiting thread — `cudnnDestroy` then fails, cudarc unwraps it, and with
/// `panic = "abort"` the whole transcode dies *after* every frame was
/// filtered. A worker thread that parks forever when its channel closes
/// never runs that destructor (threads still alive at `ExitProcess` are
/// terminated without it), so the tokio / pump threads that call
/// [`PreparedDpir::apply`] never touch the device themselves.
enum Net {
    Local(DrUnet),
    Worker(mpsc::Sender<Job>),
}

impl Net {
    /// Move `net` onto its own thread (for a GPU device) or keep it here.
    fn new(net: DrUnet, device: &Device) -> Result<Self> {
        if device.is_cpu() {
            return Ok(Self::Local(net));
        }
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("rivet-dpir-cuda".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let r = Tensor::from_vec(job.buf, job.shape, net_device(&net))
                        .map_err(anyhow::Error::from)
                        .and_then(|t| net.forward(&t))
                        .and_then(|o| Ok(o.flatten_all()?.to_vec1::<f32>()?));
                    // The requester may have given up; nothing to do about it.
                    let _ = job.reply.send(r);
                }
                // Every PreparedDpir that used this thread is gone. Stay alive,
                // idle, so the thread-local CUDA state is never torn down
                // mid-process (see the enum docs).
                loop {
                    std::thread::park();
                }
            })
            .context("spawning the dpir CUDA worker thread")?;
        Ok(Self::Worker(tx))
    }

    fn forward(&self, buf: Vec<f32>, shape: (usize, usize, usize, usize), device: &Device) -> Result<Vec<f32>> {
        match self {
            Self::Local(net) => {
                let input = Tensor::from_vec(buf, shape, device)?;
                Ok(net.forward(&input)?.flatten_all()?.to_vec1::<f32>()?)
            }
            Self::Worker(tx) => {
                let (reply, rx) = mpsc::channel();
                tx.send(Job { buf, shape, reply }).map_err(|_| anyhow::anyhow!("the dpir CUDA worker thread is gone"))?;
                rx.recv().map_err(|_| anyhow::anyhow!("the dpir CUDA worker thread dropped a request"))?
            }
        }
    }
}

/// The device a network's weights are on (all on one device).
fn net_device(net: &DrUnet) -> &Device {
    net.device()
}

impl PreparedDpir {
    /// Resolve the model file (see [`super::model_path`]), pick the device
    /// ([`ENV_DEVICE`]), load the weights, and log what was chosen.
    pub fn prepare(sigma: f32, color: bool) -> Result<Self> {
        validate_sigma(sigma)?;
        let model = DpirModel::for_color(color);
        let path = model_path(model)?;
        let device_pref = std::env::var(ENV_DEVICE).ok();
        let (device, device_name) = select_device(device_pref.as_deref())?;
        let tile = match std::env::var(ENV_TILE) {
            Ok(t) => t.trim().parse::<usize>().with_context(|| format!("{ENV_TILE}='{t}' is not a tile size in pixels (0 = whole frame)"))?,
            Err(_) => DEFAULT_TILE,
        };
        let t0 = Instant::now();
        let tensors = load_state_dict(&path, &device)?;
        let prepared = Self::from_tensors(tensors, model, sigma, device, tile, TILE_OVERLAP)
            .with_context(|| format!("loading DPIR model {}", path.display()))?;
        tracing::info!(
            model = %path.display(),
            device = %device_name,
            sigma,
            tile,
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "dpir: DRUNet ({}) loaded",
            if color { "color" } else { "gray" }
        );
        Ok(prepared)
    }

    /// Build from an in-memory state dict — how the tests run a reduced
    /// network without the 130 MB release file.
    pub(crate) fn from_tensors(
        tensors: HashMap<String, Tensor>,
        model: DpirModel,
        sigma: f32,
        device: Device,
        tile: usize,
        overlap: usize,
    ) -> Result<Self> {
        validate_sigma(sigma)?;
        let net = DrUnet::from_tensors(tensors, &device)?;
        let arch = net.arch();
        let want = model.image_channels();
        if arch.in_nc != want + 1 || arch.out_nc != want {
            bail!(
                "this DRUNet takes {} channels in and {} out; {} needs {} in and {want} out (the {} file)",
                arch.in_nc,
                arch.out_nc,
                match model {
                    DpirModel::Gray => "denoise=dpir",
                    DpirModel::Color => "denoise=dpir:color",
                },
                want + 1,
                model.file_name()
            );
        }
        let net = Net::new(net, &device)?;
        Ok(Self { sigma, model, net, device, tile, overlap })
    }

    /// The device the network runs on, as a label (`cpu`, `cuda:0`).
    pub fn device_name(&self) -> String {
        device_label(&self.device)
    }

    /// Denoise one 8-bit or 10-bit 4:2:0 frame. Gray: luma through the
    /// network, chroma copied. Colour: YUV → R'G'B' → network → YUV.
    pub fn apply(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        let bps = bps(frame.format)?;
        let (yp, up, vp) = planes(frame, bps)?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        let lv = Levels::for_bps(bps);
        match self.model {
            DpirModel::Gray => {
                let y: Vec<f32> = plane_to_f32(yp, bps).into_iter().map(|v| v / lv.max).collect();
                let out = self.run(&[y], w, h)?;
                let y: Vec<f32> = out[0].iter().map(|v| v * lv.max).collect();
                Ok(assemble(frame, frame.width, frame.height, f32_to_plane(&y, bps, lv.max), up.to_vec(), vp.to_vec()))
            }
            DpirModel::Color => {
                let (y, u, v) = (plane_to_f32(yp, bps), plane_to_f32(up, bps), plane_to_f32(vp, bps));
                let rgb = yuv420_to_rgb(&y, &u, &v, w, h, frame.color_space, lv);
                let out = self.run(&rgb, w, h)?;
                let rgb: [Vec<f32>; 3] = [out[0].clone(), out[1].clone(), out[2].clone()];
                let (y, u, v) = rgb_to_yuv420(&rgb, w, h, frame.color_space, lv);
                Ok(assemble(
                    frame,
                    frame.width,
                    frame.height,
                    f32_to_plane(&y, bps, lv.max),
                    f32_to_plane(&u, bps, lv.max),
                    f32_to_plane(&v, bps, lv.max),
                ))
            }
        }
    }

    /// Run the network over `w×h` channels in `[0, 1]`, tile by tile, and
    /// return the same number of channels. Tiles are edge-replicated up to a
    /// multiple of [`ALIGN`]; only each tile's keep region lands in the output.
    fn run(&self, chans: &[Vec<f32>], w: usize, h: usize) -> Result<Vec<Vec<f32>>> {
        let nch = chans.len();
        let in_nc = nch + 1;
        let sig = sigma_channel(self.sigma, self.model);
        let mut out = vec![vec![0f32; w * h]; nch];
        for t in tiles(w, h, self.tile, self.overlap) {
            let (pw, ph) = (align_up(t.w, ALIGN), align_up(t.h, ALIGN));
            let plane = pw * ph;
            let mut buf = vec![sig; in_nc * plane];
            for (c, src) in chans.iter().enumerate() {
                for y in 0..ph {
                    let sy = t.y + y.min(t.h - 1);
                    let row = &src[sy * w..sy * w + w];
                    let dst = &mut buf[c * plane + y * pw..c * plane + y * pw + pw];
                    for (x, d) in dst.iter_mut().enumerate() {
                        *d = row[t.x + x.min(t.w - 1)];
                    }
                }
            }
            let o = self.net.forward(buf, (1, in_nc, ph, pw), &self.device)?;
            for (c, dst) in out.iter_mut().enumerate() {
                for y in t.keep_y..t.keep_y + t.keep_h {
                    let src = &o[c * plane + (y - t.y) * pw + (t.keep_x - t.x)..][..t.keep_w];
                    dst[y * w + t.keep_x..y * w + t.keep_x + t.keep_w].copy_from_slice(src);
                }
            }
        }
        Ok(out)
    }
}

/// Load `{name: tensor}` from a legacy `.pth` (see [`pth`]) or a `.safetensors`.
pub(super) fn load_state_dict(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let is_safetensors = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    if is_safetensors {
        return candle_core::safetensors::load(path, device).with_context(|| format!("loading {}", path.display()));
    }
    let tensors = pth::read_legacy_pth_file(path)?;
    tensors
        .into_iter()
        .map(|t| Ok((t.name, Tensor::from_vec(t.data, t.shape, device)?)))
        .collect()
}

/// `pref` is [`ENV_DEVICE`]: unset / `auto` picks CUDA when this build has it
/// and a device opens (else the CPU, with a warning saying why); `cpu` and
/// `cuda[:N]` are explicit, and asking for CUDA in a build without it is an
/// error rather than a silent CPU run.
fn select_device(pref: Option<&str>) -> Result<(Device, String)> {
    let pref = pref.map(|p| p.trim().to_ascii_lowercase());
    match pref.as_deref() {
        None | Some("") | Some("auto") => {
            #[cfg(feature = "dpir-cuda")]
            match Device::new_cuda(0) {
                Ok(d) => return Ok((d, "cuda:0".into())),
                Err(e) => tracing::warn!("dpir: CUDA device 0 unavailable ({e}); running DRUNet on the CPU"),
            }
            Ok((Device::Cpu, "cpu".into()))
        }
        Some("cpu") => Ok((Device::Cpu, "cpu".into())),
        Some(c) if c == "cuda" || c.starts_with("cuda:") => {
            let idx: usize = c.strip_prefix("cuda:").unwrap_or("0").parse().with_context(|| format!("{ENV_DEVICE}='{c}': bad device index"))?;
            #[cfg(feature = "dpir-cuda")]
            {
                let d = Device::new_cuda(idx).with_context(|| format!("opening CUDA device {idx} for dpir"))?;
                Ok((d, format!("cuda:{idx}")))
            }
            #[cfg(not(feature = "dpir-cuda"))]
            bail!("{ENV_DEVICE}=cuda:{idx} but this binary was built without the `dpir-cuda` feature")
        }
        Some(o) => bail!("{ENV_DEVICE}='{o}': want cpu, cuda, or cuda:N"),
    }
}

fn device_label(d: &Device) -> String {
    match d.location() {
        candle_core::DeviceLocation::Cpu => "cpu".into(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
    }
}
