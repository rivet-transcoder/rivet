//! How much of a discrete GPU's memory the CPU can reach through its PCI BAR —
//! whether Resizable BAR is in effect.
//!
//! A discrete card exposes its VRAM to the CPU through one memory BAR. Without
//! Resizable BAR that window is the PCI default, 256 MiB, whatever the card
//! holds; with it, the window is the whole of VRAM. Most work doesn't care,
//! but some does: Intel's compute runtime (OpenCL / Level Zero, and so
//! OpenVINO's GPU plugin) won't expose an Arc card behind a small BAR on the
//! upstream `i915` driver, and prints only `WARNING: Small BAR detected`. QSV
//! encode and decode carry on regardless, which makes the cause hard to see.
//!
//! Linux only, from sysfs, and without privileges:
//! - `/sys/bus/pci/devices/<bdf>/resource` gives each BAR's current size;
//! - `resourceN_resize` exists for each BAR the device can resize (the
//!   Resizable BAR capability; the kernel has created these since 6.1), and
//!   lists the sizes the device supports.

use super::types::GpuDevice;

/// What a discrete card's VRAM window looks like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarReport {
    /// Which BAR is the VRAM window: the largest memory BAR.
    pub index: usize,
    /// Its size now, in bytes.
    pub bytes: u64,
    /// The card's VRAM in MiB, `0` when unknown.
    pub vram_mib: u64,
    /// The device has the Resizable BAR capability for this BAR. `None` when
    /// the kernel predates the `resourceN_resize` files and can't say.
    pub resizable: Option<bool>,
    /// The largest size the device supports for this BAR, in bytes, when
    /// the kernel says.
    pub max_bytes: Option<u64>,
    /// This host is a virtual machine (the CPU's `hypervisor` flag): the card
    /// is passed through, and the hypervisor decides what BAR the guest sees.
    pub virtualised: bool,
}

/// Whether the CPU can reach the card's VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarVerdict {
    /// The window covers VRAM: Resizable BAR is in effect.
    Full,
    /// The window is smaller than VRAM.
    Small,
    /// VRAM is unknown, so it can't be compared.
    Unknown,
}

impl BarReport {
    pub fn verdict(&self) -> BarVerdict {
        if self.vram_mib == 0 {
            return BarVerdict::Unknown;
        }
        // VRAM sizes are reported a little under the power of two the window
        // rounds up to; half is far above any default window and far below
        // any real card's VRAM.
        if self.bytes >= self.vram_mib * 1024 * 1024 / 2 { BarVerdict::Full } else { BarVerdict::Small }
    }

    /// One line for `rivet devices`: `full (6144 MiB of 6144 MiB)`, or
    /// `small (256 MiB of 6144 MiB): ...` with what's in the way.
    pub fn describe(&self) -> String {
        let window = format!("{} MiB", self.bytes / (1024 * 1024));
        match self.verdict() {
            BarVerdict::Full => format!("full ({window} of {} MiB VRAM)", self.vram_mib),
            BarVerdict::Unknown => format!("{window} window (VRAM unknown)"),
            BarVerdict::Small => {
                let cause = match (self.resizable, self.max_bytes) {
                    (Some(true), Some(max)) if max > self.bytes => format!(
                        "the card can resize it to {} MiB, but the platform hasn't",
                        max / (1024 * 1024)
                    ),
                    (Some(true), _) => "the card supports Resizable BAR, but the platform hasn't enabled it".into(),
                    (Some(false), _) if self.virtualised => {
                        "the card's Resizable BAR capability isn't visible here (a VM: the hypervisor may hide it)".into()
                    }
                    (Some(false), _) => "the card doesn't offer Resizable BAR to this host".into(),
                    (None, _) => "this kernel can't say whether the card supports Resizable BAR".into(),
                };
                let fix = if self.virtualised {
                    "enable Above 4G Decoding and Resizable BAR in the host's firmware, and pass the full BAR through to the VM"
                } else {
                    "enable Above 4G Decoding and Resizable BAR in the firmware"
                };
                format!("small ({window} of {} MiB VRAM): {cause}; {fix}", self.vram_mib)
            }
        }
    }

    /// What a small window costs on this vendor's card, for display.
    pub fn consequence(&self, device: &GpuDevice) -> Option<&'static str> {
        (self.verdict() == BarVerdict::Small && device.vendor == super::GpuVendor::Intel).then_some(
            "Intel's compute runtime won't expose this card: no OpenCL, Level Zero or OpenVINO GPU (QSV is unaffected)",
        )
    }
}

/// The VRAM window of `device`, a discrete card on Linux. `None` elsewhere,
/// for an integrated GPU (no VRAM of its own), or when sysfs doesn't say.
pub fn bar_report(device: &GpuDevice) -> Option<BarReport> {
    #[cfg(target_os = "linux")]
    {
        linux::bar_report(device)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        None
    }
}

/// A PCI memory BAR (`IORESOURCE_MEM`).
#[cfg(any(target_os = "linux", test))]
const IORESOURCE_MEM: u64 = 0x200;

/// The largest memory BAR among the six standard ones in a sysfs `resource`
/// file (`start end flags` per line, hex), as `(index, bytes)`.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn largest_memory_bar(resource: &str) -> Option<(usize, u64)> {
    resource
        .lines()
        .take(6)
        .enumerate()
        .filter_map(|(i, line)| {
            let mut fields = line.split_whitespace().map(|f| u64::from_str_radix(f.trim_start_matches("0x"), 16).ok());
            let (start, end, flags) = (fields.next()??, fields.next()??, fields.next()??);
            (start != 0 && end > start && flags & IORESOURCE_MEM != 0).then_some((i, end - start + 1))
        })
        .max_by_key(|&(_, bytes)| bytes)
}

/// The largest size in a `resourceN_resize` read: a hex mask whose bit `n`
/// means `2^n` MiB is supported.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn largest_supported(mask: &str) -> Option<u64> {
    let mask = u64::from_str_radix(mask.trim().trim_start_matches("0x"), 16).ok()?;
    (mask != 0).then(|| (1u64 << (63 - mask.leading_zeros())) * 1024 * 1024)
}

/// Whether kernel `release` (`6.8.0-45-generic`) creates `resourceN_resize`.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn kernel_has_resize_files(release: &str) -> bool {
    let mut parts = release.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty());
    let major: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    (major, minor) >= (6, 1)
}

#[cfg(target_os = "linux")]
mod linux {
    use std::path::Path;

    use super::*;

    pub(super) fn bar_report(device: &GpuDevice) -> Option<BarReport> {
        if device.host_pci_address.is_empty() || device.vram_mib == 0 {
            return None;
        }
        let bdf = if device.host_pci_address.matches(':').count() >= 2 {
            device.host_pci_address.clone()
        } else {
            format!("0000:{}", device.host_pci_address)
        };
        let dir = Path::new("/sys/bus/pci/devices").join(bdf);
        let (index, bytes) = largest_memory_bar(&std::fs::read_to_string(dir.join("resource")).ok()?)?;
        let resize = dir.join(format!("resource{index}_resize"));
        let resizable = if resize.exists() {
            Some(true)
        } else if std::fs::read_to_string("/proc/sys/kernel/osrelease").is_ok_and(|r| kernel_has_resize_files(r.trim())) {
            Some(false)
        } else {
            None
        };
        let max_bytes = std::fs::read_to_string(&resize).ok().and_then(|m| largest_supported(&m));
        let virtualised = std::fs::read_to_string("/proc/cpuinfo")
            .is_ok_and(|c| c.lines().any(|l| l.starts_with("flags") && l.split_whitespace().any(|f| f == "hypervisor")));
        Some(BarReport { index, bytes, vram_mib: device.vram_mib, resizable, max_bytes, virtualised })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Arc A380 behind the default window: BAR0 16 MiB registers, BAR2 256
    /// MiB VRAM aperture, an expansion ROM line, then unused entries.
    const SMALL: &str = "\
0x00000000fb000000 0x00000000fbffffff 0x0000000000040200
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000006000000000 0x000000600fffffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x00000000fc000000 0x00000000fc1fffff 0x0000000000046200
";

    fn report(bytes: u64, resizable: Option<bool>, max_bytes: Option<u64>, virtualised: bool) -> BarReport {
        BarReport { index: 2, bytes, vram_mib: 6144, resizable, max_bytes, virtualised }
    }

    #[test]
    fn finds_the_vram_window() {
        assert_eq!(largest_memory_bar(SMALL), Some((2, 256 << 20)), "the ROM line (7th) is not a BAR");
        let full = SMALL.replace("0x000000600fffffff", "0x000000617fffffff");
        assert_eq!(largest_memory_bar(&full), Some((2, 6144 << 20)));
        assert_eq!(largest_memory_bar("garbage\n"), None);
        // An I/O BAR (flags 0x101) is not a memory window, however large.
        assert_eq!(largest_memory_bar("0x1000 0xffffff 0x101\n"), None);
    }

    #[test]
    fn reads_the_supported_sizes() {
        // Bits 8..=13: 256 MiB to 8 GiB.
        assert_eq!(largest_supported("0x3f00\n"), Some(8192 << 20));
        assert_eq!(largest_supported("0x100"), Some(256 << 20));
        assert_eq!(largest_supported("0"), None);
    }

    #[test]
    fn knows_which_kernels_say() {
        assert!(kernel_has_resize_files("6.8.0-45-generic"));
        assert!(kernel_has_resize_files("7.0.0-34-generic"));
        assert!(!kernel_has_resize_files("5.15.0-100-generic"));
        assert!(!kernel_has_resize_files("6.0.19"));
    }

    #[test]
    fn verdicts_and_causes() {
        assert_eq!(report(6144 << 20, Some(true), None, false).verdict(), BarVerdict::Full);
        assert_eq!(report(8192 << 20, Some(true), None, false).verdict(), BarVerdict::Full, "rounded up past VRAM");
        let small = report(256 << 20, Some(true), Some(8192 << 20), false);
        assert_eq!(small.verdict(), BarVerdict::Small);
        assert!(small.describe().starts_with("small (256 MiB of 6144 MiB VRAM): the card can resize it to 8192 MiB"));
        assert!(report(256 << 20, Some(false), None, true).describe().contains("a VM"));
        assert!(report(256 << 20, Some(false), None, true).describe().contains("pass the full BAR through"));
        assert!(report(256 << 20, None, None, false).describe().contains("can't say"));
        let unknown = BarReport { vram_mib: 0, ..small };
        assert_eq!(unknown.verdict(), BarVerdict::Unknown);
    }
}
