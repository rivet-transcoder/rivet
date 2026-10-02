//! Content digests: SHA-256, SHA-1 and MD5.

use anyhow::{Result, bail};
use sha1::Digest as _;

/// A content digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DigestAlgorithm {
    Sha256,
    Sha1,
    Md5,
}

impl DigestAlgorithm {
    pub const ALL: [DigestAlgorithm; 3] = [DigestAlgorithm::Sha256, DigestAlgorithm::Sha1, DigestAlgorithm::Md5];

    pub fn as_str(self) -> &'static str {
        match self {
            DigestAlgorithm::Sha256 => "sha256",
            DigestAlgorithm::Sha1 => "sha1",
            DigestAlgorithm::Md5 => "md5",
        }
    }

    /// The digest of `data`.
    pub fn digest(self, data: &[u8]) -> Vec<u8> {
        match self {
            DigestAlgorithm::Sha256 => sha2::Sha256::digest(data).to_vec(),
            DigestAlgorithm::Sha1 => sha1::Sha1::digest(data).to_vec(),
            DigestAlgorithm::Md5 => md5::compute(data).0.to_vec(),
        }
    }

    /// The digest of `data` as lowercase hex.
    pub fn hex(self, data: &[u8]) -> String {
        to_hex(&self.digest(data))
    }
}

impl std::fmt::Display for DigestAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for DigestAlgorithm {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().replace('-', "").as_str() {
            "sha256" => DigestAlgorithm::Sha256,
            "sha1" => DigestAlgorithm::Sha1,
            "md5" => DigestAlgorithm::Md5,
            other => bail!("unknown digest `{other}` (sha256, sha1, md5)"),
        })
    }
}

pub(crate) fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
