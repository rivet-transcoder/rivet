//! Implementation of `rivet ipc` (Unix-domain-socket streaming server).
//!
//! On Unix the server listens on a Unix-domain socket; each connection streams
//! media in (write side), half-closes, then reads the transcoded AV1/MP4 back.
//! An optional `#rivet key=value …\n` header line can prefix the media bytes to
//! override per-job settings. On non-Unix platforms the command exists but bails
//! with an actionable error directing the user to `pipe` or `serve`.

use std::path::Path;

use anyhow::Result;
#[cfg(unix)]
use anyhow::Context;
#[cfg(unix)]
use rivet::TranscodeSettings;

/// Split an optional `#rivet key=value …\n` settings header off the front of the
/// stream. Real container magic bytes never start with `#rivet`, so this is
/// unambiguous. Returns `(parsed_settings, remaining_media_slice)`.
#[cfg(unix)]
fn split_ipc_settings(input: &[u8]) -> (Result<TranscodeSettings>, &[u8]) {
    const MAGIC: &[u8] = b"#rivet";
    if input.starts_with(MAGIC) {
        let nl = input.iter().position(|&b| b == b'\n').unwrap_or(input.len());
        let media_start = (nl + 1).min(input.len());
        let line = std::str::from_utf8(&input[MAGIC.len()..nl])
            .map(str::trim)
            .unwrap_or("");
        (TranscodeSettings::parse_kv_line(line), &input[media_start..])
    } else {
        (Ok(TranscodeSettings::default()), input)
    }
}

#[cfg(unix)]
pub(crate) fn run(socket: &Path) -> Result<()> {
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};

    // Drop a stale socket from a previous run (ignore "not found").
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)
        .with_context(|| format!("binding Unix socket {}", socket.display()))?;
    eprintln!(
        "rivet ipc: listening on {}\n           per connection: [optional `#rivet k=v …\\n` header] media → half-close → read AV1/MP4 back\n           e.g.  socat - UNIX-CONNECT:{} < in.mkv > out.mp4",
        socket.display(),
        socket.display(),
    );

    fn handle(mut stream: UnixStream) {
        let mut input = Vec::new();
        if let Err(e) = stream.read_to_end(&mut input) {
            eprintln!("rivet ipc: read error: {e}");
            return;
        }
        if input.is_empty() {
            return; // probe/keepalive connection
        }
        let (settings, media) = split_ipc_settings(&input);
        let settings = match settings {
            Ok(s) => s,
            Err(e) => {
                eprintln!("rivet ipc: bad settings header: {e:#}");
                return;
            }
        };
        eprintln!("rivet ipc: {} media bytes in", media.len());
        match super::stream_transcode(media, &settings) {
            Ok((bytes, frames, audio)) => {
                if let Err(e) = stream.write_all(&bytes) {
                    eprintln!("rivet ipc: write error: {e}");
                    return;
                }
                stream.flush().ok();
                let _ = stream.shutdown(std::net::Shutdown::Write);
                eprintln!("rivet ipc: {frames} frames → {} bytes out ({audio})", bytes.len());
            }
            Err(e) => eprintln!("rivet ipc: transcode error: {e:#}"),
        }
    }

    for stream in listener.incoming() {
        match stream {
            // One thread per connection; the process-wide GPU pool serializes
            // the actual GPU work, so concurrent clients just queue on it.
            Ok(s) => {
                std::thread::spawn(move || handle(s));
            }
            Err(e) => eprintln!("rivet ipc: accept error: {e}"),
        }
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn run(_socket: &Path) -> Result<()> {
    anyhow::bail!(
        "`rivet ipc` (Unix-domain socket) is Unix-only. On Windows, use \
         `rivet pipe` (stdin/stdout) or `rivet serve` (HTTP)."
    )
}
