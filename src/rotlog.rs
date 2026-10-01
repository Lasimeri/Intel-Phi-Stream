//! A log file that rotates by size. The stream never stops, and its logs
//! (`stream.log`, `chain.log`, `mind.log` at one line per token,
//! `reflect.log`) grew without bound; the dev stream's own audit of its
//! checks found it ("no rotation policy ... could grow unbounded"). Past
//! `MAX` bytes a log is renamed to `NAME.1` (the previous `.1` replaced)
//! and a fresh one begun, so each log holds at most twice `MAX` on disk.
//! A log that cannot be opened is skipped silently, as before. See
//! rotlog.md.

use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;

/// 64 MiB a file: about a day of `mind.log` at the stream's pace.
pub const MAX: u64 = 64 << 20;

pub struct RotLog {
    path: PathBuf,
    file: Option<File>,
    size: u64,
    max: u64,
}

impl RotLog {
    pub fn open(path: PathBuf) -> Self {
        Self::with_max(path, MAX)
    }

    pub fn with_max(path: PathBuf, max: u64) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok();
        let size = file
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map_or(0, |m| m.len());
        Self {
            path,
            file,
            size,
            max,
        }
    }

    fn rotate(&mut self) {
        self.file = None;
        let mut old = self.path.clone().into_os_string();
        old.push(".1");
        let _ = fs::rename(&self.path, &old);
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .ok();
        self.size = 0;
    }

    /// Bytes appended, the file rotated first when they would take it past
    /// its size (never a file of one oversized write alone).
    pub fn write(&mut self, bytes: &[u8]) {
        if self.size > 0 && self.size + bytes.len() as u64 > self.max {
            self.rotate();
        }
        if let Some(f) = &mut self.file {
            if f.write_all(bytes).is_ok() {
                self.size += bytes.len() as u64;
            }
        }
    }

    /// A line appended.
    pub fn line(&mut self, s: &str) {
        let mut b = Vec::with_capacity(s.len() + 1);
        b.extend_from_slice(s.as_bytes());
        b.push(b'\n');
        self.write(&b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_rotates_past_its_size_and_keeps_one_before() {
        let dir = std::env::temp_dir().join(format!("phi-stream-rotlog-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.log");
        let _ = fs::remove_file(&p);
        let mut l = RotLog::with_max(p.clone(), 10);
        l.line("abcd"); // 5 bytes
        l.line("efgh"); // 10
        assert_eq!(fs::read_to_string(&p).unwrap(), "abcd\nefgh\n");
        l.line("ijkl"); // past 10: rotated first
        assert_eq!(
            fs::read_to_string(dir.join("x.log.1")).unwrap(),
            "abcd\nefgh\n"
        );
        assert_eq!(fs::read_to_string(&p).unwrap(), "ijkl\n");
        // Reopened, it counts what the file already holds.
        let mut l = RotLog::with_max(p.clone(), 10);
        l.line("mnop");
        l.line("qrst");
        assert_eq!(
            fs::read_to_string(dir.join("x.log.1")).unwrap(),
            "ijkl\nmnop\n"
        );
        assert_eq!(fs::read_to_string(&p).unwrap(), "qrst\n");
        fs::remove_dir_all(&dir).unwrap();
    }
}
