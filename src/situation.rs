//! Where and when the stream runs, read from the running system at the
//! moment it is asked (the opening, each rollover): the time with its zone,
//! the host (name, system, processor, memory, GPUs, the Xeon Phi cards),
//! nothing written in by hand. A fact that cannot be read is left out, not
//! guessed. See situation.md.

use std::fs;
use std::path::{Path, PathBuf};

use crate::clock;

/// The host, as far as it can be read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Host {
    pub name: Option<String>,
    /// The distribution (`/etc/os-release`) and the kernel.
    pub os: Option<String>,
    pub kernel: Option<String>,
    pub cpu: Option<String>,
    pub threads: Option<usize>,
    pub mem_gib: Option<f64>,
    /// Up since, seconds (`/proc/uptime`).
    pub uptime_s: Option<f64>,
    /// GPU names, from the NVIDIA driver (`/proc/driver/nvidia/gpus`).
    pub gpus: Vec<String>,
    /// The Xeon Phi cards listed in the stack's cards file (one per line),
    /// and that file.
    pub cards: Option<(usize, PathBuf)>,
    /// The time zone's name (`TZ`, else the target of `/etc/localtime`).
    pub zone: Option<String>,
}

fn read(p: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(p).ok()
}

fn trimmed(s: Option<String>) -> Option<String> {
    s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// `KEY=value` (quotes off) from an os-release text.
fn os_release(text: &str, key: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .map(|v| v.trim().trim_matches('"').to_string())
        .filter(|v| !v.is_empty())
}

/// The first `model name` of a cpuinfo text, and how many processors it
/// lists.
fn cpuinfo(text: &str) -> (Option<String>, usize) {
    let name = text
        .lines()
        .find_map(|l| l.strip_prefix("model name"))
        .and_then(|r| r.split_once(':'))
        .map(|(_, v)| v.trim().to_string());
    let n = text.lines().filter(|l| l.starts_with("processor")).count();
    (name, n)
}

/// `MemTotal` of a meminfo text, in GiB.
fn mem_total(text: &str) -> Option<f64> {
    let kb: f64 = text
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb / (1024.0 * 1024.0))
}

/// The cards in a cards file: its lines that are neither empty nor
/// comments.
fn count_cards(text: &str) -> usize {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .count()
}

/// The stack's cards file: `PHI_CARDS`, else `~/.config/phi/cards`.
fn cards_file() -> Option<PathBuf> {
    std::env::var_os("PHI_CARDS")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/phi/cards")))
}

/// Read the host now.
pub fn host() -> Host {
    let release = read("/etc/os-release").unwrap_or_default();
    let (cpu, threads) = cpuinfo(&read("/proc/cpuinfo").unwrap_or_default());
    let mut gpus: Vec<String> = fs::read_dir("/proc/driver/nvidia/gpus")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| read(e.path().join("information")))
                .filter_map(|t| {
                    t.lines()
                        .find_map(|l| l.strip_prefix("Model:"))
                        .map(|m| m.trim().to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    gpus.sort();
    let cards = cards_file().and_then(|p| {
        let n = count_cards(&read(&p)?);
        (n > 0).then_some((n, p))
    });
    let zone = trimmed(std::env::var("TZ").ok()).or_else(|| {
        fs::read_link("/etc/localtime").ok().and_then(|p| {
            let s = p.to_string_lossy().into_owned();
            s.split_once("zoneinfo/").map(|(_, z)| z.to_string())
        })
    });
    Host {
        name: trimmed(read("/proc/sys/kernel/hostname")),
        os: os_release(&release, "PRETTY_NAME").or_else(|| os_release(&release, "NAME")),
        kernel: trimmed(read("/proc/sys/kernel/osrelease")),
        cpu,
        threads: (threads > 0).then_some(threads),
        mem_gib: mem_total(&read("/proc/meminfo").unwrap_or_default()),
        uptime_s: read("/proc/uptime")
            .and_then(|t| t.split_whitespace().next().and_then(|s| s.parse().ok())),
        gpus,
        cards,
        zone,
    }
}

/// The date and time to the microsecond, its zone and offset from UTC:
/// `2026-10-01 18:49:12.345678 CDT (UTC-05:00, America/Chicago)`.
pub fn now_line(t_us: i64, zone: Option<&str>) -> String {
    let (_, off) = clock::zone(t_us);
    let sign = if off < 0 { '-' } else { '+' };
    let off = off.abs();
    let utc = format!("UTC{sign}{:02}:{:02}", off / 3600, off % 3600 / 60);
    match zone {
        Some(z) => format!("{} ({utc}, {z})", clock::datetime(t_us)),
        None => format!("{} ({utc})", clock::datetime(t_us)),
    }
}

/// A span of seconds in words: `3 days 4 h`, `5 h 12 min`, `40 min`.
pub fn span(s: f64) -> String {
    let s = s.max(0.0) as u64;
    let (d, h, m) = (s / 86400, s % 86400 / 3600, s % 3600 / 60);
    if d > 0 {
        format!("{d} days {h} h")
    } else if h > 0 {
        format!("{h} h {m} min")
    } else {
        format!("{m} min")
    }
}

impl Host {
    /// The host in one sentence, what could be read of it.
    pub fn sentence(&self) -> String {
        let mut parts = Vec::new();
        let named = match (&self.name, &self.os, &self.kernel) {
            (Some(n), Some(o), Some(k)) => format!("{n}, {o} (Linux {k})"),
            (Some(n), Some(o), None) => format!("{n}, {o}"),
            (Some(n), None, Some(k)) => format!("{n} (Linux {k})"),
            (Some(n), None, None) => n.clone(),
            (None, Some(o), Some(k)) => format!("{o} (Linux {k})"),
            (None, Some(o), None) => o.clone(),
            (None, None, Some(k)) => format!("Linux {k}"),
            (None, None, None) => String::new(),
        };
        if !named.is_empty() {
            parts.push(named);
        }
        match (&self.cpu, self.threads) {
            (Some(c), Some(t)) => parts.push(format!("{c} ({t} threads)")),
            (Some(c), None) => parts.push(c.clone()),
            (None, Some(t)) => parts.push(format!("{t} threads")),
            (None, None) => {}
        }
        if let Some(m) = self.mem_gib {
            parts.push(format!("{m:.1} GiB of memory"));
        }
        if !self.gpus.is_empty() {
            parts.push(self.gpus.join(", "));
        }
        if let Some((n, _)) = &self.cards {
            parts.push(format!(
                "{n} Xeon Phi co-processor card{}",
                if *n == 1 { "" } else { "s" }
            ));
        }
        if let Some(u) = self.uptime_s {
            parts.push(format!("up {}", span(u)));
        }
        parts.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_are_read_as_they_are_written() {
        let rel = "NAME=\"CachyOS Linux\"\nPRETTY_NAME=\"CachyOS\"\nID=cachyos\n";
        assert_eq!(os_release(rel, "PRETTY_NAME").as_deref(), Some("CachyOS"));
        assert_eq!(os_release(rel, "VERSION"), None);
        let cpu = "processor\t: 0\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\nprocessor\t: 1\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\n";
        assert_eq!(
            cpuinfo(cpu),
            (Some("AMD Ryzen 7 5800X 8-Core Processor".into()), 2)
        );
        let mem = mem_total("MemTotal:       32772488 kB\nMemFree: 1 kB\n").unwrap();
        assert!((mem - 31.25).abs() < 0.01, "{mem}");
        let cards = "# cards\n0000:2f:00.0 /mnt/disk.img 2G\n\n0000:24:00.0 /mnt/disk1.img 2G\n";
        assert_eq!(count_cards(cards), 2);
    }

    #[test]
    fn the_time_carries_its_zone() {
        let t = clock::now_us();
        let l = now_line(t, Some("Test/Zone"));
        assert!(l.contains("UTC") && l.ends_with(", Test/Zone)"), "{l}");
        assert!(l.starts_with(&clock::datetime(t)), "{l}");
    }

    #[test]
    fn a_host_is_described_by_what_was_read() {
        let h = Host {
            name: Some("box".into()),
            os: Some("CachyOS".into()),
            kernel: Some("7.2.6".into()),
            cpu: Some("CPU".into()),
            threads: Some(16),
            mem_gib: Some(31.25),
            gpus: vec!["GPU".into()],
            cards: Some((2, PathBuf::from("/c"))),
            ..Default::default()
        };
        assert_eq!(
            h.sentence(),
            "box, CachyOS (Linux 7.2.6); CPU (16 threads); 31.2 GiB of memory; GPU; 2 Xeon Phi co-processor cards"
        );
        assert_eq!(Host::default().sentence(), "");
        // This machine, read live: whatever it has, the sentence is not empty.
        assert!(!host().sentence().is_empty());
    }
}
