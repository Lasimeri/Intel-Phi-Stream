//! Camera-feed monitoring: the cameras' status lines read, compared between
//! polls, and turned into events when something changes (`feeds.md`).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct Status {
    pub t: u64,
    #[allow(dead_code)]
    pub cam: String,
    pub faces: u32,
    #[allow(dead_code)]
    pub locked: bool,
    #[allow(dead_code)]
    pub box_x: i32,
    #[allow(dead_code)]
    pub box_y: i32,
    #[allow(dead_code)]
    pub box_w: i32,
    #[allow(dead_code)]
    pub box_h: i32,
    pub motion: f32,
    pub light: u32,
    #[allow(dead_code)]
    pub fps: u32,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// Its status stopped coming (stale past `OFFLINE_US`, or gone).
    Offline {
        cam: String,
        last_t: u64,
    },
    /// Its status came (first seen, or back after being offline).
    Online {
        cam: String,
        faces: u32,
    },
    FaceAppeared {
        cam: String,
        faces: u32,
    },
    FaceLeft {
        cam: String,
    },
    MotionNoFace {
        cam: String,
        motion: f32,
    },
    Dark {
        cam: String,
        light: u32,
    },
}

const OFFLINE_US: u64 = 10_000_000;
const MOTION_THRESH: f32 = 0.15;
const DARK_THRESH: u32 = 20;
/// A face counts as present while any second of the last five saw one: the
/// detector loses a face for a frame or a second, and a poll every 5 s that
/// read only the newest second called that leaving.
const FACE_WINDOW_US: u64 = 5_000_000;

pub fn parse_status_line(line: &str) -> Option<Status> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let mut t: Option<u64> = None;
    let mut cam: Option<String> = None;
    let mut faces: Option<u32> = None;
    let mut locked: Option<bool> = None;
    let mut bx: u32 = 0;
    let mut by: u32 = 0;
    let mut bw: u32 = 0;
    let mut bh: u32 = 0;
    let mut motion: f32 = 0.0;
    let mut light: u32 = 0;
    let mut fps: u32 = 0;
    let mut rem = line;
    while !rem.is_empty() {
        let eq = rem.find('=')?;
        let key = &rem[..eq];
        rem = &rem[eq + 1..];
        let sp = rem.find(' ');
        let val = match sp {
            Some(p) => {
                let v = &rem[..p];
                rem = &rem[p + 1..];
                v
            }
            None => {
                let v = rem;
                rem = "";
                v
            }
        };
        match key {
            "t" => t = Some(val.parse().ok()?),
            "cam" => cam = Some(val.to_string()),
            "faces" => faces = Some(val.parse().ok()?),
            "locked" => locked = Some(val == "1"),
            "box" => {
                let p: Vec<&str> = val.split(',').collect();
                if p.len() == 4 {
                    bx = p[0].parse().ok()?;
                    by = p[1].parse().ok()?;
                    bw = p[2].parse().ok()?;
                    bh = p[3].parse().ok()?
                } else {
                    return None;
                }
            }
            "motion" => motion = val.parse().ok()?,
            "light" => light = val.parse().ok()?,
            "fps" => fps = val.parse().ok()?,
            _ => {}
        }
    }
    Some(Status {
        t: t?,
        cam: cam?,
        faces: faces?,
        locked: locked.unwrap_or(false),
        box_x: bx as i32,
        box_y: by as i32,
        box_w: bw as i32,
        box_h: bh as i32,
        motion,
        light,
        fps,
    })
}

/// The most faces seen in any second of the log's tail within
/// `FACE_WINDOW_US` of `newest_t` (the status line's own time).
pub fn faces_in_window(log_tail: &str, newest_t: u64) -> u32 {
    log_tail
        .lines()
        .filter_map(parse_status_line)
        .filter(|s| s.t + FACE_WINDOW_US >= newest_t && s.t <= newest_t)
        .map(|s| s.faces)
        .max()
        .unwrap_or(0)
}

/// The last bytes of a file, from its first whole line.
fn tail(path: &Path, bytes: u64) -> String {
    let Ok(mut f) = File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map_or(0, |m| m.len());
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(bytes)));
    let mut s = String::new();
    let _ = f.read_to_string(&mut s);
    if len > bytes {
        if let Some(i) = s.find('\n') {
            s.drain(..=i);
        }
    }
    s
}

/// Every camera's newest status (`NAME.status`), its face count taken over
/// the last seconds of `NAME.log` (`faces_in_window`).
pub fn read_all_statuses(dir: &Path) -> HashMap<String, Status> {
    let mut m = HashMap::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return m;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        let Some(n) = p.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        let Some(name) = n.strip_suffix(".status").filter(|_| !n.starts_with('.')) else {
            continue;
        };
        let Ok(text) = fs::read_to_string(&p) else {
            continue;
        };
        let Some(mut s) = text.lines().filter_map(parse_status_line).next_back() else {
            continue;
        };
        let log = tail(&dir.join(format!("{name}.log")), 2048);
        s.faces = s.faces.max(faces_in_window(&log, s.t));
        m.insert(name.to_string(), s);
    }
    m
}

/// The events between two polls: a camera's state (fresh, or stale or gone)
/// at the previous poll (`prev_now_us`) and at this one (`now_us`). Each
/// event is told once, when its condition starts: going offline or coming
/// online, a face appearing or leaving, motion with no face starting, the
/// room going dark; while a condition holds nothing more is said.
pub fn compare_snapshots(
    prev_now_us: i64,
    now_us: i64,
    prev: &HashMap<String, Status>,
    cur: &HashMap<String, Status>,
) -> Vec<Event> {
    let fresh = |s: &Status, at: i64| (at.max(0) as u64).saturating_sub(s.t) <= OFFLINE_US;
    let mut names: Vec<&String> = prev.keys().chain(cur.keys()).collect();
    names.sort();
    names.dedup();
    let mut evs = Vec::new();
    for n in names {
        let p = prev.get(n).filter(|p| fresh(p, prev_now_us));
        let c = cur.get(n).filter(|c| fresh(c, now_us));
        match (p, c) {
            (Some(p), None) => evs.push(Event::Offline {
                cam: n.clone(),
                last_t: cur.get(n).map_or(p.t, |c| c.t),
            }),
            (None, Some(c)) => evs.push(Event::Online {
                cam: n.clone(),
                faces: c.faces,
            }),
            (Some(p), Some(c)) => {
                if p.faces == 0 && c.faces > 0 {
                    evs.push(Event::FaceAppeared {
                        cam: n.clone(),
                        faces: c.faces,
                    });
                } else if p.faces > 0 && c.faces == 0 {
                    evs.push(Event::FaceLeft { cam: n.clone() });
                }
                let moving = |s: &Status| s.motion > MOTION_THRESH && s.faces == 0;
                if moving(c) && !moving(p) {
                    evs.push(Event::MotionNoFace {
                        cam: n.clone(),
                        motion: c.motion,
                    });
                }
                if c.light < DARK_THRESH && p.light >= DARK_THRESH {
                    evs.push(Event::Dark {
                        cam: n.clone(),
                        light: c.light,
                    });
                }
            }
            (None, None) => {}
        }
    }
    evs
}

pub fn event_line(ev: &Event) -> String {
    let at = crate::clock::hms(crate::clock::now_us());
    let at = at.get(..8).unwrap_or(&at);
    match ev {
        Event::Offline { cam, last_t } => {
            let seen = crate::clock::hms(*last_t as i64);
            format!(
                "[{at}] camera {cam}: offline (last seen {})",
                seen.get(..8).unwrap_or(&seen)
            )
        }
        Event::Online { cam, faces } => {
            format!("[{at}] camera {cam}: online, {faces} face(s) seen")
        }
        Event::FaceAppeared { cam, faces } => {
            format!("[{at}] camera {cam}: a face appeared ({faces})")
        }
        Event::FaceLeft { cam } => format!("[{at}] camera {cam}: no face for 5 s"),
        Event::MotionNoFace { cam, motion } => {
            format!("[{at}] camera {cam}: motion ({motion:.3}) with no face seen")
        }
        Event::Dark { cam, light } => format!("[{at}] camera {cam}: dark (light {light})"),
    }
}

pub fn append_events(dir: &Path, evs: &[Event]) {
    let log_path = dir.join("events.log");
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&log_path) {
        for ev in evs {
            let _ = writeln!(f, "{}", event_line(ev));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(t: u64, faces: u32, motion: f32, light: u32) -> Status {
        Status {
            t,
            cam: "c".into(),
            faces,
            motion,
            light,
            ..Default::default()
        }
    }

    fn one(s: Status) -> HashMap<String, Status> {
        [("c".to_string(), s)].into_iter().collect()
    }

    #[test]
    fn a_status_line_parses() {
        let s = parse_status_line("t=1791020403094649 cam=bedroom faces=0 locked=0 box=0,0,0,0 motion=0.007 light=36 fps=10").unwrap();
        assert_eq!(s.cam, "bedroom");
        assert_eq!(s.faces, 0);
        assert_eq!(s.light, 36);
        assert_eq!(s.fps, 10);
    }

    #[test]
    fn a_camera_going_stale_is_offline_once() {
        // Fresh at the last poll; its status stops at t = 1 s.
        let s = st(1_000_000, 1, 0.0, 40);
        let evs = compare_snapshots(2_000_000, 50_000_000, &one(s.clone()), &one(s.clone()));
        assert!(matches!(evs.as_slice(), [Event::Offline { .. }]));
        // The next poll reads the same stale line: nothing more.
        let evs = compare_snapshots(50_000_000, 55_000_000, &one(s.clone()), &one(s));
        assert!(evs.is_empty());
        // Its file gone altogether: nothing more either.
        let evs = compare_snapshots(
            55_000_000,
            60_000_000,
            &one(st(1_000_000, 1, 0.0, 40)),
            &HashMap::new(),
        );
        assert!(evs.is_empty());
    }

    #[test]
    fn a_camera_coming_back_is_online() {
        let old = st(1_000_000, 0, 0.0, 40);
        let new = st(59_000_000, 1, 0.0, 40);
        let evs = compare_snapshots(55_000_000, 60_000_000, &one(old), &one(new));
        assert!(matches!(evs.as_slice(), [Event::Online { faces: 1, .. }]));
        let evs = compare_snapshots(
            0,
            60_000_000,
            &HashMap::new(),
            &one(st(59_000_000, 0, 0.0, 40)),
        );
        assert!(matches!(evs.as_slice(), [Event::Online { faces: 0, .. }]));
    }

    #[test]
    fn faces_motion_and_dark_are_told_when_they_start() {
        let a = st(10_000_000, 0, 0.0, 40);
        let b = st(15_000_000, 1, 0.0, 40);
        let evs = compare_snapshots(10_000_000, 15_000_000, &one(a.clone()), &one(b.clone()));
        assert!(matches!(
            evs.as_slice(),
            [Event::FaceAppeared { faces: 1, .. }]
        ));
        let evs = compare_snapshots(
            15_000_000,
            20_000_000,
            &one(b),
            &one(st(20_000_000, 0, 0.0, 40)),
        );
        assert!(matches!(evs.as_slice(), [Event::FaceLeft { .. }]));
        // Motion with no face: once when it starts, not while it lasts.
        let m = st(25_000_000, 0, 0.4, 40);
        let evs = compare_snapshots(20_000_000, 25_000_000, &one(a.clone()), &one(m.clone()));
        assert!(matches!(evs.as_slice(), [Event::MotionNoFace { .. }]));
        let m2 = st(30_000_000, 0, 0.5, 40);
        assert!(compare_snapshots(25_000_000, 30_000_000, &one(m), &one(m2)).is_empty());
        // Dark: once.
        let d = st(35_000_000, 0, 0.0, 10);
        let evs = compare_snapshots(
            30_000_000,
            35_000_000,
            &one(st(30_000_000, 0, 0.0, 40)),
            &one(d.clone()),
        );
        assert!(matches!(evs.as_slice(), [Event::Dark { light: 10, .. }]));
        assert!(compare_snapshots(
            35_000_000,
            40_000_000,
            &one(d),
            &one(st(40_000_000, 0, 0.0, 9))
        )
        .is_empty());
    }

    #[test]
    fn a_face_counts_over_the_last_five_seconds() {
        let log = "t=10000000 cam=c faces=1 locked=1 box=1,2,3,4 motion=0.01 light=40 fps=15\n\
                   t=11000000 cam=c faces=0 locked=0 box=0,0,0,0 motion=0.01 light=40 fps=15\n\
                   t=12000000 cam=c faces=0 locked=0 box=0,0,0,0 motion=0.01 light=40 fps=15\n";
        // A second without a face is not the face gone.
        assert_eq!(faces_in_window(log, 12_000_000), 1);
        // Five seconds on, the face seen at 10 s is out of the window.
        assert_eq!(faces_in_window(log, 16_000_000), 0);
    }
}
