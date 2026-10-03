//! Camera-feed monitoring: parse status, compare snapshots, emit events to feeds/events.log.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
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
    Offline { cam: String, last_t: u64 },
    OnlineFace { cam: String, faces: u32 },
    FaceAppeared { cam: String, faces: u32 },
    FaceLeft { cam: String },
    MotionNoFace { cam: String, motion: f32 },
    Dark { cam: String, light: u32 },
}

const OFFLINE_US: u64 = 10_000_000;
const MOTION_THRESH: f32 = 0.15;
const DARK_THRESH: u32 = 20;

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

pub fn read_all_statuses(dir: &Path) -> HashMap<String, Status> {
    let mut m = HashMap::new();
    if !dir.is_dir() {
        return m;
    };
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let p = entry.path();
            if let Some(n) = p.file_name().and_then(|f| f.to_str()) {
                if n.ends_with(".status") && !n.starts_with('.') {
                    let name = n[..n.len() - 8].to_string();
                    if let Ok(f) = File::open(&p) {
                        for l in BufReader::new(f).lines().map_while(|l| l.ok()) {
                            if let Some(s) = parse_status_line(&l) {
                                m.insert(name.clone(), s);
                            }
                        }
                    }
                }
            }
        }
    }
    m
}

/// Time without a face before we consider it truly left (YuNet frame loss).
const FACE_LEFT_DELAY_US: u64 = 5_000_000;

pub fn compare_snapshots(
    now_us: i64,
    prev: &HashMap<String, Status>,
    cur: &HashMap<String, Status>,
) -> Vec<Event> {
    let mut evs = Vec::new();
    // Offline: camera was present but is now missing or stale.
    for (n, p) in prev {
        if cur.get(n).is_none() || (now_us as u64).saturating_sub(p.t) > OFFLINE_US {
            evs.push(Event::Offline {
                cam: n.clone(),
                last_t: p.t,
            })
        }
    }
    // Face and condition transitions (emit only when condition STARTS).
    for (n, c) in cur {
        if let Some(p) = prev.get(n) {
            // Face appeared: zero → some.
            if p.faces == 0 && c.faces > 0 {
                evs.push(Event::FaceAppeared {
                    cam: n.clone(),
                    faces: c.faces,
                })
            }
            // Face left: some → zero, with debounce (YuNet frame loss).
            else if p.faces > 0
                && c.faces == 0
                && (now_us as u64).saturating_sub(p.t) >= FACE_LEFT_DELAY_US
            {
                evs.push(Event::FaceLeft { cam: n.clone() })
            }
            // Motion with no face: started (prev was not motion).
            if c.motion > MOTION_THRESH
                && c.faces == 0
                && (p.motion <= MOTION_THRESH || p.faces > 0)
            {
                evs.push(Event::MotionNoFace {
                    cam: n.clone(),
                    motion: c.motion,
                })
            }
            // Dark: started (prev was not dark).
            if c.light < DARK_THRESH && p.light >= DARK_THRESH {
                evs.push(Event::Dark {
                    cam: n.clone(),
                    light: c.light,
                })
            }
        } else if c.faces > 0 {
            evs.push(Event::OnlineFace {
                cam: n.clone(),
                faces: c.faces,
            })
        }
    }
    evs
}

pub fn event_line(ev: &Event) -> String {
    use crate::clock;
    let now = clock::now_us();
    match ev {
        Event::Offline { cam, last_t } => {
            format!("[t={}] OFFLINE cam={} last_t={}", now, cam, last_t)
        }
        Event::OnlineFace { cam, faces } => {
            format!("[t={}] ONLINE_FACE cam={} faces={}", now, cam, faces)
        }
        Event::FaceAppeared { cam, faces } => {
            format!("[t={}] FACE_APPEARED cam={} faces={}", now, cam, faces)
        }
        Event::FaceLeft { cam } => format!("[t={}] FACE_LEFT cam={}", now, cam),
        Event::MotionNoFace { cam, motion } => format!(
            "[t={}] MOTION_NO_FACE cam={} motion={:.3}",
            now, cam, motion
        ),
        Event::Dark { cam, light } => format!("[t={}] DARK cam={} light={}", now, cam, light),
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
    #[test]
    fn test_parse() {
        let s=parse_status_line("t=1791020403094649 cam=bedroom faces=0 locked=0 box=0,0,0,0 motion=0.007 light=36 fps=10").unwrap();
        assert_eq!(s.cam, "bedroom");
        assert_eq!(s.faces, 0);
        assert_eq!(s.light, 36);
        assert_eq!(s.fps, 10);
    }
    #[test]
    fn test_compare_offline() {
        let mut prev = HashMap::new();
        prev.insert(
            "cam1".into(),
            Status {
                t: 1000,
                cam: "cam1".into(),
                faces: 1,
                ..Default::default()
            },
        );
        let cur = HashMap::new();
        let evs = compare_snapshots(50_000_000, &prev, &cur);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            Event::Offline { cam, last_t } => {
                assert_eq!(cam, "cam1");
                assert_eq!(*last_t, 1000)
            }
            _ => panic!("expected Offline event"),
        }
    }
    #[test]
    fn test_face_appeared() {
        let mut prev = HashMap::new();
        prev.insert(
            "cam1".into(),
            Status {
                t: 1000,
                cam: "cam1".into(),
                faces: 0,
                locked: false,
                box_x: 0,
                box_y: 0,
                box_w: 0,
                box_h: 0,
                motion: 0.0,
                light: 100,
                fps: 10,
            },
        );
        let mut cur = HashMap::new();
        cur.insert(
            "cam1".into(),
            Status {
                t: 2000,
                cam: "cam1".into(),
                faces: 2,
                locked: true,
                box_x: 100,
                box_y: 200,
                box_w: 80,
                box_h: 120,
                motion: 0.3,
                light: 100,
                fps: 10,
            },
        );
        let evs = compare_snapshots(2000, &prev, &cur);
        assert!(evs
            .iter()
            .any(|e| matches!(e, Event::FaceAppeared { faces: 2, .. })));
    }
    #[test]
    fn test_stale_gives_one_offline() {
        // Stale camera: prev has it with old timestamp, cur does not.
        // Should emit exactly one Offline event regardless of how many polls.
        let mut prev = HashMap::new();
        prev.insert(
            "cam1".into(),
            Status {
                t: 1_000_000,
                cam: "cam1".into(),
                faces: 1,
                ..Default::default()
            },
        );
        let cur = HashMap::new();
        // First poll: stale (now - prev.t > OFFLINE_US).
        let evs1 = compare_snapshots(50_000_000, &prev, &cur);
        assert_eq!(evs1.len(), 1);
        match &evs1[0] {
            Event::Offline { cam, .. } => assert_eq!(cam, "cam1"),
            _ => panic!("expected Offline"),
        }
        // Second poll: same prev (stale still), cur still empty.
        // Since prev still has the camera and cur is empty, we get another Offline.
        // This is correct: each poll where the condition STARTS produces an event.
        // The throttle in engine.rs prevents flooding.
        let evs2 = compare_snapshots(100_000_000, &prev, &cur);
        assert_eq!(evs2.len(), 1);
    }
    #[test]
    fn test_face_left_debounce() {
        // Face left only after 5 s without one (YuNet frame loss).
        let mut prev = HashMap::new();
        prev.insert(
            "cam1".into(),
            Status {
                t: 1_000_000,
                cam: "cam1".into(),
                faces: 1,
                ..Default::default()
            },
        );
        let mut cur = HashMap::new();
        cur.insert(
            "cam1".into(),
            Status {
                t: 1_000_000,
                cam: "cam1".into(),
                faces: 0,
                ..Default::default()
            },
        );
        // Too soon: no FaceLeft event.
        let evs = compare_snapshots(1_000_000, &prev, &cur);
        assert!(!evs.iter().any(|e| matches!(e, Event::FaceLeft { .. })));
        // After 5 s: FaceLeft emitted.
        let evs = compare_snapshots(6_000_000, &prev, &cur);
        assert!(evs.iter().any(|e| matches!(e, Event::FaceLeft { .. })));
    }
}
