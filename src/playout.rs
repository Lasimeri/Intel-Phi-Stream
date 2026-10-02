//! The display's own clock. The engine places pieces in jumps: a token
//! every 25 ms alone, 35 ms with a check's lane beside it, nothing for a
//! third of a second while a reading's or a question's chunk is decoded
//! (the engine's thread is inside `llama_decode` then), and a burst when a
//! check lets its held token go. Shown as placed, every jump shows. After
//! BF++'s scene oracle (`runtime/bfpp_rt_3d_oracle.c` in Lasimeri/bfpp: a
//! consumer on its own thread, decoupled from its producer's rate by a
//! buffer that absorbs the mismatch, degrading gracefully, clamped against
//! runaway), the text goes out here, on a thread of its own, at the pace
//! it has been arriving, held about `horizon` behind: slower as the buffer
//! runs low, faster as it fills, never later than twice the horizon, and
//! at once when the display has been starved (a check held its token
//! longer than the buffer lasted). Other events pass
//! at once; the end of a run (`Stopped`, `Done`) and a `Flush` (a pause)
//! send everything held first. See playout.md.

use std::collections::VecDeque;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::clock;
use crate::engine::Event;

pub enum Msg {
    Event(Event),
    Flush,
}

/// The pacing, apart from the thread and the clock (`now`: microseconds of
/// the monotonic clock).
struct Pacer {
    horizon: i64,
    /// Microseconds between arrivals, smoothed; how many were seen.
    pace: f64,
    n: u32,
    last_arrival: i64,
    next_due: i64,
    held: VecDeque<(i64, Event)>,
}

/// Arrivals seen before the pace is trusted (until then: at the horizon).
const WARM: u32 = 4;

impl Pacer {
    fn new(horizon: i64) -> Self {
        Self {
            horizon,
            pace: 0.0,
            n: 0,
            last_arrival: 0,
            next_due: 0,
            held: VecDeque::new(),
        }
    }

    /// A piece of text arrives. The pace follows the arrivals, each interval
    /// clamped to a third to three times the pace so far: a stall or a burst
    /// is the jitter the buffer absorbs, not the pace.
    fn arrive(&mut self, now: i64, e: Event) {
        if self.last_arrival > 0 {
            let mut dt = (now - self.last_arrival).max(0) as f64;
            if self.n > 0 {
                dt = dt.clamp(self.pace / 3.0, self.pace * 3.0);
            }
            self.pace = if self.n == 0 {
                dt
            } else {
                0.9 * self.pace + 0.1 * dt
            };
            self.n += 1;
        }
        self.last_arrival = now;
        self.held.push_back((now, e));
    }

    /// The next piece whose time has come, if any.
    fn pop_due(&mut self, now: i64) -> Option<Event> {
        let &(arrived, _) = self.held.front()?;
        let h = self.horizon;
        let lag = now - arrived;
        let go = if h == 0 || lag >= 2 * h {
            true
        } else if self.n < WARM {
            lag >= h
        } else {
            now >= self.next_due
        };
        if !go {
            return None;
        }
        if h > 0 && self.n >= WARM {
            // Slower while the head is younger than the horizon (the buffer
            // is low), faster while it is older; a clock that fell behind is
            // not owed a burst.
            let scale = (h as f64 / lag.max(1) as f64).clamp(0.5, 2.0);
            let step = (self.pace * scale) as i64;
            self.next_due = self.next_due.max(now - step) + step;
        }
        self.held.pop_front().map(|(_, e)| e)
    }

    /// When to look again (for the wait), none when nothing is held.
    fn next_wake(&self, now: i64) -> Option<i64> {
        let &(arrived, _) = self.held.front()?;
        let h = self.horizon;
        Some(if h == 0 || now - arrived >= 2 * h {
            now
        } else if self.n < WARM {
            arrived + h
        } else {
            self.next_due.min(arrived + 2 * h)
        })
    }

    fn drain(&mut self) -> impl Iterator<Item = Event> + '_ {
        self.held.drain(..).map(|(_, e)| e)
    }
}

/// The engine's side: events in, paced events out on `out`.
pub struct Playout {
    tx: Sender<Msg>,
    handle: Option<JoinHandle<()>>,
}

impl Playout {
    pub fn start(horizon_us: i64, out: Sender<Event>) -> Self {
        let (tx, rx) = mpsc::channel::<Msg>();
        let handle = thread::spawn(move || {
            let mut p = Pacer::new(horizon_us);
            loop {
                let now = clock::mono_us();
                while let Some(e) = p.pop_due(now) {
                    let _ = out.send(e);
                }
                let wait = p
                    .next_wake(now)
                    .map_or(1_000_000, |t| (t - now).clamp(200, 1_000_000));
                match rx.recv_timeout(Duration::from_micros(wait as u64)) {
                    Ok(Msg::Event(e @ Event::Text(..))) => p.arrive(clock::mono_us(), e),
                    Ok(Msg::Event(e @ (Event::Stopped | Event::Done { .. }))) => {
                        for h in p.drain() {
                            let _ = out.send(h);
                        }
                        let _ = out.send(e);
                    }
                    Ok(Msg::Event(e)) => {
                        let _ = out.send(e);
                    }
                    Ok(Msg::Flush) => {
                        for h in p.drain() {
                            let _ = out.send(h);
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        for h in p.drain() {
                            let _ = out.send(h);
                        }
                        break;
                    }
                }
            }
        });
        Self {
            tx,
            handle: Some(handle),
        }
    }

    /// An event in; false when the playout is gone.
    pub fn send(&self, e: Event) -> bool {
        self.tx.send(Msg::Event(e)).is_ok()
    }

    /// Everything held goes out now (a pause).
    pub fn flush(&self) {
        let _ = self.tx.send(Msg::Flush);
    }

    /// Everything out, the thread ended: the receiver has every event when
    /// this returns.
    pub fn finish(mut self) {
        let (tx, _) = mpsc::channel();
        drop(std::mem::replace(&mut self.tx, tx));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Kind;

    fn text(i: usize) -> Event {
        Event::Text(format!("{i}"), Kind::Think, 0, None)
    }

    /// Arrivals at the given times; the pacer driven every millisecond up to
    /// `until`; the times each piece went out.
    fn run(horizon: i64, arrivals: &[i64], until: i64) -> Vec<i64> {
        let mut p = Pacer::new(horizon);
        let mut out = Vec::new();
        let mut a = 0;
        let mut now = 0;
        while now <= until {
            while a < arrivals.len() && arrivals[a] <= now {
                p.arrive(arrivals[a], text(a));
                a += 1;
            }
            while p.pop_due(now).is_some() {
                out.push(now);
            }
            now += 1_000;
        }
        out
    }

    #[test]
    fn a_zero_horizon_passes_everything_at_once() {
        let out = run(0, &[1_000, 2_000, 3_000], 10_000);
        assert_eq!(out, vec![1_000, 2_000, 3_000]);
    }

    #[test]
    fn a_stall_shorter_than_the_horizon_does_not_show() {
        // A token every 25 ms for 4 s, but nothing for 344 ms at 2 s: the
        // question's chunk going in.
        let mut arrivals = Vec::new();
        let mut t = 0;
        while t < 4_000_000 {
            arrivals.push(t);
            t += if (2_000_000..2_025_000).contains(&t) {
                344_000
            } else {
                25_000
            };
        }
        let out = run(1_000_000, &arrivals, 8_000_000);
        assert_eq!(out.len(), arrivals.len());
        // In the steady part after warming up, no gap between pieces shown
        // reaches the stall: the largest is under 100 ms.
        let gaps: Vec<i64> = out.windows(2).map(|w| w[1] - w[0]).collect();
        let worst = gaps[20..gaps.len() - 5].iter().copied().max().unwrap();
        assert!(worst < 100_000, "a gap of {worst} us showed");
        // Placed every 25 ms, shown as placed the stall would be a 344 ms gap.
        let placed_worst = arrivals.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert_eq!(placed_worst, 344_000);
    }

    #[test]
    fn nothing_waits_longer_than_twice_the_horizon() {
        let arrivals: Vec<i64> = (0..200).map(|i| i * 25_000).collect();
        let out = run(1_000_000, &arrivals, 20_000_000);
        for (a, o) in arrivals.iter().zip(&out) {
            let lag = o - a;
            assert!((0..=2_000_000).contains(&lag), "lag {lag}");
        }
    }

    #[test]
    fn a_check_longer_than_the_horizon_resumes_at_once() {
        // 25 ms tokens; a check holds its token 1.25 s, then everything
        // placed meanwhile arrives at once (50 pieces), then 25 ms again.
        let mut arrivals: Vec<i64> = (0..120).map(|i| i * 25_000).collect();
        let held = 3_000_000 + 1_250_000;
        arrivals.extend(std::iter::repeat_n(held, 50));
        arrivals.extend((1..120).map(|i| held + i * 25_000));
        let out = run(1_000_000, &arrivals, 15_000_000);
        assert_eq!(out.len(), arrivals.len());
        let worst = out.windows(2).map(|w| w[1] - w[0]).skip(20).max().unwrap();
        // The buffer covers most of the hold: what shows is well under it.
        assert!(worst < 500_000, "a gap of {worst} us showed");
    }

    #[test]
    fn the_lag_settles_near_the_horizon() {
        let arrivals: Vec<i64> = (0..400).map(|i| i * 25_000).collect();
        let out = run(1_000_000, &arrivals, 30_000_000);
        let late: Vec<i64> = arrivals
            .iter()
            .zip(&out)
            .skip(300)
            .map(|(a, o)| o - a)
            .collect();
        let mean = late.iter().sum::<i64>() / late.len() as i64;
        assert!((700_000..=1_300_000).contains(&mean), "mean lag {mean}");
    }

    #[test]
    fn the_end_and_a_flush_send_everything_in_order() {
        let (tx, rx) = mpsc::channel();
        let p = Playout::start(5_000_000, tx);
        for i in 0..5 {
            assert!(p.send(text(i)));
        }
        assert!(p.send(Event::Stopped));
        p.finish();
        let got: Vec<String> = rx
            .try_iter()
            .map(|e| match e {
                Event::Text(t, _, _, _) => t,
                Event::Stopped => "stopped".into(),
                _ => "other".into(),
            })
            .collect();
        assert_eq!(got, vec!["0", "1", "2", "3", "4", "stopped"]);
    }
}
