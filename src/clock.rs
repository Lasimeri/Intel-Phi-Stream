//! The real-time clock the stream is kept against: every token, reading
//! and event is stamped in microseconds of `CLOCK_REALTIME` at the moment
//! it exists, and durations are taken on `CLOCK_MONOTONIC`, so the chain
//! of thought is placed on the wall clock rather than counted in cycles of
//! whatever hardware runs it. See clock.md.

/// Microseconds since the Unix epoch, `CLOCK_REALTIME` (the system's
/// wall clock, as kept by NTP or whatever disciplines it).
pub fn now_us() -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a plain call with a valid out-pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
    ts.tv_sec * 1_000_000 + ts.tv_nsec / 1_000
}

/// Microseconds of `CLOCK_MONOTONIC`: never steps, for durations.
pub fn mono_us() -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: as above.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec * 1_000_000 + ts.tv_nsec / 1_000
}

/// The local broken-down time of `t_us` and its microseconds.
fn local(t_us: i64) -> (libc::tm, i64) {
    let secs = t_us.div_euclid(1_000_000) as libc::time_t;
    let micros = t_us.rem_euclid(1_000_000);
    // SAFETY: localtime_r fills `tm` from a valid time_t; tm is plain data.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&secs, &mut tm) };
    (tm, micros)
}

/// The local zone at `t_us`: its abbreviation (`CDT`) and its offset from
/// UTC in seconds, as the C library has them (TZ, else /etc/localtime).
pub fn zone(t_us: i64) -> (String, i64) {
    let (tm, _) = local(t_us);
    let name = if tm.tm_zone.is_null() {
        String::new()
    } else {
        // SAFETY: localtime_r points tm_zone at a static NUL-terminated name.
        unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
            .to_string_lossy()
            .into_owned()
    };
    (name, tm.tm_gmtoff)
}

/// `HH:MM:SS.uuuuuu`, local time.
pub fn hms(t_us: i64) -> String {
    let (tm, us) = local(t_us);
    format!(
        "{:02}:{:02}:{:02}.{:06}",
        tm.tm_hour, tm.tm_min, tm.tm_sec, us
    )
}

/// `YYYY-MM-DD HH:MM:SS.uuuuuu ZONE`, local time.
pub fn datetime(t_us: i64) -> String {
    let (tm, us) = local(t_us);
    let zone = if tm.tm_zone.is_null() {
        String::new()
    } else {
        // SAFETY: localtime_r points tm_zone at a static NUL-terminated name.
        unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
            .to_string_lossy()
            .into_owned()
    };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:06} {zone}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        us
    )
    .trim_end()
    .to_string()
}

/// A duration in words a mind reads easily: `0.42 s`, `41.2 s`,
/// `3 min 12 s`, `2 h 5 min`.
pub fn span(us: i64) -> String {
    let us = us.max(0);
    let s = us as f64 / 1e6;
    if s < 10.0 {
        format!("{s:.2} s")
    } else if s < 60.0 {
        format!("{s:.1} s")
    } else if s < 3600.0 {
        let m = (s / 60.0).floor() as i64;
        format!("{m} min {} s", (s - m as f64 * 60.0).floor() as i64)
    } else {
        let h = (s / 3600.0).floor() as i64;
        format!(
            "{h} h {} min",
            ((s - h as f64 * 3600.0) / 60.0).floor() as i64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clocks_advance_in_microseconds() {
        let a = now_us();
        let m = mono_us();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(now_us() - a >= 1_900);
        assert!(mono_us() - m >= 1_900);
        // 2026 is about 1.79e15 microseconds after the epoch.
        assert!(a > 1_700_000_000_000_000);
    }

    #[test]
    fn times_print_with_microseconds() {
        let t = now_us();
        let h = hms(t);
        assert_eq!(h.len(), 15, "{h}");
        assert_eq!(&h[8..9], ".");
        assert_eq!(h[9..].parse::<i64>().unwrap(), t.rem_euclid(1_000_000));
        assert!(datetime(t).contains(&h));
    }

    #[test]
    fn spans_read_as_words() {
        assert_eq!(span(420_000), "0.42 s");
        assert_eq!(span(41_200_000), "41.2 s");
        assert_eq!(span(192_000_000), "3 min 12 s");
        assert_eq!(span(7_500_000_000), "2 h 5 min");
    }
}
