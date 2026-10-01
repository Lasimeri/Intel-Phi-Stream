//! The terminal's screen, double-buffered. After BF++'s TUI runtime
//! (`__tui_begin` and `__tui_end` in Lasimeri/bfpp): a frame is drawn into a
//! back buffer, compared with what the terminal shows, and only the cells
//! that changed are written. A redraw in which one token arrived writes that
//! token and the lines that changed with it, not the screen. See screen.md.

use std::io::{self, Write};

use crossterm::style::{
    Attribute, Color, Print, SetAttribute, SetBackgroundColor, SetForegroundColor,
};
use crossterm::{cursor, queue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Plain,
    Bold,
    Italic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub weight: Weight,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    ch: char,
    style: Style,
}

/// The cell a wide character's second column holds: the terminal draws
/// the character over both, so nothing is written for it.
const TAIL: char = '\0';

/// Terminal columns a character takes: 2 for wide ones (CJK, which the
/// stream's tokens hold), 0 for combining marks and other zero-width
/// characters, 1 otherwise (control characters, shown as spaces).
pub fn columns(ch: char) -> usize {
    use unicode_width::UnicodeWidthChar;
    if ch.is_control() {
        1
    } else {
        ch.width().unwrap_or(1)
    }
}

/// A text's width in terminal columns.
pub fn text_columns(s: &str) -> usize {
    s.chars().map(columns).sum()
}

/// A grid of cells, one terminal column each; a wide character takes two
/// (itself, then `TAIL`). Counting characters instead put a row of CJK
/// tokens past the right edge, onto the next row (seen on the live mind
/// strip).
pub struct Screen {
    w: usize,
    h: usize,
    cells: Vec<Cell>,
}

impl Screen {
    /// A screen of spaces in `style`.
    pub fn new(w: usize, h: usize, style: Style) -> Self {
        Self {
            w,
            h,
            cells: vec![Cell { ch: ' ', style }; w * h],
        }
    }

    /// `text` at `row`, `col`, clipped at the right edge (control
    /// characters shown as spaces, zero-width ones dropped, a wide one that
    /// does not fit shown as a space); the column after it.
    pub fn put(&mut self, row: usize, col: usize, text: &str, style: Style) -> usize {
        let mut c = col;
        if row >= self.h {
            return c;
        }
        for ch in text.chars() {
            if c >= self.w {
                break;
            }
            let ch = if ch.is_control() { ' ' } else { ch };
            match columns(ch) {
                0 => {}
                2 if c + 1 < self.w => {
                    self.set(row, c, Cell { ch, style });
                    self.set(row, c + 1, Cell { ch: TAIL, style });
                    c += 2;
                }
                2 => {
                    self.set(row, c, Cell { ch: ' ', style });
                    c += 1;
                }
                _ => {
                    self.set(row, c, Cell { ch, style });
                    c += 1;
                }
            }
        }
        c
    }

    /// One cell, keeping wide characters whole: a wide character half
    /// overwritten leaves a space in its other half.
    fn set(&mut self, row: usize, c: usize, cell: Cell) {
        let i = row * self.w + c;
        let old = self.cells[i];
        if old.ch == TAIL && cell.ch != TAIL && c > 0 {
            self.cells[i - 1].ch = ' ';
        }
        if old.ch != TAIL && columns(old.ch) == 2 && c + 1 < self.w && self.cells[i + 1].ch == TAIL
        {
            self.cells[i + 1].ch = ' ';
        }
        self.cells[i] = cell;
    }

    /// Spaces in `style` from `col` to the end of `row`.
    pub fn fill(&mut self, row: usize, col: usize, style: Style) {
        if row >= self.h {
            return;
        }
        for c in col.min(self.w)..self.w {
            self.set(row, c, Cell { ch: ' ', style });
        }
    }

    /// A whole row: `text`, then spaces, in one style.
    pub fn line(&mut self, row: usize, text: &str, style: Style) {
        let c = self.put(row, 0, text, style);
        self.fill(row, c, style);
    }

    /// Queue on `out` what makes a terminal showing `front` (none: unknown,
    /// or another size) show this screen: runs of changed cells, each with a
    /// cursor move, the style set only when it changes.
    pub fn diff(&self, front: Option<&Screen>, out: &mut impl Write) -> io::Result<()> {
        let front = front.filter(|f| f.w == self.w && f.h == self.h);
        let differs = |i: usize| front.is_none_or(|f| f.cells[i] != self.cells[i]);
        // A wide character counts as changed when its second column did, so
        // a run never begins on a second column (it is written by the
        // character).
        let changed = |i: usize| {
            differs(i)
                || (i % self.w + 1 < self.w && self.cells[i + 1].ch == TAIL && differs(i + 1))
        };
        let mut cur: Option<Style> = None;
        let mut run = String::new();
        for row in 0..self.h {
            let mut col = 0;
            while col < self.w {
                if !changed(row * self.w + col) {
                    col += 1;
                    continue;
                }
                queue!(out, cursor::MoveTo(col as u16, row as u16))?;
                while col < self.w && changed(row * self.w + col) {
                    let c = self.cells[row * self.w + col];
                    if cur != Some(c.style) {
                        if !run.is_empty() {
                            queue!(out, Print(&run))?;
                            run.clear();
                        }
                        set_style(out, c.style)?;
                        cur = Some(c.style);
                    }
                    // The wide character before it covers its second column.
                    if c.ch != TAIL {
                        run.push(c.ch);
                    }
                    col += 1;
                }
                queue!(out, Print(&run))?;
                run.clear();
            }
        }
        Ok(())
    }
}

fn set_style(out: &mut impl Write, s: Style) -> io::Result<()> {
    queue!(out, SetAttribute(Attribute::Reset))?;
    match s.weight {
        Weight::Plain => {}
        Weight::Bold => queue!(out, SetAttribute(Attribute::Bold))?,
        Weight::Italic => queue!(out, SetAttribute(Attribute::Italic))?,
    }
    queue!(out, SetBackgroundColor(s.bg), SetForegroundColor(s.fg))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Style = Style {
        fg: Color::White,
        bg: Color::Black,
        weight: Weight::Plain,
    };
    const B: Style = Style {
        fg: Color::Yellow,
        bg: Color::Black,
        weight: Weight::Bold,
    };

    fn bytes(s: &Screen, front: Option<&Screen>) -> Vec<u8> {
        let mut v = Vec::new();
        s.diff(front, &mut v).unwrap();
        v
    }

    fn frame(text: &str) -> Screen {
        let mut s = Screen::new(80, 24, S);
        for r in 0..24 {
            s.line(r, &format!("row {r} {text}"), S);
        }
        s
    }

    #[test]
    fn an_unchanged_frame_writes_nothing() {
        assert!(bytes(&frame("a"), Some(&frame("a"))).is_empty());
    }

    #[test]
    fn one_changed_word_writes_that_word() {
        let mut b = frame("a");
        b.put(10, 20, "token", B);
        let d = bytes(&b, Some(&frame("a")));
        let full = bytes(&b, None);
        let s = String::from_utf8_lossy(&d);
        assert!(s.contains("token") && !s.contains("row 3"), "{s:?}");
        assert!(d.len() < 80, "{} bytes", d.len());
        assert!(full.len() > 24 * 80, "{} bytes", full.len());
    }

    #[test]
    fn another_size_or_no_front_redraws_everything() {
        let a = frame("a");
        let small = Screen::new(10, 2, S);
        assert_eq!(bytes(&a, Some(&small)), bytes(&a, None));
    }

    #[test]
    fn text_is_clipped_and_controls_blanked() {
        let mut s = Screen::new(5, 1, S);
        assert_eq!(s.put(0, 3, "a\nbcd", S), 5);
        let d = String::from_utf8_lossy(&bytes(&s, Some(&Screen::new(5, 1, S)))).into_owned();
        assert!(
            d.contains('a') && !d.contains('b') && !d.contains('\n'),
            "{d:?}"
        );
    }

    /// One row's cells as text: a second column as `_`.
    fn row_text(s: &Screen) -> String {
        s.cells
            .iter()
            .map(|c| if c.ch == TAIL { '_' } else { c.ch })
            .collect()
    }

    #[test]
    fn wide_characters_take_two_columns_and_stay_in_the_row() {
        // The live mind strip: CJK tokens among Latin ones. "ab " is 3
        // columns, three wide characters 6, a space 1: the next wide
        // character takes 11 and 12, and nothing more fits.
        let mut s = Screen::new(12, 1, S);
        assert_eq!(s.put(0, 0, "ab 而不是 而非 x", S), 12);
        assert_eq!(row_text(&s), "ab 而_不_是_ 而_");
        let d = bytes(&s, None);
        assert!(!d.contains(&0), "a second column is never written");
        let printed: String = String::from_utf8_lossy(&d)
            .chars()
            .filter(|c| !c.is_ascii_control())
            .collect();
        assert!(printed.contains("ab 而不是 而"), "{printed:?}");
    }

    #[test]
    fn a_wide_character_that_does_not_fit_is_a_space() {
        let mut s = Screen::new(5, 1, S);
        assert_eq!(s.put(0, 4, "是", S), 5);
        assert_eq!(row_text(&s), "     ");
    }

    #[test]
    fn half_overwritten_wide_characters_leave_spaces() {
        let mut s = Screen::new(6, 1, S);
        s.put(0, 0, "而不是", S);
        // Over the second column of the first, and the first of the last.
        s.put(0, 1, "x", S);
        s.put(0, 4, "y", S);
        assert_eq!(row_text(&s), " x不_y ");
    }

    #[test]
    fn a_changed_second_column_rewrites_its_character() {
        let mut a = Screen::new(6, 1, S);
        a.put(0, 0, "而不是", S);
        let mut b = Screen::new(6, 1, S);
        b.put(0, 0, "而不是", S);
        b.cells[3].style = B;
        let d = String::from_utf8_lossy(&bytes(&b, Some(&a))).to_string();
        assert!(d.contains('不'), "{d:?}");
    }

    #[test]
    fn zero_width_characters_take_no_column() {
        let mut s = Screen::new(4, 1, S);
        // e and a combining acute accent, then x.
        assert_eq!(s.put(0, 0, "e\u{301}x", S), 2);
        assert_eq!(text_columns("e\u{301}x"), 2);
    }
}
