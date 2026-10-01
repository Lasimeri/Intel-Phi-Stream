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

/// A grid of cells, one character each (the terminal's columns are
/// counted in characters, as everywhere in the terminal).
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
    /// characters shown as spaces); the column after it.
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
            self.cells[row * self.w + c] = Cell { ch, style };
            c += 1;
        }
        c
    }

    /// Spaces in `style` from `col` to the end of `row`.
    pub fn fill(&mut self, row: usize, col: usize, style: Style) {
        if row >= self.h {
            return;
        }
        for c in col.min(self.w)..self.w {
            self.cells[row * self.w + c] = Cell { ch: ' ', style };
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
        let changed = |i: usize| front.is_none_or(|f| f.cells[i] != self.cells[i]);
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
                    run.push(c.ch);
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
}
