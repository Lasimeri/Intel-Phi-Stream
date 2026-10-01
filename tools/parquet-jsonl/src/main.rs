//! `parquet-jsonl IN.parquet > OUT.jsonl`: every row of a parquet file as
//! one JSON object per line, columns by name, through the parquet
//! crate's record reader. See main.md.

use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{Context as _, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: parquet-jsonl IN.parquet > OUT.jsonl")?;
    let file = File::open(&path).with_context(|| format!("opening {path}"))?;
    let reader = SerializedFileReader::new(file).with_context(|| format!("{path} is not a parquet file"))?;
    let rows = reader.get_row_iter(None)?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut n = 0usize;
    for row in rows {
        let row = row?;
        writeln!(out, "{}", serde_json::to_string(&row.to_json_value())?)?;
        n += 1;
    }
    out.flush()?;
    eprintln!("{path}: {n} rows");
    Ok(())
}
