//! MCP stdio transport — framing helpers shared by the server loop and unit tests.
//!
//! Wire format
//! -----------
//! The MCP spec (and Codex) require LSP-style Content-Length framing:
//!
//!   Content-Length: <N>\r\n
//!   \r\n
//!   <N bytes of UTF-8 JSON>
//!
//! As a convenience, bare NDJSON (newline-terminated JSON with no headers) is
//! also accepted on the *read* side so Claude Code keeps working.
//! All *writes* are framed so both clients are happy.

use std::io::{BufRead, Write};

/// Read one JSON-RPC message from `reader`.
///
/// Returns `None` on clean EOF.  Errors are returned as `Err`.
pub fn read_message(reader: &mut impl BufRead) -> std::io::Result<Option<String>> {
    loop {
        let mut first_line = String::new();
        match reader.read_line(&mut first_line)? {
            0 => return Ok(None),  // EOF
            _ => {}
        }

        let trimmed = first_line.trim();
        if trimmed.is_empty() {
            continue;  // skip blank lines between messages
        }

        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            // ── LSP-framed ────────────────────────────────────────────────────
            let len: usize = rest.trim().parse().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid Content-Length: {}", rest.trim()),
                )
            })?;

            // Consume remaining headers until the mandatory blank separator line.
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h)? == 0 {
                    break;
                }
                if h.trim().is_empty() {
                    break;
                }
            }

            let mut body = vec![0u8; len];
            reader.read_exact(&mut body)?;
            return Ok(Some(
                String::from_utf8(body).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?,
            ));
        } else {
            // ── NDJSON (bare JSON line) ───────────────────────────────────────
            return Ok(Some(trimmed.to_string()));
        }
    }
}

/// Write one JSON-RPC message to `writer` using Content-Length framing.
pub fn write_message(writer: &mut impl Write, json: &str) -> std::io::Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", json.len(), json)?;
    writer.flush()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    fn framed(s: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", s.len(), s).into_bytes()
    }

    // ── read_message ─────────────────────────────────────────────────────────

    #[test]
    fn reads_framed_message() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let bytes = framed(json);
        let mut r = BufReader::new(bytes.as_slice());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), json);
    }

    #[test]
    fn reads_ndjson_message() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let input = format!("{}\n", json);
        let mut r = BufReader::new(input.as_bytes());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), json);
    }

    #[test]
    fn skips_blank_lines_between_messages() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let input = format!("\n\n{}\n", json);
        let mut r = BufReader::new(input.as_bytes());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), json);
    }

    #[test]
    fn reads_multiple_framed_messages_sequentially() {
        let a = r#"{"id":1,"method":"initialize"}"#;
        let b = r#"{"id":2,"method":"tools/list"}"#;
        let mut input = framed(a);
        input.extend(framed(b));
        let mut r = BufReader::new(input.as_slice());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), a);
        assert_eq!(read_message(&mut r).unwrap().unwrap(), b);
    }

    #[test]
    fn ignores_extra_headers_before_blank_line() {
        let json = r#"{"id":1,"method":"ping"}"#;
        let raw = format!(
            "Content-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            json.len(),
            json
        );
        let mut r = BufReader::new(raw.as_bytes());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), json);
    }

    #[test]
    fn returns_none_on_eof() {
        let mut r = BufReader::new(&b""[..]);
        assert!(read_message(&mut r).unwrap().is_none());
    }

    // ── write_message ────────────────────────────────────────────────────────

    #[test]
    fn write_produces_content_length_header() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, json).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(
            s.starts_with(&format!("Content-Length: {}\r\n\r\n", json.len())),
            "expected Content-Length header, got: {:?}",
            &s[..s.len().min(60)]
        );
        assert!(s.ends_with(json));
    }

    #[test]
    fn write_then_read_roundtrip() {
        let json = r#"{"jsonrpc":"2.0","id":42,"result":{"ok":true}}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, json).unwrap();
        let mut r = BufReader::new(buf.as_slice());
        assert_eq!(read_message(&mut r).unwrap().unwrap(), json);
    }
}
