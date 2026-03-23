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
//! Claude Code CLI sends bare NDJSON (newline-terminated JSON) and expects
//! NDJSON responses.  Codex sends and expects Content-Length framing.
//!
//! The server mirrors the client: if a message arrives as NDJSON, the response
//! is NDJSON; if it arrives Content-Length-framed, the response is framed.

use std::io::{BufRead, Write};

/// Wire format detected from an incoming message.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    ContentLength,
    Ndjson,
}

/// Read one JSON-RPC message from `reader`.
///
/// Returns `None` on clean EOF.  The `Format` indicates how the message was
/// framed so the caller can reply in kind.
pub fn read_message(reader: &mut impl BufRead) -> std::io::Result<Option<(String, Format)>> {
    loop {
        let mut first_line = String::new();
        match reader.read_line(&mut first_line)? {
            0 => return Ok(None), // EOF
            _ => {}
        }

        let trimmed = first_line.trim();
        if trimmed.is_empty() {
            continue; // skip blank lines between messages
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
            return Ok(Some((
                String::from_utf8(body)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
                Format::ContentLength,
            )));
        } else {
            // ── NDJSON (bare JSON line) ───────────────────────────────────────
            return Ok(Some((trimmed.to_string(), Format::Ndjson)));
        }
    }
}

/// Write one JSON-RPC message to `writer`, mirroring the client's wire format.
///
/// - `Format::ContentLength` → `Content-Length: N\r\n\r\n{json}` (required by Codex)
/// - `Format::Ndjson`        → `{json}\n`                          (required by Claude Code CLI)
pub fn write_message(writer: &mut impl Write, json: &str, format: Format) -> std::io::Result<()> {
    match format {
        Format::ContentLength => write!(writer, "Content-Length: {}\r\n\r\n{}", json.len(), json)?,
        Format::Ndjson => writeln!(writer, "{}", json)?,
    }
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
        let (msg, fmt) = read_message(&mut r).unwrap().unwrap();
        assert_eq!(msg, json);
        assert_eq!(fmt, Format::ContentLength);
    }

    #[test]
    fn reads_ndjson_message() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let input = format!("{}\n", json);
        let mut r = BufReader::new(input.as_bytes());
        let (msg, fmt) = read_message(&mut r).unwrap().unwrap();
        assert_eq!(msg, json);
        assert_eq!(fmt, Format::Ndjson);
    }

    #[test]
    fn skips_blank_lines_between_messages() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let input = format!("\n\n{}\n", json);
        let mut r = BufReader::new(input.as_bytes());
        assert_eq!(read_message(&mut r).unwrap().unwrap().0, json);
    }

    #[test]
    fn reads_multiple_framed_messages_sequentially() {
        let a = r#"{"id":1,"method":"initialize"}"#;
        let b = r#"{"id":2,"method":"tools/list"}"#;
        let mut input = framed(a);
        input.extend(framed(b));
        let mut r = BufReader::new(input.as_slice());
        assert_eq!(read_message(&mut r).unwrap().unwrap().0, a);
        assert_eq!(read_message(&mut r).unwrap().unwrap().0, b);
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
        assert_eq!(read_message(&mut r).unwrap().unwrap().0, json);
    }

    #[test]
    fn returns_none_on_eof() {
        let mut r = BufReader::new(&b""[..]);
        assert!(read_message(&mut r).unwrap().is_none());
    }

    // ── write_message ────────────────────────────────────────────────────────

    #[test]
    fn write_clf_produces_content_length_header() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, json, Format::ContentLength).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(
            s.starts_with(&format!("Content-Length: {}\r\n\r\n", json.len())),
            "expected Content-Length header, got: {:?}",
            &s[..s.len().min(60)]
        );
        assert!(s.ends_with(json));
    }

    #[test]
    fn write_ndjson_produces_newline_terminated_json() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, json, Format::Ndjson).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, format!("{}\n", json));
    }

    #[test]
    fn write_clf_then_read_roundtrip() {
        let json = r#"{"jsonrpc":"2.0","id":42,"result":{"ok":true}}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, json, Format::ContentLength).unwrap();
        let mut r = BufReader::new(buf.as_slice());
        assert_eq!(read_message(&mut r).unwrap().unwrap().0, json);
    }
}
