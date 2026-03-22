//! Integration tests for the MCP stdio transport.
//!
//! Each test spawns the real `codesurgeon-mcp` binary against a throwaway
//! tempdir workspace and drives it over stdin/stdout exactly as Codex does —
//! Content-Length-framed JSON-RPC 2.0 messages.
//!
//! These tests guard the invariants that have been broken by accident:
//!   • `jsonrpc: "2.0"` must appear in every response
//!   • Responses must be Content-Length framed
//!   • `initialize` must advertise `resources` capability
//!   • `resources/list` and `resources/templates/list` must return empty lists
//!   • A second simultaneous connection must NOT be killed by the PID lock
//!
//! Run:  cargo test -p cs-mcp --test mcp_protocol

use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

// Cargo sets this env var for integration tests in the same package as the binary.
const BIN: &str = env!("CARGO_BIN_EXE_codesurgeon-mcp");

// ── Wire helpers ──────────────────────────────────────────────────────────────

fn encode_framed(msg: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{}", msg.len(), msg).into_bytes()
}

/// Read one Content-Length-framed message from `reader`.
/// Panics with a descriptive message if the framing is malformed.
fn decode_framed(reader: &mut BufReader<ChildStdout>) -> Value {
    let mut content_length: Option<usize> = None;

    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .expect("failed to read from server stdout");
        assert!(!line.is_empty(), "server closed stdout unexpectedly");

        let trimmed = line.trim();
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                rest.trim()
                    .parse()
                    .expect("Content-Length value is not a number"),
            );
        }
    }

    let len = content_length.expect("response had no Content-Length header");
    assert!(len > 0, "Content-Length was zero");

    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .expect("failed to read response body");

    serde_json::from_slice(&body).expect("response body is not valid JSON")
}

// ── Session ───────────────────────────────────────────────────────────────────

struct Session {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Keep the tempdir alive for the lifetime of the session.
    _dir: TempDir,
}

impl Session {
    fn new_in(dir: &TempDir) -> Self {
        let mut child = Command::new(BIN)
            .env("CS_WORKSPACE", dir.path())
            .env("CS_LOG", "error")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn codesurgeon-mcp");

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        // We need a _dir field but the caller owns the real dir — use a dummy.
        Self {
            child,
            stdin,
            stdout,
            _dir: tempfile::tempdir().unwrap(),
        }
    }

    fn send(&mut self, msg: &str) {
        self.stdin
            .write_all(&encode_framed(msg))
            .expect("write to server stdin failed");
        self.stdin.flush().expect("flush to server stdin failed");
    }

    fn recv(&mut self) -> Value {
        decode_framed(&mut self.stdout)
    }

    /// Perform the initialize / notifications/initialized handshake and
    /// return the `initialize` response.
    fn handshake(&mut self) -> Value {
        self.send(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
                "protocolVersion":"2024-11-05",
                "capabilities":{},
                "clientInfo":{"name":"test","version":"0"}
            }}"#,
        );
        let resp = self.recv();
        // The initialized notification has no response; just fire and forget.
        self.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#);
        resp
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── Invariant tests ───────────────────────────────────────────────────────────

/// Every response must carry `"jsonrpc": "2.0"`.
/// This field was accidentally removed during a refactor — see CLAUDE.md invariants.
#[test]
fn initialize_response_has_jsonrpc_field() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    let resp = s.handshake();
    assert_eq!(
        resp["jsonrpc"].as_str(),
        Some("2.0"),
        "initialize response missing jsonrpc field: {resp}"
    );
}

/// Responses must be Content-Length framed (decode_framed panics if they aren't).
/// Codex drops the connection if responses are bare NDJSON.
#[test]
fn responses_are_content_length_framed() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    // decode_framed will panic with a descriptive error if the response is not framed.
    let _resp = s.handshake();
}

/// `initialize` must advertise `resources: {}` so Codex knows to probe resource methods.
#[test]
fn initialize_advertises_resources_capability() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    let resp = s.handshake();
    assert!(
        resp["result"]["capabilities"]["resources"].is_object(),
        "initialize response missing capabilities.resources: {resp}"
    );
}

/// `resources/list` must return an empty resources array, not a -32601 error.
/// Codex probes this method; a -32601 causes it to report "MCP startup failed".
#[test]
fn resources_list_returns_empty_array() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    s.handshake();
    s.send(r#"{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{}}"#);
    let resp = s.recv();
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert!(resp["error"].is_null(), "resources/list returned error: {resp}");
    assert!(
        resp["result"]["resources"].is_array(),
        "resources/list result missing 'resources' array: {resp}"
    );
}

/// `resources/templates/list` must return an empty array — same reasoning as above.
#[test]
fn resources_templates_list_returns_empty_array() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    s.handshake();
    s.send(
        r#"{"jsonrpc":"2.0","id":2,"method":"resources/templates/list","params":{}}"#,
    );
    let resp = s.recv();
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert!(resp["error"].is_null(), "resources/templates/list returned error: {resp}");
    assert!(
        resp["result"]["resourceTemplates"].is_array(),
        "resources/templates/list result missing 'resourceTemplates': {resp}"
    );
}

/// A second connection to the same workspace must NOT be killed by the PID lock.
///
/// Codex spawns two processes simultaneously when it probes resources/list and
/// resources/templates/list. The old code called exit(0) on the second instance,
/// causing "connection closed: initialize response".
#[test]
fn parallel_connections_both_complete_handshake() {
    let dir = tempfile::tempdir().unwrap();

    // Start the primary instance and complete its handshake so the PID file is written.
    let mut primary = Session::new_in(&dir);
    let resp1 = primary.handshake();
    assert_eq!(resp1["jsonrpc"].as_str(), Some("2.0"), "primary: {resp1}");

    // Give the OS a moment to flush the PID file to disk.
    std::thread::sleep(Duration::from_millis(50));

    // Start a secondary instance against the same workspace.
    // It must NOT exit — it should still respond to initialize.
    let mut secondary = Session::new_in(&dir);
    let resp2 = secondary.handshake();
    assert_eq!(
        resp2["jsonrpc"].as_str(),
        Some("2.0"),
        "secondary instance was killed by PID lock (got: {resp2})"
    );
}

/// Bare NDJSON input (no Content-Length headers) must also be accepted.
/// Claude Code sends messages this way.
#[test]
fn ndjson_input_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(BIN)
        .env("CS_WORKSPACE", dir.path())
        .env("CS_LOG", "error")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Send NDJSON — no Content-Length header.
    let msg = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#;
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();

    // Response must still be framed.
    let resp = decode_framed(&mut stdout);
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert!(resp["error"].is_null(), "initialize via NDJSON returned error: {resp}");

    let _ = child.kill();
    let _ = child.wait();
}

/// `ping` must return an empty result (not an error).
#[test]
fn ping_returns_ok() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    s.handshake();
    s.send(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}"#);
    let resp = s.recv();
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert!(resp["error"].is_null(), "ping returned error: {resp}");
}

/// Unknown methods must return -32601, not crash or close the connection.
#[test]
fn unknown_method_returns_minus_32601() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    s.handshake();
    s.send(r#"{"jsonrpc":"2.0","id":2,"method":"nonexistent/method","params":{}}"#);
    let resp = s.recv();
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert_eq!(
        resp["error"]["code"].as_i64(),
        Some(-32601),
        "expected -32601 for unknown method, got: {resp}"
    );
}

/// `tools/list` must return a non-empty tools array.
#[test]
fn tools_list_returns_tools() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new_in(&dir);
    s.handshake();
    s.send(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#);
    let resp = s.recv();
    assert_eq!(resp["jsonrpc"].as_str(), Some("2.0"), "{resp}");
    assert!(resp["error"].is_null(), "tools/list returned error: {resp}");
    let tools = resp["result"]["tools"].as_array().expect("no tools array");
    assert!(!tools.is_empty(), "tools/list returned an empty array");
}
