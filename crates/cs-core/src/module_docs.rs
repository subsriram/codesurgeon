//! Module documentation generation and Swift enrichment helpers.

/// Probe for Xcode 26+ MCP bridge availability. Result is cached after the first call
/// so repeated `run_pipeline` or `index_status` calls pay the subprocess cost only once.
pub(crate) fn detect_xcode_mcp() -> bool {
    use std::sync::OnceLock;
    static XCODE_MCP: OnceLock<bool> = OnceLock::new();
    *XCODE_MCP.get_or_init(|| {
        std::process::Command::new("xcrun")
            .args(["--find", "mcpbridge"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Human-readable hint appended to capsule output when Swift symbols are present.
/// Tells the agent which enrichment path is available and what the fallback is,
/// so it never silently operates on incomplete type information.
pub(crate) fn swift_enrichment_hint(xcode_mcp_available: bool) -> String {
    if xcode_mcp_available {
        "\n> **Swift symbols detected.** \
         Xcode MCP is available — call its tools for resolved types and live build diagnostics. \
         codesurgeon results reflect tree-sitter parsing and remain available for semantic search \
         and session memory.\n"
            .to_string()
    } else {
        "\n> **Swift symbols detected.** \
         Xcode MCP was not found — results are based on tree-sitter parsing only (no resolved types, \
         no macro-expanded symbols). \
         To enable full Swift enrichment: install Xcode 26+ and turn on \
         Settings → Intelligence → Enable Model Context Protocol, \
         then wire it up with `xcrun mcpbridge`. \
         codesurgeon's graph is still usable for semantic search and session memory.\n"
            .to_string()
    }
}
