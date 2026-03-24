//! Diff parsing helpers for `get_diff_capsule`.
//!
//! Parses unified diff output into `(file_path, start_line, end_line)` triples
//! that the engine can map back to indexed symbols.

/// Parse a unified diff and return (file_path, start_line, end_line) for each changed hunk.
pub(crate) fn parse_diff_symbols(diff: &str) -> Vec<(String, u32, u32)> {
    let mut result = Vec::new();
    let mut current_file = String::new();
    let mut hunk_start = 0u32;
    let mut hunk_end = 0u32;

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            // Flush previous hunk
            if !current_file.is_empty() && hunk_end >= hunk_start {
                result.push((current_file.clone(), hunk_start, hunk_end));
            }
            current_file = rest.trim().to_string();
            hunk_start = 0;
            hunk_end = 0;
        } else if line.starts_with("@@ ") {
            // Flush previous hunk for this file
            if !current_file.is_empty() && hunk_end >= hunk_start && hunk_start > 0 {
                result.push((current_file.clone(), hunk_start, hunk_end));
            }
            // Parse "@@ -old_start,old_len +new_start,new_len @@"
            // We care about the new file's line range (+new_start,new_len)
            if let Some((start, len)) = parse_hunk_header(line) {
                hunk_start = start;
                hunk_end = start + len.saturating_sub(1);
            }
        }
    }

    // Flush last hunk
    if !current_file.is_empty() && hunk_end >= hunk_start && hunk_start > 0 {
        result.push((current_file, hunk_start, hunk_end));
    }

    result
}

fn parse_hunk_header(line: &str) -> Option<(u32, u32)> {
    // "@@ -a,b +c,d @@" — extract c and d
    let plus_part = line.split('+').nth(1)?;
    let range_part = plus_part.split(' ').next()?;
    let mut parts = range_part.splitn(2, ',');
    let start: u32 = parts.next()?.parse().ok()?;
    let len: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    Some((start, len))
}
