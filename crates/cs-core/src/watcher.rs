use anyhow::Result;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

/// File change event emitted to the engine.
#[derive(Debug)]
pub struct FileChangeEvent {
    pub path: PathBuf,
    pub kind: ChangeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    Created,
    Modified,
    Removed,
}

/// Watches a workspace directory for file changes.
/// Emits `FileChangeEvent`s on a channel.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    pub receiver: Receiver<FileChangeEvent>,
}

impl FileWatcher {
    pub fn new(workspace_root: &Path) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<FileChangeEvent>();

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<Event>| match res {
                Ok(event) => {
                    let kind = match event.kind {
                        EventKind::Create(_) => Some(ChangeKind::Created),
                        EventKind::Modify(_) => Some(ChangeKind::Modified),
                        EventKind::Remove(_) => Some(ChangeKind::Removed),
                        _ => None,
                    };

                    if let Some(kind) = kind {
                        for path in event.paths {
                            if should_watch(&path) {
                                let _ = tx.send(FileChangeEvent {
                                    path,
                                    kind: kind.clone(),
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("File watcher error: {}", e);
                }
            })?;

        watcher.watch(workspace_root, RecursiveMode::Recursive)?;

        Ok(FileWatcher {
            _watcher: watcher,
            receiver: rx,
        })
    }

    /// Poll for pending events with a timeout.
    pub fn poll(&self, timeout: Duration) -> Vec<FileChangeEvent> {
        let mut events = Vec::new();
        // Wait for first event, then drain remaining non-blocking
        if let Ok(ev) = self.receiver.recv_timeout(timeout) {
            events.push(ev);
            while let Ok(ev) = self.receiver.try_recv() {
                events.push(ev);
            }
        }
        deduplicate_events(events)
    }
}

/// Compute blake3 hash of file content for change detection.
pub fn hash_content(content: &[u8]) -> String {
    blake3::hash(content).to_hex().to_string()
}

/// Check if a file should be watched (based on extension and path).
fn should_watch(path: &Path) -> bool {
    // Skip hidden directories
    for component in path.components() {
        if let std::path::Component::Normal(c) = component {
            if c.to_string_lossy().starts_with('.') {
                return false;
            }
        }
    }

    // Skip common noise directories
    for component in path.components() {
        let s = component.as_os_str().to_string_lossy();
        if matches!(
            s.as_ref(),
            "node_modules" | "target" | "__pycache__" | ".git" | "dist" | "build" | ".build"
        ) {
            return false;
        }
    }

    // Only watch known source file extensions
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        matches!(
            ext,
            "py" | "pyw"
                | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "mjs"
                | "cjs"
                | "sh"
                | "bash"
                | "html"
                | "htm"
                | "rs"
                | "swift"
                | "sql"
                | "md"
                | "mdx"
        )
    } else {
        false
    }
}

/// Remove duplicate events for the same path, keeping the most recent.
fn deduplicate_events(events: Vec<FileChangeEvent>) -> Vec<FileChangeEvent> {
    let mut seen: std::collections::HashMap<PathBuf, ChangeKind> = std::collections::HashMap::new();
    for ev in events {
        seen.insert(ev.path, ev.kind);
    }
    seen.into_iter()
        .map(|(path, kind)| FileChangeEvent { path, kind })
        .collect()
}
