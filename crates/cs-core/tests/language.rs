use cs_core::language::detect_language;
use std::path::PathBuf;

#[test]
fn detects_python() {
    assert_eq!(
        detect_language(&PathBuf::from("foo.py")),
        Some(cs_core::language::Language::Python)
    );
}

#[test]
fn detects_tsx() {
    assert_eq!(
        detect_language(&PathBuf::from("App.tsx")),
        Some(cs_core::language::Language::Tsx)
    );
}

#[test]
fn detects_rust() {
    assert_eq!(
        detect_language(&PathBuf::from("main.rs")),
        Some(cs_core::language::Language::Rust)
    );
}
