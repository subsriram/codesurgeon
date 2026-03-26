use cs_core::language::{detect_language, Language};
use std::path::PathBuf;

#[test]
fn detects_python() {
    assert_eq!(
        detect_language(&PathBuf::from("foo.py")),
        Some(cs_core::language::Language::Python)
    );
}

/// `.pyi` stub files (Python type stubs) must be detected as Python.
#[test]
fn detects_pyi_as_python() {
    assert_eq!(
        detect_language(&PathBuf::from("requests.pyi")),
        Some(Language::Python)
    );
}

/// `.swiftinterface` stub files must be detected as Swift.
#[test]
fn detects_swiftinterface_as_swift() {
    assert_eq!(
        detect_language(&PathBuf::from("Foundation.swiftinterface")),
        Some(Language::Swift)
    );
}

/// `.d.ts` files have the `.ts` extension and must be detected as TypeScript.
#[test]
fn detects_dts_as_typescript() {
    assert_eq!(
        detect_language(&PathBuf::from("index.d.ts")),
        Some(Language::TypeScript)
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
