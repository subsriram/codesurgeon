use cs_core::language::Language;
use cs_core::skeletonizer::skeletonize;
use cs_core::symbol::{Symbol, SymbolKind};

fn make_sym(kind: SymbolKind, signature: &str, body: &str) -> Symbol {
    Symbol::new(
        "test.py",
        "test_fn",
        kind,
        1,
        10,
        signature.to_string(),
        Some("Does something.".to_string()),
        body.to_string(),
        Language::Python,
    )
}

#[test]
fn callable_skeleton_has_no_body() {
    let sym = make_sym(
        SymbolKind::Function,
        "def compute(x: int) -> int:",
        "def compute(x: int) -> int:\n    result = x * 2\n    return result\n",
    );
    let skel = skeletonize(&sym);
    assert!(skel.contains("compute"));
    assert!(!skel.contains("result = x"));
}
