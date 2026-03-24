use cs_core::capsule::chunk_for_query;

#[test]
fn chunk_for_query_returns_relevant_window() {
    let mut body = String::from("fn process_data(input: &[u8]) -> Vec<u8> {\n");
    for i in 0..20 {
        body.push_str(&format!("    let _pad{i} = unrelated_stuff();\n"));
    }
    body.push_str("    // validate the jwt token here\n");
    body.push_str("    let token = parse_jwt(input);\n");
    body.push_str("    let valid = validate_signature(token);\n");
    for i in 0..20 {
        body.push_str(&format!("    let _end{i} = more_unrelated();\n"));
    }
    body.push('}');

    let chunk = chunk_for_query(&body, "jwt token validate", 500);
    assert!(chunk.contains("jwt"), "chunk should contain query-relevant content");
    assert!(chunk.contains("validate_signature"), "chunk should include the target lines");
}

#[test]
fn chunk_for_query_short_body_unchanged() {
    let body = "fn tiny() {\n    do_thing();\n}";
    let result = chunk_for_query(body, "some query", 1000);
    assert_eq!(result, body);
}
