use std::fs;
use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use cs_core::{CoreEngine, EngineConfig};
use tempfile::TempDir;

/// Path to the `cs-core` source tree, used as a deterministic fixed corpus.
fn corpus_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Copy the corpus into a fresh tempdir so each benchmark iteration starts
/// from a clean workspace with no `.codesurgeon/` state.
fn make_corpus() -> TempDir {
    let td = tempfile::Builder::new()
        .prefix("cs-bench-")
        .tempdir()
        .expect("tempdir");
    copy_dir(&corpus_source(), &td.path().join("src"));
    td
}

fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create_dir_all");
    for entry in fs::read_dir(src).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let ty = entry.file_type().expect("file_type");
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir(&entry.path(), &to);
        } else if ty.is_file() {
            fs::copy(entry.path(), to).expect("copy");
        }
    }
}

fn make_engine(workspace: &Path) -> CoreEngine {
    let cfg = EngineConfig::new(workspace).without_embedder();
    CoreEngine::new(cfg).expect("engine")
}

/// Cold index: fresh workspace, no prior state. Measures parse + symbol
/// extract + edge build + SQLite write on the cs-core source corpus.
fn bench_index_cold(c: &mut Criterion) {
    c.bench_function("index_cold", |b| {
        b.iter_batched(
            make_corpus,
            |corpus| {
                let engine = make_engine(corpus.path());
                engine.index_workspace().expect("index");
                corpus
            },
            BatchSize::LargeInput,
        );
    });
}

/// Warm index: second pass against an already-indexed workspace with no file
/// changes. Exercises the hash-match incremental skip path.
fn bench_index_warm(c: &mut Criterion) {
    c.bench_function("index_warm", |b| {
        b.iter_batched(
            || {
                let corpus = make_corpus();
                let engine = make_engine(corpus.path());
                engine.index_workspace().expect("initial index");
                (corpus, engine)
            },
            |(corpus, engine)| {
                engine.index_workspace().expect("warm index");
                (corpus, engine)
            },
            BatchSize::LargeInput,
        );
    });
}

/// run_pipeline: capsule assembly against a pre-built index. Shared engine
/// across iterations since the pipeline is read-only.
fn bench_run_pipeline(c: &mut Criterion) {
    let corpus = make_corpus();
    let engine = make_engine(corpus.path());
    engine.index_workspace().expect("prewarm index");

    let mut group = c.benchmark_group("run_pipeline");
    for query in &[
        "fix retry logic",
        "token budget assembly",
        "how does BM25 search work",
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(query), query, |b, &q| {
            b.iter(|| engine.run_pipeline(q, None, None, None).expect("pipeline"));
        });
    }
    group.finish();
    drop(corpus);
}

criterion_group!(
    benches,
    bench_index_cold,
    bench_index_warm,
    bench_run_pipeline
);
criterion_main!(benches);
