use anyhow::Result;
use clap::{Parser, Subcommand};
use cs_core::engine::EngineConfig;
use cs_core::CoreEngine;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "codesurgeon",
    version,
    about = "Local-first codebase context engine for AI coding agents"
)]
struct Cli {
    /// Workspace root (defaults to current directory or CS_WORKSPACE env var)
    #[arg(short, long, global = true)]
    workspace: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index the workspace (or re-index if already done)
    Index,

    /// Show index statistics
    Status,

    /// Search for symbols matching a query
    Search {
        query: String,
        #[arg(short, long, default_value = "4000")]
        budget: u32,
    },

    /// Show skeleton (signatures only) for a file
    Skeleton { file_path: String },

    /// Show what would break if a symbol changed
    Impact { symbol_fqn: String },

    /// Trace a logic path between two symbols
    Flow { from: String, to: String },

    /// Show session memory / observations
    Memory {
        /// Delete an observation by ID
        #[arg(short, long)]
        delete: Option<String>,
    },

    /// Save an observation
    Observe {
        content: String,
        #[arg(short, long)]
        symbol: Option<String>,
    },

    /// Context capsule for a git diff (pipe or pass diff text)
    Diff {
        /// git diff text (or use: git diff | codesurgeon diff -)
        diff: String,
        #[arg(short, long, default_value = "4000")]
        budget: u32,
    },

    /// Show query stats: token savings, latency, intent breakdown
    Stats {
        /// Look-back window in days (default: 30)
        #[arg(short, long, default_value = "30")]
        days: u32,
    },

    /// Auto-generate per-directory CLAUDE.md summaries
    Docs {
        /// Write CLAUDE.md files to disk (default: preview only)
        #[arg(short, long)]
        write: bool,
    },

    /// Start the MCP server (same as codesurgeon-mcp binary)
    Mcp,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("CS_LOG").unwrap_or_else(|_| "warn".to_string()))
        .init();

    let cli = Cli::parse();
    let workspace = resolve_workspace(cli.workspace);
    let config = EngineConfig::new(&workspace);
    let engine = CoreEngine::new(config)?;

    match cli.command {
        Commands::Index => {
            println!("Indexing {}...", workspace.display());
            let stats = engine.index_workspace()?;
            println!(
                "Done: {} symbols | {} edges | {} files | session: {}",
                stats.symbol_count, stats.edge_count, stats.file_count, stats.session_id
            );
        }

        Commands::Status => {
            let stats = engine.index_stats()?;
            println!("Symbols : {}", stats.symbol_count);
            println!("Edges   : {}", stats.edge_count);
            println!("Files   : {}", stats.file_count);
            println!("Session : {}", stats.session_id);
        }

        Commands::Search { query, budget } => {
            let capsule = engine.get_context_capsule(&query, Some(budget), None, None)?;
            println!("{}", capsule);
        }

        Commands::Skeleton { file_path } => {
            let result = engine.get_skeleton(&file_path, None)?;
            println!(
                "Skeleton: {} ({} symbols, ~{} tokens)\n",
                result.file_path,
                result.symbols.len(),
                result.token_estimate
            );
            for sym in &result.symbols {
                println!("## {} ({}) @ line {}", sym.fqn, sym.kind, sym.start_line);
                println!("{}\n", sym.skeleton);
            }
        }

        Commands::Impact { symbol_fqn } => {
            let result = engine.get_impact_graph(&symbol_fqn, None, true)?;
            println!("Impact graph for: {}", result.target_fqn);
            println!("Total affected: {}\n", result.total_affected);

            if !result.direct_dependents.is_empty() {
                println!("Direct dependents:");
                for s in &result.direct_dependents {
                    println!("  {} ({}:{})", s.fqn, s.file_path, s.start_line);
                }
            }
            if !result.transitive_dependents.is_empty() {
                println!("\nTransitive dependents:");
                for s in &result.transitive_dependents {
                    println!("  {} ({}:{})", s.fqn, s.file_path, s.start_line);
                }
            }
        }

        Commands::Flow { from, to } => {
            let result = engine.search_logic_flow(&from, &to)?;
            if result.found {
                println!("Path from {} to {}:", from, to);
                for (i, sym) in result.path.iter().enumerate() {
                    println!(
                        "  {}. {} ({}:{})",
                        i + 1,
                        sym.fqn,
                        sym.file_path,
                        sym.start_line
                    );
                }
            } else {
                println!("No path found from {} to {}", from, to);
            }
        }

        Commands::Memory { delete } => {
            if let Some(id) = delete {
                if engine.delete_observation(&id)? {
                    println!("Deleted observation {id}.");
                } else {
                    eprintln!("No observation found with id {id}.");
                    std::process::exit(1);
                }
                return Ok(());
            }
            let observations = engine.get_session_context()?;
            if observations.observations.is_empty() {
                println!("No session observations.");
                return Ok(());
            }
            for obs in &observations.observations {
                let stale = if obs.is_stale { " [STALE]" } else { "" };
                println!(
                    "[{}]{} (id: {}): {}",
                    obs.created_at, stale, obs.id, obs.content
                );
            }
        }

        Commands::Observe { content, symbol } => {
            engine.save_observation(&content, symbol.as_deref())?;
            println!("Observation saved.");
        }

        Commands::Diff { diff, budget } => {
            // Support "-" to read from stdin
            let diff_text = if diff == "-" {
                use std::io::Read;
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else {
                diff
            };
            println!("{}", engine.get_diff_capsule(&diff_text, Some(budget))?);
        }

        Commands::Stats { days } => {
            println!("{}", engine.get_stats(Some(days))?);
        }

        Commands::Docs { write } => {
            let docs = engine.generate_module_docs(write)?;
            println!("{}", docs);
            if write {
                println!("CLAUDE.md files written to each module directory.");
            }
        }

        Commands::Mcp => {
            // Delegate to the MCP binary
            println!(
                "Use `codesurgeon-mcp` directly, or add to Claude Code config:\n\
                 {{\"mcpServers\":{{\"codesurgeon\":{{\"command\":\"codesurgeon-mcp\",\"env\":{{\"CS_WORKSPACE\":\"{}\"}}}}}}}}",
                workspace.display()
            );
        }
    }

    Ok(())
}

fn resolve_workspace(arg: Option<PathBuf>) -> PathBuf {
    if let Some(p) = arg {
        return p;
    }
    if let Ok(ws) = std::env::var("CS_WORKSPACE") {
        return PathBuf::from(ws);
    }
    if let Ok(ws) = std::env::var("CLAUDE_CODE_WORKSPACE") {
        return PathBuf::from(ws);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
