use crate::symbol::{Edge, EdgeKind, Symbol};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::HashMap;

/// The in-memory dependency graph.
/// Nodes = Symbols, Edges = EdgeKind relationships.
pub struct CodeGraph {
    graph: DiGraph<Symbol, EdgeKind>,
    /// Map from Symbol.id → NodeIndex for fast lookup
    id_to_idx: HashMap<u64, NodeIndex>,
}

impl CodeGraph {
    pub fn new() -> Self {
        CodeGraph {
            graph: DiGraph::new(),
            id_to_idx: HashMap::new(),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    pub fn add_symbol(&mut self, symbol: Symbol) -> NodeIndex {
        if let Some(&existing) = self.id_to_idx.get(&symbol.id) {
            // Update in place if re-indexed
            self.graph[existing] = symbol;
            return existing;
        }
        let id = symbol.id;
        let idx = self.graph.add_node(symbol);
        self.id_to_idx.insert(id, idx);
        idx
    }

    pub fn add_edge(&mut self, from_id: u64, to_id: u64, kind: EdgeKind) {
        if let (Some(&from_idx), Some(&to_idx)) =
            (self.id_to_idx.get(&from_id), self.id_to_idx.get(&to_id))
        {
            // Avoid duplicate edges
            if !self.graph.contains_edge(from_idx, to_idx) {
                self.graph.add_edge(from_idx, to_idx, kind);
            }
        }
    }

    pub fn add_edges_batch(&mut self, edges: &[Edge]) {
        for edge in edges {
            self.add_edge(edge.from_id, edge.to_id, edge.kind.clone());
        }
    }

    /// Remove all symbols from a file (used on re-index).
    pub fn remove_file(&mut self, file_path: &str) {
        // Collect symbol IDs first — NodeIndex values are invalidated by
        // DiGraph::remove_node (swap-remove), so we can't hold onto them.
        let to_remove: Vec<u64> = self
            .graph
            .node_indices()
            .filter(|&i| self.graph[i].file_path == file_path)
            .map(|i| self.graph[i].id)
            .collect();

        for sym_id in to_remove {
            if let Some(idx) = self.id_to_idx.remove(&sym_id) {
                self.graph.remove_node(idx);
                // DiGraph::remove_node swap-removes: the previously-last node
                // is moved to `idx`. Update its id_to_idx entry.
                if idx.index() < self.graph.node_count() {
                    let swapped_id = self.graph[idx].id;
                    self.id_to_idx.insert(swapped_id, idx);
                }
            }
        }
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    pub fn get_symbol(&self, id: u64) -> Option<&Symbol> {
        self.id_to_idx.get(&id).map(|&idx| &self.graph[idx])
    }

    pub fn get_symbol_mut(&mut self, id: u64) -> Option<&mut Symbol> {
        self.id_to_idx.get(&id).map(|&idx| &mut self.graph[idx])
    }

    pub fn all_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.graph.node_weights()
    }

    pub fn symbol_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Everything that depends on `id` (reverse edges — callers, importers).
    pub fn dependents(&self, id: u64) -> Vec<&Symbol> {
        self.id_to_idx
            .get(&id)
            .map(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .map(|n| &self.graph[n])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Everything that `id` depends on (forward edges — callees, imports).
    pub fn dependencies(&self, id: u64) -> Vec<&Symbol> {
        self.id_to_idx
            .get(&id)
            .map(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .map(|n| &self.graph[n])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Symbols in the same file.
    pub fn file_symbols(&self, file_path: &str) -> Vec<&Symbol> {
        self.graph
            .node_weights()
            .filter(|s| s.file_path == file_path)
            .collect()
    }

    /// Compute a simple degree-based centrality score (combined in+out).
    /// High-centrality nodes are "pivot" candidates.
    pub fn centrality_score(&self, id: u64) -> f32 {
        let idx = match self.id_to_idx.get(&id) {
            Some(&i) => i,
            None => return 0.0,
        };
        let in_degree = self
            .graph
            .neighbors_directed(idx, Direction::Incoming)
            .count() as f32;
        let out_degree = self
            .graph
            .neighbors_directed(idx, Direction::Outgoing)
            .count() as f32;
        // Sigmoid-like normalization independent of workspace size.
        let raw = in_degree * 2.0 + out_degree;
        raw / (raw + 15.0)
    }

    /// Pure in-degree centrality: "how many other symbols depend on this one?"
    /// Use this for Structural queries — a class that many files call is truly central.
    /// A class that calls many things (high out-degree) is a leaf consumer, not a hub.
    pub fn in_degree_score(&self, id: u64) -> f32 {
        let idx = match self.id_to_idx.get(&id) {
            Some(&i) => i,
            None => return 0.0,
        };
        let in_degree = self
            .graph
            .neighbors_directed(idx, Direction::Incoming)
            .count() as f32;
        // k=5 so even lightly-referenced types score meaningfully vs unreferenced ones
        in_degree / (in_degree + 5.0)
    }

    /// Family in-degree: in-degree of a type PLUS the in-degrees of all its methods.
    /// This is the right centrality signal for type definitions because call edges
    /// go to methods (PDFLibrary::addDocument), not to the class type itself.
    /// A class with in-degree=11 but whose methods receive 130 calls scores correctly
    /// as more central than a data model with in-degree=100 but no method callers.
    /// Matches methods by name prefix: methods of `Foo` have name `Foo::methodName`.
    pub fn family_in_degree_score(&self, id: u64) -> f32 {
        let sym = match self.id_to_idx.get(&id).map(|&idx| &self.graph[idx]) {
            Some(s) => s,
            None => return 0.0,
        };
        let own_in = self
            .id_to_idx
            .get(&id)
            .map(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count() as u32
            })
            .unwrap_or(0);

        let prefix = format!("{}::", sym.name);
        let method_in: u32 = self
            .graph
            .node_weights()
            .filter(|s| s.name.starts_with(&prefix))
            .map(|s| {
                self.id_to_idx
                    .get(&s.id)
                    .map(|&idx| {
                        self.graph
                            .neighbors_directed(idx, Direction::Incoming)
                            .count() as u32
                    })
                    .unwrap_or(0)
            })
            .sum();

        let total = (own_in + method_in) as f32;
        // k=5 consistent with in_degree_score
        total / (total + 5.0)
    }

    /// Find a path from `from_id` to `to_id` (for search_logic_flow).
    /// Returns symbol IDs in order (shortest path), or empty if no path exists.
    /// Uses BFS — O(V+E), safe on dense graphs unlike all_simple_paths which is exponential.
    pub fn find_path(&self, from_id: u64, to_id: u64) -> Vec<u64> {
        let from_idx = match self.id_to_idx.get(&from_id) {
            Some(&i) => i,
            None => return vec![],
        };
        let to_idx = match self.id_to_idx.get(&to_id) {
            Some(&i) => i,
            None => return vec![],
        };

        let mut prev: std::collections::HashMap<NodeIndex, NodeIndex> =
            std::collections::HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        prev.insert(from_idx, from_idx);
        queue.push_back(from_idx);

        while let Some(node) = queue.pop_front() {
            if node == to_idx {
                let mut path = vec![];
                let mut cur = to_idx;
                loop {
                    path.push(self.graph[cur].id);
                    let p = prev[&cur];
                    if p == cur {
                        break;
                    }
                    cur = p;
                }
                path.reverse();
                return path;
            }
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if let std::collections::hash_map::Entry::Vacant(e) = prev.entry(neighbor) {
                    e.insert(node);
                    queue.push_back(neighbor);
                }
            }
        }
        vec![]
    }

    /// Blast-radius: all symbols transitively depending on `id`.
    /// Bounded by `max_depth` to avoid runaway traversal.
    pub fn blast_radius(&self, id: u64, max_depth: u32) -> Vec<&Symbol> {
        let start_idx = match self.id_to_idx.get(&id) {
            Some(&i) => i,
            None => return vec![],
        };

        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_idx, 0u32));

        while let Some((idx, depth)) = queue.pop_front() {
            if depth >= max_depth || visited.contains(&idx) {
                continue;
            }
            visited.insert(idx);

            for neighbor in self.graph.neighbors_directed(idx, Direction::Incoming) {
                queue.push_back((neighbor, depth + 1));
            }
        }

        visited
            .iter()
            .filter(|&&idx| idx != start_idx)
            .map(|&idx| &self.graph[idx])
            .collect()
    }

    /// Find symbols by name (case-insensitive prefix match).
    pub fn find_by_name(&self, name: &str) -> Vec<&Symbol> {
        let lower = name.to_lowercase();
        self.graph
            .node_weights()
            .filter(|s| s.name.to_lowercase().starts_with(&lower))
            .collect()
    }

    /// Find a symbol by its fully-qualified name.
    pub fn find_by_fqn(&self, fqn: &str) -> Option<&Symbol> {
        self.graph.node_weights().find(|s| s.fqn == fqn)
    }

    /// Find symbols whose FQN or name contains `query` (case-insensitive).
    /// Used for anti-hallucination "did you mean?" suggestions.
    pub fn fuzzy_fqn_matches(&self, query: &str, limit: usize) -> Vec<&Symbol> {
        let q = query.to_lowercase();
        // Prefer symbols whose name contains the query term
        let mut scored: Vec<(&Symbol, usize)> = self
            .graph
            .node_weights()
            .filter_map(|s| {
                let fqn_lower = s.fqn.to_lowercase();
                let name_lower = s.name.to_lowercase();
                if name_lower.contains(&q) {
                    Some((s, 2)) // name match scores higher
                } else if fqn_lower.contains(&q) {
                    Some((s, 1))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.fqn.cmp(&b.0.fqn)));
        scored.into_iter().take(limit).map(|(s, _)| s).collect()
    }

    /// Symbols that overlap with the given file + line range.
    /// Used for diff-aware capsule to find changed symbols.
    pub fn symbols_in_range(&self, file_path: &str, start: u32, end: u32) -> Vec<&Symbol> {
        self.graph
            .node_weights()
            .filter(|s| s.file_path == file_path && s.start_line <= end && s.end_line >= start)
            .collect()
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}
