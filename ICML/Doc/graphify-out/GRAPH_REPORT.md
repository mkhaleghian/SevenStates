# Graph Report - C:\Users\kpk628\OneDrive - University of Tennessee\Documents\CUIP Projects\EV PTOJECT\sevenstate\ICML\Doc  (2026-06-27)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 12 nodes · 7 edges · 6 communities (2 shown, 4 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `128a1aa4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]

## God Nodes (most connected - your core abstractions)
1. `TA-GNN` - 5 edges
2. `GM-Zero` - 2 edges
3. `GraphMetaLearner` - 2 edges
4. `HistoryAwareNodeEmbedding` - 1 edges
5. `HistoryLengthAttention` - 1 edges
6. `MaskedNegativeLogLikelihood` - 1 edges
7. `ContrastiveStationEmbedding` - 1 edges
8. `Curriculum Learning` - 1 edges
9. `build_target_tensor` - 0 edges
10. `build_distance_adjacency` - 0 edges

## Surprising Connections (you probably didn't know these)
- `GraphMetaLearner` --references--> `TA-GNN`  [INFERRED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 0 → community 1_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **TA-GNN Components** — gnn_ev_icml_claude_chat_history_aware_node_embedding, gnn_ev_icml_claude_chat_history_length_attention, gnn_ev_icml_claude_chat_masked_negative_log_likelihood [EXTRACTED 1.00]
- **GM-Zero Components** — gnn_ev_icml_claude_chat_graph_meta_learner, gnn_ev_icml_claude_chat_contrastive_station_embedding [EXTRACTED 1.00]

## Communities (6 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.40
Nodes (5): Curriculum Learning, HistoryAwareNodeEmbedding, HistoryLengthAttention, MaskedNegativeLogLikelihood, TA-GNN

### Community 1 - "Community 1"
Cohesion: 0.67
Nodes (3): ContrastiveStationEmbedding, GM-Zero, GraphMetaLearner

## Knowledge Gaps
- **9 isolated node(s):** `build_target_tensor`, `build_distance_adjacency`, `build_provider_adjacency`, `compute_history_features`, `HistoryAwareNodeEmbedding` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TA-GNN` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.327) - this node is a cross-community bridge._
- **Why does `GraphMetaLearner` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **What connects `build_target_tensor`, `build_distance_adjacency`, `build_provider_adjacency` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._