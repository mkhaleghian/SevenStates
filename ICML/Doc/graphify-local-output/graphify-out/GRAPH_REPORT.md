# Graph Report - C:\Users\kpk628\OneDrive - University of Tennessee\Documents\CUIP Projects\EV PTOJECT\sevenstate\ICML\Doc\graphify-local-output  (2026-06-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 353 nodes · 520 edges · 20 communities (19 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.8)
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
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]

## God Nodes (most connected - your core abstractions)
1. `EV Charging GNN Literature Corpus` - 60 edges
2. `attention` - 27 edges
3. `GCN` - 26 edges
4. `GAT` - 15 edges
5. `Nodes: charging stations` - 14 edges
6. `CityEVCP` - 12 edges
7. `Traffic-Weather Graph Convolutional Network (TW-GCN)` - 11 edges
8. `LSTM` - 10 edges
9. `Heterogeneous Spatio-Temporal Graph Convolutional Network` - 9 edges
10. `Transformer` - 8 edges

## Surprising Connections (you probably didn't know these)
- `EV Charging GNN Literature Corpus` --contains_entry--> `Predicting electric vehicle charging demand using a heterogeneous spatio-temporal GCN`  [EXTRACTED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 8 → community 5_
- `EV Charging GNN Literature Corpus` --contains_entry--> `Modeling of machine learning with SHAP approach for... choice behavior prediction`  [EXTRACTED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 8 → community 15_
- `EV Charging GNN Literature Corpus` --contains_entry--> `An electric vehicle charging demand prediction approach based on a Graph-based Spatio-temporal Attention Network`  [EXTRACTED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 8 → community 7_
- `EV Charging GNN Literature Corpus` --contains_entry--> `Deep Spatio-Temporal Forecasting of Electrical Vehicle Charging Demand`  [EXTRACTED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 8 → community 2_
- `EV Charging GNN Literature Corpus` --contains_entry--> `Spatiotemporal Prediction of Electric Vehicle Charging Load Based on Large Language Models`  [EXTRACTED]
  GNN_EV_ICML_claude_chat.txt → GNN_EV_ICML_claude_chat.txt  _Bridges community 8 → community 1_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **TA-GNN Components** — concept_ta_gnn, concept_historyawarenodeembedding, concept_historylengthattention, concept_directional_neighbor_attention, concept_adaptive_spatial_temporal_fusion [INFERRED 0.80]
- **Graph Construction Strategies** — concept_adaptive_adjacency, concept_dtw, concept_haversine, concept_poi, concept_multi_graph, concept_hypergraph [EXTRACTED 0.90]

## Communities (20 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (41): attention, Autoformer, GAT, multi-graph, 20 charging stations in a city distributed network., Datasets from Palo Alto, Boulder, Dundee, and Perth., PeMSD4 traffic dataset (San Francisco) and IEEE 33-bus power system data., Xingxing Charging data from a Chinese city (256 stations). (+33 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (35): AGCRN, DCRNN, DTW, GCN, STGCN, Charging data from 10 stations in Palo Alto, California., Charging load data from a city in Northeast China (likely Wuhan based on text or example), FreewayEV (Zhejiang) and UrbanEV (Shenzhen). (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (31): Haversine, LSTM, ChargePoint dataset (86 stations) in Tennessee, USA, plus traffic and weather data., ChargePoint dataset in Tennessee, USA., Fast and slow charging loads from EVCS in Perth, UK., Perth, UK., Public charging data from Palo Alto, California (6 EVCSs)., Publicly available charging transaction data from Palo Alto, USA. (+23 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (31): LLM, reinforcement learning, Transformer, 46 battery swap stations in a city in Northeast China., ChargeFinder data for Sweden (Värnamo, Varberg, Malmö)., EV2Gym simulator., Palo Alto (USA) and Dundee (Scotland) datasets., Trajectories generated using EV2Gym simulator. (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (31): POI, queuing, PeMSD4 traffic dataset (San Francisco) and IEEE 33-bus power system., Simulation in MATLAB., Simulation in MATLAB compared with ANN, AIChOa, and NSGA-II., Traffic flow and road network data from a city in East China., Traffic flow data, temperature data, and road network topology from a city in East China.,  Edges: connections. (+23 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (30): adaptive graph, GRU, 22 TAZs in Futian District, Shenzhen, China., Georgia Tech EV charging dataset (Atlanta, USA)., GPS trajectory dataset of 76,774 EVs and 1,128 stations in Beijing., GPS trajectory dataset of EVs in Beijing., GPS trajectory dataset of over 76,000 private EVs in Beijing, China., Real-world EV GPS trajectory dataset of 76,774 private EVs and 1,128 charging stations in Beijing, China. (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (27): federated, meta-learning, 18,061 charging piles in Shenzhen, China., 18,061 EV charging piles in Shenzhen, China., 25,246 EV charging piles across six cities in the Greater Bay Area, China., 6 cities in the Greater Bay Area, China., ST-EVCDP dataset (18,061 piles, 247 traffic zones) from Shenzhen, China.,  Edges: neighboring relationships. (+19 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (25): TCN, 18,061 charging piles in 247 traffic zones in Shenzhen, China., 247 traffic zones in Shenzhen., England highway traffic flow dataset and highway driving data from Guizhou Province., Real-world EV charging records from 732 stations in Beijing, China., UK highway traffic flow and Guizhou driving data.,  Edges: highway connections.,  Edges: spatial connections. (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (24): 10 EV charging stations in Xi’an, China., 10 fast charging stations in Beijing., 10 fast charging stations in Beijing, China., 11 EV charging stations in Changsha, China., 1,681 public charging stations in Shenzhen, China., 330 fast charging stations in Chongqing, China., EV trajectory and charging data from Wuhan, China., FreewayEV dataset (Zhejiang, China) and UrbanEV dataset (Shenzhen, China). (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (20): adaptive adjacency, Copula, static graph, METR-LA and PEMS-BAY traffic datasets., Operational data from 321 new energy electric buses in Shanghai, converted into a charging load matrix and distance matrix., Real-world data from 247 public EV charging stations in Shenzhen, China.,  Edges: adaptive/learned dependencies.,  Edges: correlations between load attributes (intra-station) or spatial dependencies (inter-station). (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (11): hypergraph, ST-EVCDP dataset containing 18,061 public charging piles in Shenzhen, China., ST-EVCDP dataset (Shenzhen).,  Edges: spatial neighbors and hyperedges (land-use attributes).,  Edges: spatial neighbors (GAT) and hyperedges representing groups of areas with similar land-use attributes., CityEVCP, Nodes: travel communities/sub-areas, Citywide electric vehicle charging demand prediction approach considering urban region and dynamic influences (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (8): adaptive spatial-temporal fusion, cold-start, directional neighbor attention, HistoryAwareNodeEmbedding, HistoryLengthAttention, hurdle loss, static GNN aggregation, TA-GNN

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (6): Informer, Dundee EV Charging Station dataset (57 points).,  Edges: connectivity based on road distance., Attribute-Augmented Spatiotemporal Graph Informer Network (AST-GIN), AST-GIN: Attribute-Augmented Spatiotemporal Graph Informer Network for Electric Vehicle Charging Station Availability Forecasting, Charging station availability.

### Community 13 - "Community 13"
Cohesion: 0.33
Nodes (6): 400,000 IoT sensor records from Texas (Dallas metro)., Nodes/Edges: not explicitly defined as a graph for the prediction target in the traditional sense, but GNN is used to learn interconnections between EV load consumption and user dynamics., GNN-ViGNet (Hybrid regression model using GNN, VGGNet, and Inception), Nodes/Edges are not explicitly defined as a graph for the prediction target in the traditional sense, but GNN is used to learn interconnections between EV load consumption and user dynamics., Enhancing Urban Electric Vehicle (EV) Fleet Management Efficiency in Smart Cities: A Predictive Hybrid Deep Learning Framework, EV charging load.

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (6): 4 charging stations and 15 surrounding roads in Xi’an, China.,  Edges: connections., Spatiotemporal Graph Convolutional Network (SGCN), Nodes: road intersections and charging stations (as virtual roads), Operating Status Prediction Model at EV Charging Stations With Fusing Spatiotemporal Graph Convolutional Network, Operating status.

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (6): 500 EVs in Japan., N/A (Classification task, not graph-based forecasting)., Machine Learning classifiers (XGBoost, RF, etc, N/A (Classification task, not graph-based forecasting)., Modeling of machine learning with SHAP approach for... choice behavior prediction, Charging station choice class.

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (6): ERCOT load data and Texas traffic flow data (synthetic 2000-bus grid).,  Edges: power lines., GCNN-LSTM fusion model for prediction; Q-learning for allocation, Nodes: power substations (buses), Dynamic Spatio-Temporal Planning Strategy of EV Charging Stations and DGs Using GCNN-Based Predicted Power Demand, EV charging demand growth (Power).

### Community 17 - "Community 17"
Cohesion: 0.50
Nodes (4): demand similarity graph, distance graph, multi-graph fusion, traffic network graph

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (3): ContrastiveStationEmbedding, GM-Zero, GraphMetaLearner

## Knowledge Gaps
- **194 isolated node(s):** ` Edges: the connectivity between them.`, `Charging load (number of vehicles) for electric buses.`, `Operational data from 321 new energy electric buses in Shanghai, converted into a charging load matrix and distance matrix.`, `Nodes: traffic nodes (intersections/endpoints) or charging station locations within the Transportation Network (TN)`, ` Edges: road segments.` (+189 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EV Charging GNN Literature Corpus` connect `Community 8` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`?**
  _High betweenness centrality (0.458) - this node is a cross-community bridge._
- **Why does `GCN` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 12`, `Community 14`, `Community 16`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Why does `attention` connect `Community 0` to `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **What connects ` Edges: the connectivity between them.`, `Charging load (number of vehicles) for electric buses.`, `Operational data from 321 new energy electric buses in Shanghai, converted into a charging load matrix and distance matrix.` to the rest of the system?**
  _194 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06463414634146342 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06890756302521009 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.07311827956989247 - nodes in this community are weakly interconnected._