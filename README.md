# SafeStreets-LDN: Graph Transformers for Crash Risk in London

## Overview
SafeStreets-LDN is a machine learning and graph-based road safety project that predicts crash risk at road intersections across Greater London. The project uses open geospatial data and collision records to identify potentially high-risk junctions and support proactive road-safety analysis.

This project combines:
- OpenStreetMap road network data
- STATS19 accident records (2020–2024)
- Graph Neural Networks (TransformerConv)
- XGBoost baseline models

The aim is to predict which intersections are more likely to experience crashes and assess crash severity risk using only static, open-access data.

---

## Problem Statement
Road traffic accidents remain a major public safety issue. Traditional blackspot analysis is often reactive, meaning dangerous areas are identified only after serious accidents have already happened.

This project explores whether crash risk in London can be predicted using:
- Road network structure
- Junction connectivity
- Road type information
- Static geospatial features

without relying on expensive traffic sensors or real-time traffic flow data.

---

## Project Goals
The main goals of this project are to:

- Build a Greater London road graph from OpenStreetMap
- Map collision records to the nearest intersections
- Engineer node and edge features from the road network
- Train graph-based models for:
  - Accident occurrence prediction
  - Accident severity prediction
- Compare TransformerConv performance against XGBoost and simple baselines
- Generate city-wide crash-risk estimates for safety screening

---

## Key Features
- Road network represented as a graph
- Node-level risk prediction
- Transformer-based graph convolution model
- Spatial train/test split for realistic evaluation
- Comparison with XGBoost and baseline models
- Risk mapping for London intersections and boroughs
- Reproducible pipeline using open data

---

## Dataset
This project uses two main data sources:

### 1. OpenStreetMap (OSM)
Used to build the London road network graph, including:
- Intersections
- Road types
- Edge lengths
- Estimated speeds
- Travel times

### 2. STATS19 Collision Data
Used for accident records from **2020 to 2024**, including:
- Latitude/longitude
- Severity
- Time information

Collisions are mapped to the nearest road-network node to create risk labels.

---

## Methodology
The road network is modeled as a graph where:

- **Nodes** represent intersections and dead-ends
- **Edges** represent road segments between nodes

### Node Features
- Node degree
- Mean edge length
- Mean estimated speed
- One-hot encoded road categories

### Edge Features
- Length
- Speed
- Travel time
- Direction and bearing (in the full feature version)

### Models Used
- Majority baseline
- Random baseline
- XGBoost
- TransformerConv Graph Neural Network

### Prediction Tasks
1. **Accident Occurrence**  
   Binary classification: whether crashes occur at an intersection

2. **Severity Prediction**  
   Multi-class classification based on weighted severity risk bins

---

## Results
The project evaluates models using a spatial holdout split in eastern London.

### Test Performance
| Model | Occurrence | Severity |
|------|-----------:|---------:|
| Majority baseline | 0.853 | 0.853 |
| Random baseline | 0.711 | 0.717 |
| XGBoost | 0.840 | 0.839 |
| TransformerConv (tuned) | 0.854 | 0.840 |

The tuned TransformerConv model achieved the best overall performance, especially for accident occurrence prediction under spatial testing.

---

## Tech Stack
- Python
- Jupyter Notebook
- PyTorch
- PyTorch Geometric
- XGBoost
- Optuna
- OSMnx
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
