---
layout: post
title: "Delux Ai Zero!"
date: 2021-09-12 13:16:00 +0800
categories: reinforcement-learning 
---


# Delux Ai ZERO

Solving Lux Ai with the implementation of MuZero model. 

## Data Input Pipeline
### Unit Encoder 
- Inputs : game_state - List of units inforamation
- Ouputs : Embedded_units, Unit_embedding, Raw_units
	- Embedded_units (BS, num_units, 16) : embedding of each unit
	- Unit_embedding (BS, 16) : embedding of every units 
	- Raw_units (BS, num_units, 24) : before embedding

The `units_info` go through transformer layer and FC to generate `embedded_units`. The mean of `embedded_units` across all units pass through another layer of FC to get `unit_embedding`.

### Spatial Encoder
- Inputs : game_state - map
- Outputs : Spatial_embedding, Spatial_map
	- Spatial_embedding (BS, 64) 
	- Spatial_map (BS, 2, 32, 32)

Get the map of resources and roads and pad it to have the shape of (BS, 4, 32, 32). The `embedded_units` and `raw_units` are pass into scatter map of shape (BS,  32, 32, 32) where each `embedded_unit` information "placed" on. The scatter map and the game map is then concatenated and pass through 2 layers of Conv2d to get `embedded_map`. Another FC layer is applied to get the `spatial_embedding`.

### Scalar Encoder
- Inputs : game_state - game stats
- Outputs : Scalar_embedding
	- Scalar_embedding (BS, 16)

General game stats pass through two layers of FC and to get `scalar_embedding`.