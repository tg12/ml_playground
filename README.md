# Forecast Model Digest

This repository sits at the intersection of fundamental storage analytics, regime-aware flow modeling, and a pure price-based validation channel. It exists not just as a collection of scripts, but as a guided exploration through the structural dynamics that shape energy and digital-asset regimes. Dive in, inspect the numbers, and let the narrative unfold: every section below is intentionally curated to help you understand why the signal matters, how we validate it, and what to look for next.

The three pipelines highlighted below were selected because they each expose a different structural lens:

- **EIA storage** encapsulates the weekly US inventory releases that still anchor where regime transitions happen in North American gas markets. Its cadence, public availability, and direct link to storage-led mean-reversion make it the natural starting point for a research-grade fundamental model.  
- **GIE storage and LNG** complements the US story with a regime-candidate surface derived from European working volume, injections, withdrawals, and new LNG send-out proxies; it is designed to capture stress accumulation or relief in a basin with different infrastructure and seasonality.  
- **Coinbase candles** offer a divergent, price-only perspective, embedding volatility, order-flow curvature, and directional bias from BTC/USD sequences so we can reason about execution signals once the structural regime filter is in place.

Each section below expands the methodological choices, records the most recent validation narrative, and describes how to rerun the experiment in a deterministic setting.

## Why this stack deserves attention

- **Multi-angle clarity:** The fundamentals, regimes, and price-only views complement one another so you can triangulate structural shifts before overlaying execution or investment filters.
- **Reproducible craft:** Every summary, chart, and optimizer seed is locked behind deterministic RNGs so collaborators can pick up the thread and build on it without chasing stochastic noise.
- **Narrative-ready outputs:** Whether you are writing a desk memo, building a dashboard, or stress-testing a strategy, this repo already captures the storyline just point, proof, and publish.

## Research Stack

The following pipelines earn their place because they expose clean signals, highlight regime transitions, and feed story-ready diagnostics that you can cite in a notebook or presentation. Each subsection below first explains the analytical hypothesis, then lays out the deterministic process that proves it, and finally shares the latest validation narrative.

### EIA weekly storage forecast

The probabilistic hypothesis is that shifting fundamental drivers (e.g., regional balances, weather, LNG flows) can explain a large piece of next-week storage deltas if we respect the report chronology. We keep the narrative transparent so you can audition new drivers, add bespoke regional splits, or stress-test the weekly aggregations without fighting randomness. The pipeline:

1. Joins daily combined historical and forecast matrices to align each forecast column with its realized counterpart, then aggregates every driver into mean/min/max statistics on the EIA Friday week ending.  
2. Treats each regional change (Lower48, East, Midwest, Mountain, Pacific, SouthCentral, SouthCentral Salt, SouthCentral NonSalt) as an independent target and instantiates a seeded gradient boosting regressor, preserving reproducibility for Python and scikit-learn RNGs.  
3. Validates on the final 25% of the chronologically ordered matrix and tracks MAE, RMSE, and bias for every region, so it is clear how the feature bundle holds up before trusting the next forecast.  
4. Converts the predicted weekly change back into a level forecast by adding it to the latest verified storage total, delivering both directional delta and absolute level for downstream dashboards.

We treat the validation statistics as the opening lines of a dialogue if the MAE widens when a new driver is added, we note it here so the next researcher can retrace the trade-off.

### GIE regime and stress canvas

This pipeline operationalizes a regime filter grounded in European storage dynamics plus newly derived LNG send-out proxies. We designed it for analysts who want to justify a regime signal with both physical capacity cues and stress narratives. The research premise is that persistent asymmetries in capacity utilization and send-out pressure precede regime shifts, so:

1. The dataset undergoes synthetic feature construction (body size, trend momentum, breakout injections/withdrawals, utilization ratios) and normalization so injection/withdrawal capacity is visible even when raw volumes are sparse.  
2. Additional LNG variables such as send-out rate, momentum, volatility, and efficiency capture the export-driven part of the market; diagnostic thresholds are scaled to adaptive short/medium/long lookbacks to respect the available records.  
3. A three-component Gaussian mixture clusters multidimensional snapshots (`body_size`, `wick_polarity`, `breakout_*`, `utilization`, `sendout`) into regimes, which in turn feed a traffic-light summary of stress per country.  
4. Flow summary statistics report net withdrawal, storage change, fill level, and price impact narrative yielding a structured story for both research review and downstream filtering.

This traffic-light summary is deliberately verbose so you can slide the regime output into your own monitor and immediately see which countries are worth digging into.

<img width="2375" height="1109" alt="image" src="https://github.com/user-attachments/assets/3c6053a9-2480-4d33-9f2d-09b279391e71" />


### Coinbase candle sequence forecast

The goal here is to maintain a price-only comparator that can uncover volatility or order-flow features the storage models cannot see. It doubles as a humility check if the price-only model behaves predictably while total flows misfire, we know which signals to trust. The research approach:

1. Load the hourly candle history and normalize the close price via z-score so the networks can focus on curvature and acceleration rather than absolute scale.  
2. Slice the series into 24-hour windows with a 48-hour horizon, creating a deterministic `SequenceDataset` so every training/validation split is reproducible.  
3. Train a small GRU encoder (hidden size 64, two layers) capped by an MLP head, using Adam and a consistent seed across Python, NumPy, and PyTorch to avoid stochasticity.  
4. Evaluate on the holdout window, unscale the predictions, and examine residuals visually via `coinbase_forecast.png` to decide if an uncertainty head or risk filter is necessary.

The residual plot sits in the repo to make your sanity checks fast if the residuals widen during a regime shift, you can link that to the corresponding EIA/GIE signal without leaving the repo.

<img width="1680" height="1120" alt="image" src="https://github.com/user-attachments/assets/a08904fa-0b57-4524-a4d8-c2a84e8da8c3" />


## Reproducibility and next steps

Each pipeline can be rerun via the corresponding entry script with environment variables (e.g., `MPLBACKEND=Agg` for headless graphing). The essential research log is maintained in this README because the narrative should mirror the code; whenever you trigger Optuna sweeps, new regime labels, or probability overlays, add the key findings, validation metrics, and narrative takeaways here so teammates can trace the thread forward.

Future directions  
- Expand the gradient boosting study with structured hyperparameter sweeps (e.g., `n_estimators`, window aggregation) and compare MAE/RMSE deltas before declaring an edition, logging how each change shifts the directional bias.  
- Translate the GIE traffic light into a regime-transition probability mesh, then overlay directional filters from order-flow proxies to generate actionable edges; log the conditions under which a red country spills into amber or green so the mesh learns the cascade.  
- Augment the Coinbase network with uncertainty quantification (e.g., a heteroscedastic head) so the residuals can guide execution thresholds once a regime signal is layered on top, and archive the calibration diagnostics for quick review.

## Production dashboard orchestrator

The `production_dashboard.py` orchestrates both the EIA and GIE summaries to produce a single artefact that combines delta forecasts with a stress traffic light. It seeds the random generators, retrains the EIA models, recalculates the GIE regime summary, and renders a figure so that analysts can see both basins side-by-side. The script also logs the headline flow trend and price signal, enabling easy review during build/test runs.

### Running the dashboard

```sh
MPLCONFIGDIR=/tmp MPLBACKEND=Agg python3 production_dashboard.py
```

The environment variables keep Matplotlib from blocking on font cache writes in sandboxed contexts while allowing the PNG artefact to be saved for embedding into any monitoring board.

## Next steps

The numbered experiments below keep the story dynamic run them, capture the outcomes, and note the validation narrative here so we can debate causalities instead of chasing noise.

1. Sweep `n_estimators`/`window` choices with a structured grid or Optuna study and log the delta in the README when you discover stronger directional bias.  
2. Feed the GIE traffic-light table into a regime-driven overlay that flags when red countries start spilling into amber/green, then pair that with directional filters from order-flow proxies.  
3. For Coinbase, add an uncertainty head (e.g., predict a variance term) and calibrate trading thresholds relative to the residuals stored in the plot to turn the probability edge into an execution rule.

## Legal & Usage

This repository is a research and analytics effort. Nothing here establishes a fiduciary relationship or personal recommendation. The models, forecasts, commentary, and sample dashboards are for experimental, educational, or internal planning use only they are **not investment advice**, nor do they substitute for your own diligence or consultation with licensed advisors. All outputs are provided “as is” without any implied warranty; before applying a signal to an execution system, verify that the inputs, hyperparameters, and regime narratives still reflect current market conditions.
