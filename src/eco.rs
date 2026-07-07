//! The ecosystem's producer base — rung 1 of the continuous (generation-less)
//! terrarium.
//!
//! This is a **separate, self-contained** dynamical-field sim that *reuses* the
//! shared substrate (the typed cell [`Grid`], the seeded-ChaCha8 determinism
//! discipline, rayon, and the JSONL-metrics/viewer patterns) without touching
//! the generational [`crate::Simulator`]. The two coexist behind a `--mode`
//! switch. There are no animals and no hecs entities yet: the two lowest
//! trophic layers are modeled as **per-cell scalar fields** on the grid — arrays
//! for the spatial substrate, exactly what later rungs (herbivores that graze
//! `biomass`) will read.
//!
//! ## The two layers
//!
//! 1. **Nutrient** (`Cell::nutrient`) — the soil/decomposer field. Each tick it
//!    diffuses (a von-Neumann stencil, no-flux at walls/edges so it is
//!    conserved by the diffusion step) and is replenished toward a soil capacity
//!    by the decomposer/bacteria process. Later rungs feed it from corpses.
//! 2. **Biomass** (`Cell::biomass`) — the plant/producer field. Where nutrient
//!    is plentiful, biomass grows *consuming that nutrient* (the self-limiting
//!    mechanism); plants spread to empty neighbors by probabilistic seeding and
//!    decay slowly, returning a fraction of their mass to the soil (a partial
//!    nutrient cycle — it closes fully at rung 2 when corpses arrive).
//!
//! ## Why it self-regulates (the signal-first gate)
//!
//! Plant uptake draws nutrient down, and diffusion couples the nutrient pool
//! across the meadow, so **more plants ⇒ lower nutrient everywhere**. Seeding is
//! gated on local nutrient (`seed_nutrient_min`), so once the shared pool is
//! drawn down to that floor, new seeding stalls — a negative feedback that pins
//! coverage at a stable, patchy plateau instead of a runaway green fill or a
//! collapse. Because seeding also requires a *mature* neighbor, growth is
//! spatially autocatalytic, so the vegetated cells clump into patches rather
//! than a uniform wash. `seed_nutrient_min` is the main knob on the plateau
//! height.
//!
//! ## Determinism + parallelism
//!
//! The update is a **double-buffered** cellular automaton: every tick reads an
//! immutable snapshot (`grid.cells`) and writes a fresh scratch buffer
//! (`next`), so it is order-independent and rayon-parallel over cells. The only
//! stochastic step — seeding — draws from a `ChaCha8Rng` seeded deterministically
//! from `(seed, tick, cell-index)`, so two same-seed runs are bit-identical.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

use crate::grid::{Cell, Grid};

/// The tunable dynamics of the meadow. Kept as a struct (rather than bare
/// consts) so tests can isolate a single process — e.g. zero the reaction terms
/// to check the diffusion step conserves nutrient. [`Default`] holds the values
/// tuned to hit the self-regulation gate (a patchy ~20–60% coverage plateau).
#[derive(Clone)]
pub struct EcoParams {
    // --- nutrient (soil) field ---
    /// Diffusion coefficient for nutrient. Must be ≤ 0.25 for the explicit
    /// 4-neighbor stencil to stay non-oscillatory; higher = a more strongly
    /// shared (globally coupled) nutrient pool, which is what makes coverage
    /// self-limit rather than fill via a traveling wave.
    pub diffusion: f32,
    /// Rate the decomposer process replenishes each cell toward `soil_cap`.
    pub replenish: f32,
    /// Soil nutrient capacity — the equilibrium a bare cell relaxes to.
    pub soil_cap: f32,

    // --- plant (producer) field ---
    /// Peak nutrient uptake per unit biomass (Michaelis–Menten in nutrient).
    /// Uptake is what draws the shared pool down as biomass rises.
    pub uptake: f32,
    /// Half-saturation nutrient level for uptake.
    pub half_sat: f32,
    /// Biomass produced per unit nutrient taken up (logistically capped).
    pub conversion: f32,
    /// Fractional biomass lost to senescence each tick.
    pub decay: f32,
    /// Fraction of decayed (and killed) biomass returned to soil nutrient (the
    /// partial cycle — a death dumps a local nutrient pulse that lets the gap
    /// recover and reseed, which is what keeps the mosaic breathing).
    pub decay_return: f32,
    /// Maximum standing biomass per cell.
    pub biomass_max: f32,
    /// Per-tick probability a standing (mature) plant is cleared by disturbance.
    /// A gentle turnover term: it opens gaps that reseed, so the plateau *breathes*
    /// (a shifting patch mosaic) instead of freezing at a fixed point. Rung 2's
    /// herbivores will remove biomass the same way, so the loop already handles it.
    pub mortality: f32,

    // --- seeding (spread) ---
    /// Biomass a freshly seeded cell starts with.
    pub seed_set: f32,
    /// A cell must have at least this much nutrient to be seeded (the plateau
    /// knob: coverage rises until the shared pool is drawn down to here).
    pub seed_nutrient_min: f32,
    /// A neighbor must exceed this biomass to cast seeds (defines "mature").
    pub mature: f32,
    /// Per-tick seeding probability contributed by each mature neighbor.
    pub seed_rate: f32,

    // --- metrics / init ---
    /// Biomass above which a cell counts as "vegetated" for the coverage metric.
    pub coverage_threshold: f32,
    /// Fraction of cells sown with a founder plant at initialization.
    pub init_density: f32,
    /// Biomass of each initial founder (above `mature`, so founders seed at once).
    pub init_biomass: f32,
}

impl Default for EcoParams {
    fn default() -> EcoParams {
        // Tuned by the headless runner (seed 42, 128², 4000 ticks) to a stable
        // patchy plateau; see DESIGN.md for the measured curve.
        EcoParams {
            diffusion: 0.20,
            replenish: 0.050,
            soil_cap: 1.0,
            uptake: 0.30,
            half_sat: 0.35,
            conversion: 1.0,
            decay: 0.020,
            decay_return: 0.5,
            biomass_max: 1.0,
            mortality: 0.0018,
            seed_set: 0.08,
            seed_nutrient_min: 0.42,
            mature: 0.30,
            seed_rate: 0.05,
            coverage_threshold: 0.10,
            init_density: 0.01,
            init_biomass: 0.50,
        }
    }
}

/// Everything that defines a reproducible eco run: grid size, master seed, and
/// the dynamics. Same config + same build ⇒ byte-identical metrics.
#[derive(Clone)]
pub struct EcoConfig {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub params: EcoParams,
}

/// One metrics record (per recorded tick). No wall-clock field, so two same-seed
/// runs produce byte-identical JSONL.
#[derive(Clone, Serialize)]
pub struct EcoMetrics {
    pub tick: u64,
    /// Sum of biomass over all soil cells.
    pub total_biomass: f64,
    /// Fraction of soil cells with biomass above `coverage_threshold` (0..1).
    pub coverage: f64,
    /// Mean nutrient over all soil cells.
    pub mean_nutrient: f64,
}

/// Derive a deterministic per-cell seed for the stochastic seeding draw, in the
/// same SplitMix64 spirit as the generational sim's `agent_seed`. Independent of
/// thread/iteration order, so the parallel update stays reproducible.
fn cell_seed(master: u64, tick: u64, index: usize) -> u64 {
    let s = crate::mix(master);
    let s = crate::mix(s ^ tick);
    crate::mix(s ^ index as u64)
}

/// Init tag mixed into the master seed for the founder scatter, kept distinct
/// from any real tick used by `cell_seed` so the two RNG streams never alias.
const INIT_TAG: u64 = 0xEC05_EED0_0000_0001;

/// The nutrient field after one diffusion step: an in-bounds, non-obstacle
/// 4-neighbor stencil with **no-flux boundaries** (walls/edges are simply not
/// summed), which makes total nutrient conserved by diffusion and keeps every
/// value within the min/max of its neighborhood (so it cannot run away). Pure
/// and parallel; used by `tick` and unit-tested directly.
pub(crate) fn diffused_nutrient(cells: &[Cell], width: usize, height: usize, d: f32) -> Vec<f32> {
    (0..cells.len())
        .into_par_iter()
        .map(|i| {
            if cells[i].obstacle {
                return 0.0;
            }
            let x = i % width;
            let y = i / width;
            let n = cells[i].nutrient;
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;
            let mut add = |xx: usize, yy: usize| {
                let j = yy * width + xx;
                if !cells[j].obstacle {
                    sum += cells[j].nutrient;
                    cnt += 1.0;
                }
            };
            if x > 0 {
                add(x - 1, y);
            }
            if x + 1 < width {
                add(x + 1, y);
            }
            if y > 0 {
                add(x, y - 1);
            }
            if y + 1 < height {
                add(x, y + 1);
            }
            n + d * (sum - cnt * n)
        })
        .collect()
}

/// Count the von-Neumann neighbors of cell `i` whose biomass exceeds `mature`
/// (i.e. that are established enough to cast seeds). A gather over the snapshot,
/// so it is order-independent.
fn mature_neighbors(cells: &[Cell], width: usize, height: usize, i: usize, mature: f32) -> u32 {
    let x = i % width;
    let y = i / width;
    let mut count = 0u32;
    let mut check = |xx: usize, yy: usize| {
        let c = &cells[yy * width + xx];
        if !c.obstacle && c.biomass > mature {
            count += 1;
        }
    };
    if x > 0 {
        check(x - 1, y);
    }
    if x + 1 < width {
        check(x + 1, y);
    }
    if y > 0 {
        check(x, y - 1);
    }
    if y + 1 < height {
        check(x, y + 1);
    }
    count
}

/// The continuous producer-base simulation over a shared cell [`Grid`].
pub struct EcoSim {
    grid: Grid,
    /// Double-buffer scratch: next tick's `(nutrient, biomass)` per cell.
    next: Vec<(f32, f32)>,
    tick: u64,
    config: EcoConfig,
    metrics_out: Option<File>,
    /// Every recorded tick's metrics (drives the live viewer sparklines).
    history: Vec<EcoMetrics>,
    /// Record metrics (history + JSONL) every this-many ticks. 1 for the
    /// determinism/headless runs; the viewer bumps it to bound memory.
    metrics_interval: u64,
}

impl EcoSim {
    pub fn new(config: EcoConfig) -> EcoSim {
        let n = config.width * config.height;
        EcoSim {
            grid: Grid::new(config.width, config.height),
            next: vec![(0.0, 0.0); n],
            tick: 0,
            config,
            metrics_out: None,
            history: Vec::new(),
            metrics_interval: 1,
        }
    }

    /// Open (truncating) a JSONL metrics file; one line is written per recorded
    /// tick.
    pub fn set_metrics(&mut self, path: &Path) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        self.metrics_out = Some(file);
        Ok(())
    }

    /// Set how often (in ticks) metrics are recorded to history + JSONL.
    pub fn set_metrics_interval(&mut self, interval: u64) {
        self.metrics_interval = interval.max(1);
    }

    /// Fill the soil to capacity and scatter sparse founder plants using the
    /// master seeded RNG (deterministic), then record the tick-0 metrics. The
    /// eco analogue of `generate_initial_generation`.
    pub fn seed_initial(&mut self) {
        let p = &self.config.params;
        let (cap, density, founder) = (p.soil_cap, p.init_density, p.init_biomass);
        let mut rng = ChaCha8Rng::seed_from_u64(cell_seed(self.config.seed, INIT_TAG, 0));
        for cell in &mut self.grid.cells {
            if cell.obstacle {
                continue;
            }
            cell.nutrient = cap;
            cell.biomass = if rng.random::<f32>() < density { founder } else { 0.0 };
        }
        self.record_metrics();
    }

    /// Advance the meadow one tick: diffuse nutrient, then react (replenish,
    /// plant uptake→growth, decay, stochastic seeding) into the scratch buffer,
    /// then swap it in. Order-independent and rayon-parallel over cells.
    pub fn tick(&mut self) {
        let (w, h) = (self.grid.width, self.grid.height);
        let p = self.config.params.clone();
        let seed = self.config.seed;
        let t = self.tick;

        // Phase 1: nutrient diffusion (pure gather over the snapshot).
        let diff_n = diffused_nutrient(&self.grid.cells, w, h, p.diffusion);

        // Phase 2: local reaction + seeding, writing the fresh double-buffer.
        let cells = &self.grid.cells;
        self.next
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                if cells[i].obstacle {
                    *slot = (0.0, 0.0);
                    return;
                }
                let b = cells[i].biomass;
                let mut n = diff_n[i];
                // Empty coming into this tick? The seeding and mortality branches
                // partition on this, so each cell draws its per-cell RNG at most
                // once (and a plant that dies this tick can't reseed until next).
                let was_empty = b < p.seed_set;

                // Decomposer replenishment toward soil capacity.
                n += p.replenish * (p.soil_cap - n);

                // Plant uptake (Michaelis–Menten in nutrient, linear in biomass)
                // → logistically-capped growth. Uptake consumes nutrient: this
                // is the self-limiting draw-down on the shared pool.
                let mut b_new = b;
                if b > 0.0 {
                    let uptake = (p.uptake * (n / (n + p.half_sat)) * b).min(n.max(0.0));
                    n -= uptake;
                    let growth = p.conversion * uptake * (1.0 - b / p.biomass_max);
                    b_new = b + growth;
                }

                // Senescence: biomass decays, returning a fraction to the soil.
                let decayed = p.decay * b_new;
                b_new -= decayed;
                n += decayed * p.decay_return;

                if was_empty {
                    // Seeding: a (near-)empty, fertile cell adjacent to a mature
                    // plant germinates with probability rising per mature
                    // neighbor. A gather (this cell reads its neighbors), so it
                    // stays order-independent and parallel-safe.
                    if n > p.seed_nutrient_min {
                        let mature = mature_neighbors(cells, w, h, i, p.mature);
                        if mature > 0 {
                            let prob = (p.seed_rate * mature as f32).min(1.0);
                            let mut rng = ChaCha8Rng::seed_from_u64(cell_seed(seed, t, i));
                            if rng.random::<f32>() < prob {
                                b_new = p.seed_set;
                            }
                        }
                    }
                } else if b_new > p.mature {
                    // Disturbance mortality: a standing plant may be cleared,
                    // dumping a local nutrient pulse. This is the turnover that
                    // keeps the mosaic shifting rather than frozen.
                    let mut rng = ChaCha8Rng::seed_from_u64(cell_seed(seed, t, i));
                    if rng.random::<f32>() < p.mortality {
                        n += b_new * p.decay_return;
                        b_new = 0.0;
                    }
                }

                *slot = (n.max(0.0), b_new.clamp(0.0, p.biomass_max));
            });

        // Swap the double-buffer back into the authoritative grid cells.
        for (cell, &(n, b)) in self.grid.cells.iter_mut().zip(&self.next) {
            cell.nutrient = n;
            cell.biomass = b;
        }

        self.tick += 1;
        if self.tick.is_multiple_of(self.metrics_interval) {
            self.record_metrics();
        }
    }

    /// Compute + emit the current metrics (serial, so the f64 sums are
    /// order-deterministic).
    fn record_metrics(&mut self) {
        let thresh = self.config.params.coverage_threshold;
        let mut total_biomass = 0.0f64;
        let mut total_nutrient = 0.0f64;
        let mut vegetated = 0u64;
        let mut soil = 0u64;
        for cell in &self.grid.cells {
            if cell.obstacle {
                continue;
            }
            soil += 1;
            total_biomass += cell.biomass as f64;
            total_nutrient += cell.nutrient as f64;
            if cell.biomass > thresh {
                vegetated += 1;
            }
        }
        let denom = soil.max(1) as f64;
        let m = EcoMetrics {
            tick: self.tick,
            total_biomass,
            coverage: vegetated as f64 / denom,
            mean_nutrient: total_nutrient / denom,
        };
        if let Some(file) = self.metrics_out.as_mut()
            && let Ok(line) = serde_json::to_string(&m)
        {
            let _ = writeln!(file, "{line}");
        }
        self.history.push(m);
    }

    // ----- accessors (rendering / reporting; no internals leak) -----

    /// Ticks elapsed.
    pub fn tick_count(&self) -> u64 {
        self.tick
    }

    pub fn width(&self) -> usize {
        self.grid.width
    }

    pub fn height(&self) -> usize {
        self.grid.height
    }

    /// The dynamics parameters (the viewer scales its color washes by
    /// `soil_cap` / `biomass_max`).
    pub fn params(&self) -> &EcoParams {
        &self.config.params
    }

    /// Read-only view of the field cells, row-major (`cells()[y*width+x]`). The
    /// viewer reads `nutrient` / `biomass` / `obstacle` for its diorama.
    pub fn cells(&self) -> &[Cell] {
        &self.grid.cells
    }

    /// Every recorded tick's metrics (drives the viewer's biomass/coverage
    /// sparklines and the headless coverage summary).
    pub fn metrics_history(&self) -> &[EcoMetrics] {
        &self.history
    }

    /// The most recent metrics record, if any.
    pub fn latest(&self) -> Option<&EcoMetrics> {
        self.history.last()
    }

    #[cfg(test)]
    fn set_cell(&mut self, x: usize, y: usize, nutrient: f32, biomass: f32) {
        let i = y * self.grid.width + x;
        self.grid.cells[i].nutrient = nutrient;
        self.grid.cells[i].biomass = biomass;
    }

    #[cfg(test)]
    fn cell_at(&self, x: usize, y: usize) -> (f32, f32) {
        let c = &self.grid.cells[y * self.grid.width + x];
        (c.nutrient, c.biomass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seed: u64, w: usize, h: usize) -> EcoConfig {
        EcoConfig { width: w, height: h, seed, params: EcoParams::default() }
    }

    #[test]
    fn diffusion_conserves_total_and_stays_bounded() {
        // A non-uniform nutrient field, one diffusion step: total nutrient is
        // conserved (no-flux boundaries) and every output value stays within the
        // input min/max (an averaging stencil can't overshoot).
        let mut grid = Grid::new(5, 4);
        // Seed a spike and a couple of other values.
        grid.cells[0].nutrient = 4.0;
        grid.cells[7].nutrient = 2.0;
        grid.cells[18].nutrient = 1.0;
        let before: f32 = grid.cells.iter().map(|c| c.nutrient).sum();
        let out = diffused_nutrient(&grid.cells, 5, 4, 0.2);
        let after: f32 = out.iter().sum();
        assert!((before - after).abs() < 1e-4, "diffusion must conserve nutrient: {before} -> {after}");
        let (mn, mx) = (0.0f32, 4.0f32);
        for v in out {
            assert!((mn..=mx).contains(&v), "diffused value {v} left the input range");
        }
    }

    #[test]
    fn diffusion_conserves_with_an_obstacle() {
        // No-flux at an interior obstacle: the obstacle holds no nutrient and the
        // soil cells still conserve their total.
        let mut grid = Grid::new(4, 4);
        for (i, c) in grid.cells.iter_mut().enumerate() {
            c.nutrient = (i as f32) * 0.1;
        }
        let (ox, oy) = (1usize, 1usize);
        grid.set_obstacle((ox as u32, oy as u32));
        grid.cells[oy * 4 + ox].nutrient = 0.0;
        let before: f32 = grid.cells.iter().filter(|c| !c.obstacle).map(|c| c.nutrient).sum();
        let out = diffused_nutrient(&grid.cells, 4, 4, 0.2);
        let after: f32 = out.iter().sum(); // obstacle slot is 0.0
        assert!((before - after).abs() < 1e-4, "obstacle diffusion must conserve soil nutrient");
    }

    #[test]
    fn growth_consumes_nutrient() {
        // A single fertile, vegetated cell (no neighbors ⇒ no diffusion, no
        // seeding) must gain biomass and lose nutrient across a tick. Replenish
        // is zeroed so the only nutrient change is uptake vs decay-return.
        let mut c = cfg(1, 1, 1);
        c.params.replenish = 0.0;
        let mut sim = EcoSim::new(c);
        sim.set_cell(0, 0, 1.0, 0.3);
        let (n0, b0) = sim.cell_at(0, 0);
        sim.tick();
        let (n1, b1) = sim.cell_at(0, 0);
        assert!(b1 > b0, "biomass should grow: {b0} -> {b1}");
        assert!(n1 < n0, "growth should consume nutrient: {n0} -> {n1}");
    }

    #[test]
    fn biomass_is_capped() {
        // Biomass never exceeds biomass_max even from a rich cell run for a while.
        let mut sim = EcoSim::new(cfg(2, 1, 1));
        sim.set_cell(0, 0, 1.0, 0.9);
        let max = sim.params().biomass_max;
        for _ in 0..500 {
            sim.tick();
        }
        assert!(sim.cell_at(0, 0).1 <= max + 1e-6, "biomass exceeded cap");
    }

    #[test]
    fn coverage_metric_counts_vegetated_cells() {
        // Coverage = fraction of soil cells with biomass above the threshold.
        // Hand-set 3 of 9 cells above it; expect 3/9 with no ticks run.
        let mut sim = EcoSim::new(cfg(3, 3, 3));
        let thr = sim.params().coverage_threshold;
        sim.set_cell(0, 0, 0.5, thr + 0.2);
        sim.set_cell(1, 1, 0.5, thr + 0.2);
        sim.set_cell(2, 2, 0.5, thr + 0.2);
        sim.set_cell(0, 1, 0.5, thr - 0.05); // below threshold, not counted
        sim.record_metrics();
        let cov = sim.latest().unwrap().coverage;
        assert!((cov - 3.0 / 9.0).abs() < 1e-9, "coverage {cov} != 3/9");
    }

    #[test]
    fn double_buffer_update_is_deterministic() {
        // Two same-seed runs are bit-identical in both fields and in metrics —
        // the double-buffer + per-cell seeded RNG guarantee. A stochastic
        // (seeding-active) run, so this actually exercises the RNG path.
        let run = || {
            let mut sim = EcoSim::new(cfg(777, 48, 48));
            sim.seed_initial();
            for _ in 0..200 {
                sim.tick();
            }
            let fields: Vec<(u32, u32)> = sim
                .cells()
                .iter()
                .map(|c| (c.nutrient.to_bits(), c.biomass.to_bits()))
                .collect();
            let cov: Vec<u64> = sim.metrics_history().iter().map(|m| m.coverage.to_bits()).collect();
            (fields, cov)
        };
        let a = run();
        let b = run();
        assert!(a.0 == b.0, "fields diverged under an identical seed");
        assert!(a.1 == b.1, "coverage metrics diverged under an identical seed");
    }

    #[test]
    fn different_seed_differs() {
        // Different seeds ⇒ a different meadow (the seeding RNG actually varies).
        let run = |s: u64| {
            let mut sim = EcoSim::new(cfg(s, 48, 48));
            sim.seed_initial();
            for _ in 0..200 {
                sim.tick();
            }
            sim.cells().iter().map(|c| c.biomass.to_bits()).collect::<Vec<_>>()
        };
        assert!(run(1) != run(2), "different seeds produced an identical meadow");
    }

    #[test]
    fn meadow_self_regulates_to_a_patchy_plateau() {
        // The gate, as a test: from a sparse seeding, coverage neither explodes
        // to ~100% nor collapses to ~0% — it settles in a patchy mid-band. Run a
        // modest grid a few thousand ticks and assert the late-time coverage is
        // in (0.10, 0.85) and biomass is non-trivial. (The headless runner
        // reports the precise curve; this is the always-on regression guard.)
        let mut sim = EcoSim::new(cfg(42, 96, 96));
        sim.seed_initial();
        for _ in 0..3000 {
            sim.tick();
        }
        let cov = sim.latest().unwrap().coverage;
        assert!(cov > 0.10, "meadow collapsed (coverage {cov})");
        assert!(cov < 0.85, "meadow filled (coverage {cov})");
        assert!(sim.latest().unwrap().total_biomass > 0.0, "no standing biomass");
    }
}
