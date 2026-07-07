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

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use hecs::Entity;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

use crate::Position;
use crate::agent::{Brain, mutate_genome};
use crate::grid::{Cell, Grid};

/// Herbivore sensor layout — the forager sensorium fed to the reused
/// feed-forward [`Brain`] each tick. Width is [`HERB_INPUTS`]; the genome decodes
/// source ids *modulo* this width, so a different count from the challenge
/// agent's 17 is fine. Outputs reuse the movement scheme (trigger + 4 dirs).
///
/// |  id | sensor      | meaning                                                       |
/// |----:|-------------|---------------------------------------------------------------|
/// |   0 | bias        | constant 1.0                                                  |
/// |   1 | oscillator  | tick parity (0.0 / 1.0)                                       |
/// |   2 | energy      | own energy / `energy_max`, clamped 0..1 (satiation state)    |
/// |   3 | random      | fresh per-tick uniform 0..1 (exploration noise)              |
/// |   4 | food_here   | biomass at the current cell / `biomass_max`                  |
/// |   5 | grad_ns     | (biomass[y+1] − biomass[y−1]) / `biomass_max`, clamp −1..1   |
/// |   6 | grad_ew     | (biomass[x+1] − biomass[x−1]) / `biomass_max`, clamp −1..1   |
/// |   7 | blocked_n   | 1.0 if the +y neighbor is off-grid or occupied              |
/// |   8 | blocked_s   | 1.0 if the −y neighbor is off-grid or occupied              |
/// |   9 | blocked_e   | 1.0 if the +x neighbor is off-grid or occupied              |
/// |  10 | blocked_w   | 1.0 if the −x neighbor is off-grid or occupied              |
/// |  11 | density     | occupied fraction of the 8-neighborhood (herbivore crowding) |
const HERB_INPUTS: usize = 12;

/// A herbivore: the first mobile trophic level and the project's first
/// generation-less birth/death evolver. Bundled as one hecs component (mirroring
/// the challenge sim's [`crate::agent::Agent`]) alongside a standalone
/// [`Position`]; energy and lineage ride here because the serial entity phase
/// touches them together with the brain, so splitting them buys no parallelism.
pub(crate) struct Herbivore {
    /// The connection-list genome; a mutated copy is passed to each child.
    genome: Vec<u32>,
    /// The decoded feed-forward brain (12 inputs → movement), rebuilt from the
    /// genome at construction so brain and genome always agree.
    brain: Brain,
    /// Body energy. Grazing adds, metabolism/movement subtract; ≤0 ⇒ death,
    /// ≥ `repro_threshold` ⇒ a mutated child (energy split in half).
    energy: f32,
    /// Dynasty id (founder index, inherited verbatim) — viewer coloring only.
    lineage: u32,
    /// Inner-neuron count the brain was built with (to build children the same).
    amt_inners: u8,
}

/// A read-only per-herbivore snapshot for the viewer, materialized in stable
/// `order` sequence (keeps the render code off the ECS internals, like the
/// challenge sim's `AgentView`).
pub struct HerbView {
    pub pos: (u32, u32),
    pub lineage: u32,
    /// Energy as a 0..1 fraction of `energy_max` (drives brightness).
    pub energy_frac: f32,
}

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

    // --- herbivores (rung 2: the first mobile trophic level) ---
    /// Max biomass a herbivore can graze from its cell in one tick (the intake
    /// ceiling — the main lever on how fast herbivores drain the meadow).
    pub graze_cap: f32,
    /// Energy gained per unit biomass grazed (trophic transfer efficiency).
    pub graze_efficiency: f32,
    /// Baseline energy spent per tick just staying alive (the starvation clock).
    pub metabolism: f32,
    /// Extra energy spent on a tick where the herbivore actually changes cell.
    pub move_cost: f32,
    /// Energy at/above which a herbivore spawns a mutated child (splitting its
    /// energy in half) onto an empty neighbor — the reproduction gate.
    pub repro_threshold: f32,
    /// Satiation cap: energy never exceeds this, and grazing is throttled to the
    /// headroom below it (a full animal stops eating), which bounds overgrazing.
    pub energy_max: f32,
    /// Energy each founder / reseeded herbivore starts with.
    pub herb_init_energy: f32,
    /// Nutrient deposited into a herbivore's cell when it dies (body → soil):
    /// this is what closes the trophic loop herbivore → nutrient → plant.
    pub corpse_nutrient: f32,
    /// Inner-neuron count of each herbivore brain.
    pub herb_inner_neurons: u8,
    /// Gene (connection) count of each founder herbivore genome.
    pub herb_genome_length: usize,
    /// Per-bit mutation probability applied to a child's inherited genome.
    pub herb_mutation_rate: f32,
    /// Number of founder herbivores scattered at initialization.
    pub init_herbivores: usize,
    /// If true, scatter a fresh founder cohort whenever the herbivores go extinct
    /// (a tuning-robustness aid; the coexistence gate is measured with it OFF).
    pub reseed_on_extinction: bool,

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
            // Herbivore economy tuned (seeds 42/7/123, 128², 40k ticks) for a
            // robust, self-sustaining plant↔herbivore coexistence: an opening
            // predator-prey cycle (plant bloom → grazer boom → plant crash) that
            // relaxes to a persistent steady state (coverage ~12%, ~1600–2000
            // grazers, mean energy ~3.3). Large energy reserves relative to
            // metabolism keep grazers off the starvation edge, which is what
            // makes coexistence robust across seeds. See DESIGN.md for the curve
            // and the balance logic (why it coexists rather than crashing/exploding).
            graze_cap: 0.28,
            graze_efficiency: 1.0,
            metabolism: 0.02,
            move_cost: 0.015,
            repro_threshold: 7.0,
            energy_max: 10.0,
            herb_init_energy: 5.0,
            corpse_nutrient: 0.5,
            herb_inner_neurons: 12,
            herb_genome_length: 48,
            herb_mutation_rate: 0.02,
            init_herbivores: 150,
            reseed_on_extinction: false,
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
    /// Live herbivore count — the prey axis of the predator-prey readout.
    pub population: u64,
    /// Herbivore births since the previous recorded tick.
    pub births: u64,
    /// Herbivore deaths since the previous recorded tick.
    pub deaths: u64,
    /// Mean herbivore energy (0.0 when the population is empty).
    pub mean_energy: f64,
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

/// Domain tag mixed into every herbivore RNG stream so a herbivore at index `k`
/// on tick `t` never draws the same numbers as the *plant* cell at index `k`
/// (whose stream is `cell_seed`) — the two subsystems stay decorrelated.
const HERB_TAG: u64 = 0x4845_5242_1000_0001;

/// Per-herbivore RNG seed for the serial entity phase: the analogue of the
/// generational sim's `agent_seed`, keyed by `(seed, tick, order-index)`. Index
/// is the herbivore's position in the stable `order` at the tick's start, so the
/// stream is reproducible regardless of how many draws each brain makes.
fn herbivore_seed(master: u64, tick: u64, index: usize) -> u64 {
    let s = crate::mix(master ^ HERB_TAG);
    let s = crate::mix(s ^ tick);
    crate::mix(s ^ index as u64)
}

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

/// The continuous producer-base simulation over a shared cell [`Grid`], now with
/// a mobile herbivore layer (rung 2) living as hecs entities.
pub struct EcoSim {
    grid: Grid,
    /// Double-buffer scratch: next tick's `(nutrient, biomass)` per cell.
    next: Vec<(f32, f32)>,
    /// Herbivores live here as entities: a [`Position`] component + a
    /// [`Herbivore`] bundle. hecs archetype iteration is not stable across
    /// despawns, so every determinism-sensitive pass goes through `order`.
    ecs: hecs::World,
    /// The stable birth-order of live herbivores — the determinism backbone,
    /// mirroring the challenge sim. Position in `order` is the per-entity RNG
    /// index: pushed on birth, `retain`-culled on death (survivors keep their
    /// relative order), so an entity's index only ever decreases.
    order: Vec<Entity>,
    tick: u64,
    config: EcoConfig,
    metrics_out: Option<File>,
    /// Every recorded tick's metrics (drives the live viewer sparklines).
    history: Vec<EcoMetrics>,
    /// Record metrics (history + JSONL) every this-many ticks. 1 for the
    /// determinism/headless runs; the viewer bumps it to bound memory.
    metrics_interval: u64,
    /// Births / deaths accumulated since the last recorded metrics tick (so the
    /// counts are correct even when `metrics_interval > 1`).
    births_accum: u64,
    deaths_accum: u64,
}

impl EcoSim {
    pub fn new(config: EcoConfig) -> EcoSim {
        let n = config.width * config.height;
        EcoSim {
            grid: Grid::new(config.width, config.height),
            next: vec![(0.0, 0.0); n],
            ecs: hecs::World::new(),
            order: Vec::new(),
            tick: 0,
            config,
            metrics_out: None,
            history: Vec::new(),
            metrics_interval: 1,
            births_accum: 0,
            deaths_accum: 0,
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
        self.seed_herbivores();
        self.record_metrics();
    }

    /// Scatter the founder herbivore cohort onto random empty cells, each with a
    /// fresh random genome (like the challenge sim's founders) and a founder
    /// lineage id. Uses a dedicated `herbivore_seed`-derived stream so it never
    /// aliases the plant founder scatter. A no-op when `init_herbivores == 0`
    /// (the pure producer-base regression tests rely on that).
    fn seed_herbivores(&mut self) {
        let p = self.config.params.clone();
        if p.init_herbivores == 0 {
            return;
        }
        let (w, h) = (self.grid.width as u32, self.grid.height as u32);
        let mut rng = ChaCha8Rng::seed_from_u64(herbivore_seed(self.config.seed, INIT_TAG, 0));
        let mut placed = 0usize;
        // Bounded attempts so a near-full grid can't spin forever.
        let mut attempts = 0usize;
        let max_attempts = p.init_herbivores.saturating_mul(200).max(10_000);
        while placed < p.init_herbivores && attempts < max_attempts {
            attempts += 1;
            let pos = (rng.random_range(0..w), rng.random_range(0..h));
            if self.grid.blocked(pos) {
                continue;
            }
            let genome: Vec<u32> =
                (0..p.herb_genome_length).map(|_| rng.random::<u32>()).collect();
            let e = self.spawn_herbivore(
                pos,
                genome,
                p.herb_init_energy,
                placed as u32,
                p.herb_inner_neurons,
            );
            self.order.push(e);
            placed += 1;
        }
    }

    /// Build a herbivore entity at `pos` (decoding its brain from the genome),
    /// mark the cell occupied, and return the new entity. Does **not** append to
    /// `order` — callers control ordering (founders push directly; births are
    /// appended after the entity phase so they act next tick).
    fn spawn_herbivore(
        &mut self,
        pos: (u32, u32),
        genome: Vec<u32>,
        energy: f32,
        lineage: u32,
        amt_inners: u8,
    ) -> Entity {
        let brain = Brain::from(genome.clone(), HERB_INPUTS, amt_inners);
        let e = self.ecs.spawn((
            Position { x: pos.0, y: pos.1 },
            Herbivore { genome, brain, energy, lineage, amt_inners },
        ));
        self.grid.set_occupant(pos, e);
        e
    }

    /// Read a herbivore's grid position (copied out, so no ECS borrow escapes).
    #[inline]
    fn herb_pos(&self, e: Entity) -> (u32, u32) {
        let p = self.ecs.get::<&Position>(e).expect("herbivore has Position");
        (p.x, p.y)
    }

    /// Read a herbivore's current energy.
    #[inline]
    fn herb_energy(&self, e: Entity) -> f32 {
        self.ecs.get::<&Herbivore>(e).expect("herbivore has Herbivore").energy
    }

    /// Build the herbivore's sensor input vector at `pos` (see [`HERB_INPUTS`]
    /// for the layout). Pure gather over the grid snapshot; `rand01` is the
    /// pre-drawn random-input value from the entity's per-tick RNG.
    fn sense(&self, pos: (u32, u32), energy: f32, rand01: f32) -> Vec<f32> {
        let (w, h) = (self.grid.width as i32, self.grid.height as i32);
        let bm = self.config.params.biomass_max.max(1e-6);
        let emax = self.config.params.energy_max.max(1e-6);
        let cells = &self.grid.cells;
        // Biomass at a cell, treating off-grid as barren.
        let biomass_at = |x: i32, y: i32| -> f32 {
            if x < 0 || y < 0 || x >= w || y >= h {
                0.0
            } else {
                cells[y as usize * w as usize + x as usize].biomass
            }
        };
        // "Blocked" for a neighbor: off-grid edges count as blocked, as does an
        // obstacle or another creature standing there.
        let blocked_at = |x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= w || y >= h {
                true
            } else {
                cells[y as usize * w as usize + x as usize].blocked()
            }
        };
        let (x, y) = (pos.0 as i32, pos.1 as i32);
        let grad_ns = ((biomass_at(x, y + 1) - biomass_at(x, y - 1)) / bm).clamp(-1.0, 1.0);
        let grad_ew = ((biomass_at(x + 1, y) - biomass_at(x - 1, y)) / bm).clamp(-1.0, 1.0);
        let mut occ = 0.0f32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if blocked_at(x + dx, y + dy) {
                    occ += 1.0;
                }
            }
        }
        let b = |cond: bool| if cond { 1.0 } else { 0.0 };
        vec![
            1.0,
            (self.tick % 2) as f32,
            (energy / emax).clamp(0.0, 1.0),
            rand01,
            (biomass_at(x, y) / bm).clamp(0.0, 1.0),
            grad_ns,
            grad_ew,
            b(blocked_at(x, y + 1)),
            b(blocked_at(x, y - 1)),
            b(blocked_at(x + 1, y)),
            b(blocked_at(x - 1, y)),
            occ / 8.0,
        ]
    }

    /// Pick a random empty (unblocked, on-grid) cell from the 8-neighborhood of
    /// `pos`, drawing from `rng` (deterministic). `None` if the herbivore is
    /// boxed in — reproduction is then skipped this tick.
    fn empty_neighbor(&self, pos: (u32, u32), rng: &mut ChaCha8Rng) -> Option<(u32, u32)> {
        let (w, h) = (self.grid.width as i32, self.grid.height as i32);
        let (x, y) = (pos.0 as i32, pos.1 as i32);
        let mut opts: Vec<(u32, u32)> = Vec::with_capacity(8);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                if !self.grid.cells[ny as usize * w as usize + nx as usize].blocked() {
                    opts.push((nx as u32, ny as u32));
                }
            }
        }
        if opts.is_empty() {
            None
        } else {
            Some(opts[rng.random_range(0..opts.len())])
        }
    }

    /// The serial, order-stable entity phase (rung 2). For each live herbivore in
    /// birth order: sense → brain → move (costs energy, blocked by the grid) →
    /// graze the current cell (biomass → energy, up to `graze_cap` and the
    /// satiation headroom) → pay metabolism → then die (deposit a corpse nutrient
    /// pulse) or reproduce (spawn a mutated child on an empty neighbor, splitting
    /// energy). Runs *after* the parallel field update, single-threaded, so it
    /// never races the CA and every RNG draw is reproducible.
    fn herbivore_phase(&mut self) {
        let p = self.config.params.clone();
        let (seed, tick) = (self.config.seed, self.tick);
        let w = self.grid.width;
        let (max_x, max_y) = (self.grid.max_x(), self.grid.max_y());

        // Iterate a fixed prefix: children appended this tick (index ≥ n) act
        // next tick, and `order` is not mutated until after the loop, so
        // `self.order[idx]` and the RNG index `idx` are stable throughout.
        let n = self.order.len();
        let mut dead: Vec<Entity> = Vec::new();
        let mut births: Vec<Entity> = Vec::new();

        for idx in 0..n {
            let e = self.order[idx];
            let pos = self.herb_pos(e);
            let mut rng = ChaCha8Rng::seed_from_u64(herbivore_seed(seed, tick, idx));
            let rand01 = rng.random::<f32>();
            let inputs = self.sense(pos, self.herb_energy(e), rand01);
            // Brain step in a scoped mutable borrow (dropped before any spawn).
            let translation = {
                let mut hb = self.ecs.get::<&mut Herbivore>(e).expect("herbivore exists");
                hb.brain.step(inputs, &mut rng)
            };

            // --- movement: only onto an empty, in-bounds cell; costs energy ---
            let target = (
                (pos.0 as i32 + translation.0).clamp(0, max_x) as u32,
                (pos.1 as i32 + translation.1).clamp(0, max_y) as u32,
            );
            let mut energy = self.herb_energy(e);
            let mut cur = pos;
            if target != pos && !self.grid.blocked(target) {
                self.grid.clear_occupant(pos);
                self.grid.set_occupant(target, e);
                let mut pc = self.ecs.get::<&mut Position>(e).expect("herbivore has Position");
                pc.x = target.0;
                pc.y = target.1;
                cur = target;
                energy -= p.move_cost;
            }

            // --- graze the current cell (throttled to satiation headroom) ---
            let ci = cur.1 as usize * w + cur.0 as usize;
            let avail = self.grid.cells[ci].biomass;
            let headroom = (p.energy_max - energy).max(0.0);
            let by_headroom = if p.graze_efficiency > 1e-6 {
                headroom / p.graze_efficiency
            } else {
                0.0
            };
            let grazed = avail.min(p.graze_cap).min(by_headroom);
            if grazed > 0.0 {
                self.grid.cells[ci].biomass = avail - grazed;
                energy += grazed * p.graze_efficiency;
            }

            // --- metabolism (the starvation clock) ---
            energy -= p.metabolism;

            // --- death: despawn + return the body to the soil (closes the loop) ---
            if energy <= 0.0 {
                self.grid.clear_occupant(cur);
                self.grid.cells[ci].nutrient += p.corpse_nutrient + energy.max(0.0);
                dead.push(e);
                continue;
            }

            // --- reproduction: split energy onto a mutated child, if room ---
            if energy >= p.repro_threshold
                && let Some(child_cell) = self.empty_neighbor(cur, &mut rng)
            {
                let child_energy = energy * 0.5;
                energy -= child_energy;
                let (pgenome, lineage, amt) = {
                    let hb = self.ecs.get::<&Herbivore>(e).expect("parent exists");
                    (hb.genome.clone(), hb.lineage, hb.amt_inners)
                };
                let child_genome = mutate_genome(&pgenome, p.herb_mutation_rate, &mut rng);
                let child = self.spawn_herbivore(child_cell, child_genome, child_energy, lineage, amt);
                births.push(child);
            }

            // Commit the (surviving) parent's energy, clamped to the satiation cap.
            let mut hb = self.ecs.get::<&mut Herbivore>(e).expect("herbivore exists");
            hb.energy = energy.min(p.energy_max);
        }

        // Apply deaths (despawn + drop from `order`, preserving survivor order).
        if !dead.is_empty() {
            for &e in &dead {
                let _ = self.ecs.despawn(e);
            }
            let dset: HashSet<Entity> = dead.iter().copied().collect();
            self.order.retain(|e| !dset.contains(e));
        }
        // Append this tick's newborns in birth (parent-processing) order.
        self.order.extend(&births);

        self.births_accum += births.len() as u64;
        self.deaths_accum += dead.len() as u64;

        // Optional robustness aid (off for the coexistence gate): if the
        // herbivores died out, scatter a fresh founder cohort.
        if self.order.is_empty() && p.reseed_on_extinction {
            self.seed_herbivores();
        }
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

        // Phase 3: the serial, order-stable herbivore entity phase. Runs on the
        // freshly-grown meadow, mutating biomass (grazing), nutrient (corpses),
        // and occupancy — never concurrently with the parallel CA above.
        self.herbivore_phase();

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

        // Herbivore aggregates, summed serially over `order` so the f64 total is
        // order-deterministic (byte-identical across same-seed runs).
        let population = self.order.len() as u64;
        let total_energy: f64 = self.order.iter().map(|&e| self.herb_energy(e) as f64).sum();
        let mean_energy = if population > 0 { total_energy / population as f64 } else { 0.0 };

        let m = EcoMetrics {
            tick: self.tick,
            total_biomass,
            coverage: vegetated as f64 / denom,
            mean_nutrient: total_nutrient / denom,
            population,
            births: self.births_accum,
            deaths: self.deaths_accum,
            mean_energy,
        };
        // Births/deaths are reported per recording interval, so reset the running
        // counters once folded into a record.
        self.births_accum = 0;
        self.deaths_accum = 0;
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

    /// Live herbivore count (the prey axis of the predator-prey readout).
    pub fn population(&self) -> usize {
        self.order.len()
    }

    /// A render snapshot (position + lineage + energy fraction) per herbivore, in
    /// stable `order` sequence — the viewer draws creatures from this without
    /// touching the ECS.
    pub fn herbivore_views(&self) -> Vec<HerbView> {
        let emax = self.config.params.energy_max.max(1e-6);
        self.order
            .iter()
            .map(|&e| {
                let pos = self.herb_pos(e);
                let hb = self.ecs.get::<&Herbivore>(e).expect("herbivore exists");
                HerbView { pos, lineage: hb.lineage, energy_frac: (hb.energy / emax).clamp(0.0, 1.0) }
            })
            .collect()
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

    /// Spawn one herbivore at `(x, y)` with an explicit genome + energy and
    /// append it to `order` (so the entity phase processes it). Test-only escape
    /// hatch for the rung-2 unit tests, mirroring `set_cell`.
    #[cfg(test)]
    fn test_spawn_herbivore(&mut self, x: u32, y: u32, genome: Vec<u32>, energy: f32) -> Entity {
        let amt = self.config.params.herb_inner_neurons;
        let e = self.spawn_herbivore((x, y), genome, energy, 0, amt);
        self.order.push(e);
        e
    }

    #[cfg(test)]
    fn is_live(&self, e: Entity) -> bool {
        self.ecs.contains(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seed: u64, w: usize, h: usize) -> EcoConfig {
        EcoConfig { width: w, height: h, seed, params: EcoParams::default() }
    }

    /// Freeze the plant/nutrient CA (zero every reaction + diffusion term) so a
    /// herbivore test can isolate grazing/energy/death from field dynamics.
    fn freeze_plants(p: &mut EcoParams) {
        p.diffusion = 0.0;
        p.replenish = 0.0;
        p.uptake = 0.0;
        p.decay = 0.0;
        p.mortality = 0.0;
        p.seed_rate = 0.0;
        p.init_herbivores = 0; // only manually-placed herbivores in these tests
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
        // Pure producer base (herbivores off), so this stays a regression guard
        // on rung 1's self-regulation independent of the grazing layer.
        let mut c = cfg(42, 96, 96);
        c.params.init_herbivores = 0;
        let mut sim = EcoSim::new(c);
        sim.seed_initial();
        for _ in 0..3000 {
            sim.tick();
        }
        let cov = sim.latest().unwrap().coverage;
        assert!(cov > 0.10, "meadow collapsed (coverage {cov})");
        assert!(cov < 0.85, "meadow filled (coverage {cov})");
        assert!(sim.latest().unwrap().total_biomass > 0.0, "no standing biomass");
    }

    // ----- rung 2: herbivore grazing / energy / death / reproduction -----

    #[test]
    fn grazing_converts_biomass_to_energy_and_depletes_the_cell() {
        // A non-moving herbivore (empty genome ⇒ no outputs fire) on a vegetated
        // cell grazes up to `graze_cap`, gaining that much energy (efficiency 1)
        // and removing exactly that much biomass; metabolism is the only other
        // term. Plants frozen so the cell's biomass change is grazing alone.
        let mut c = cfg(1, 8, 8);
        freeze_plants(&mut c.params);
        c.params.graze_cap = 0.10;
        c.params.graze_efficiency = 1.0;
        c.params.metabolism = 0.02;
        c.params.move_cost = 0.0;
        c.params.repro_threshold = 100.0; // out of reach ⇒ no reproduction
        c.params.energy_max = 100.0; // ample headroom ⇒ graze isn't throttled
        let mut sim = EcoSim::new(c);
        sim.set_cell(4, 4, 0.5, 0.30);
        let e = sim.test_spawn_herbivore(4, 4, vec![], 1.0);
        sim.tick();
        let (_, b1) = sim.cell_at(4, 4);
        assert!((b1 - 0.20).abs() < 1e-4, "grazing should remove 0.10 biomass: -> {b1}");
        let en = sim.herb_energy(e);
        assert!((en - 1.08).abs() < 1e-4, "energy = 1.0 + 0.10 grazed - 0.02 metabolism: {en}");
    }

    #[test]
    fn starvation_kills_at_zero_energy_and_deposits_a_corpse() {
        // With no food, metabolism drives energy ≤ 0; the herbivore despawns and
        // its body energy (here just `corpse_nutrient`) is added to the cell's
        // nutrient — closing the loop back to the soil.
        let mut c = cfg(2, 8, 8);
        freeze_plants(&mut c.params);
        c.params.metabolism = 0.05;
        c.params.move_cost = 0.0;
        c.params.corpse_nutrient = 0.40;
        let mut sim = EcoSim::new(c);
        sim.set_cell(3, 3, 0.0, 0.0);
        let e = sim.test_spawn_herbivore(3, 3, vec![], 0.03);
        assert_eq!(sim.population(), 1);
        sim.tick();
        assert_eq!(sim.population(), 0, "starved herbivore should die");
        assert!(!sim.is_live(e), "dead herbivore must be despawned");
        let (n, _) = sim.cell_at(3, 3);
        assert!((n - 0.40).abs() < 1e-4, "death should deposit corpse_nutrient: {n}");
    }

    /// Config for the reproduction tests: frozen plants, energy conserved
    /// (metabolism/move free), a low reproduction threshold, deterministic
    /// (mutation-free) children.
    fn repro_cfg(seed: u64) -> EcoConfig {
        let mut c = cfg(seed, 12, 12);
        freeze_plants(&mut c.params);
        c.params.metabolism = 0.0;
        c.params.move_cost = 0.0;
        c.params.repro_threshold = 1.0;
        c.params.energy_max = 3.0;
        c.params.herb_mutation_rate = 0.0;
        c
    }

    #[test]
    fn reproduction_splits_energy_onto_an_empty_neighbor() {
        // Above threshold with a free neighbor ⇒ one child, energy split in half.
        let mut sim = EcoSim::new(repro_cfg(3));
        let parent = sim.test_spawn_herbivore(6, 6, vec![], 1.5);
        sim.tick();
        assert_eq!(sim.population(), 2, "should spawn one child");
        let ep = sim.herb_energy(parent);
        assert!((ep - 0.75).abs() < 1e-4, "parent keeps half its energy: {ep}");
        let child = sim.order[1];
        let ec = sim.herb_energy(child);
        assert!((ec - 0.75).abs() < 1e-4, "child gets the other half: {ec}");
    }

    #[test]
    fn reproduction_needs_an_empty_neighbor() {
        // Boxed in by obstacles on all 8 neighbors ⇒ no room, so no child even
        // though energy is above threshold.
        let mut sim = EcoSim::new(repro_cfg(4));
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                sim.grid.set_obstacle(((6 + dx) as u32, (6 + dy) as u32));
            }
        }
        sim.test_spawn_herbivore(6, 6, vec![], 1.5);
        sim.tick();
        assert_eq!(sim.population(), 1, "a boxed-in herbivore cannot reproduce");
    }

    #[test]
    fn order_is_stable_across_a_birth_and_a_death_tick() {
        // In one tick: A reproduces, B starves, C survives. `order` must drop B
        // (survivors keep their relative slots) and append A's newborn last.
        let mut c = cfg(5, 16, 16);
        freeze_plants(&mut c.params);
        c.params.metabolism = 0.05;
        c.params.move_cost = 0.0;
        c.params.repro_threshold = 1.0;
        c.params.energy_max = 3.0;
        c.params.herb_mutation_rate = 0.0;
        let mut sim = EcoSim::new(c);
        let a = sim.test_spawn_herbivore(2, 2, vec![], 2.0); // reproduces
        let b = sim.test_spawn_herbivore(9, 9, vec![], 0.03); // starves
        let cc = sim.test_spawn_herbivore(5, 5, vec![], 0.5); // survives, no repro
        sim.tick();
        assert!(!sim.is_live(b), "B should have starved");
        assert_eq!(sim.population(), 3, "A + C survive, plus A's newborn");
        assert_eq!(sim.order[0], a, "A keeps its birth slot");
        assert_eq!(sim.order[1], cc, "C shifts up into B's freed slot, order preserved");
        assert!(sim.order[2] != a && sim.order[2] != cc, "the newborn is appended last");
    }

    #[test]
    fn herbivore_run_is_deterministic() {
        // The whole rung-2 sim (parallel field CA + serial entity phase) is
        // byte-identical across two same-seed runs: fields *and* the herbivore
        // metrics (population, biomass, mean energy).
        let run = || {
            let mut sim = EcoSim::new(cfg(2024, 64, 64));
            sim.seed_initial();
            for _ in 0..300 {
                sim.tick();
            }
            let fields: Vec<(u32, u32)> = sim
                .cells()
                .iter()
                .map(|c| (c.nutrient.to_bits(), c.biomass.to_bits()))
                .collect();
            let metrics: Vec<(u64, u64, u64, u64, u64)> = sim
                .metrics_history()
                .iter()
                .map(|m| {
                    (m.population, m.births, m.deaths, m.total_biomass.to_bits(), m.mean_energy.to_bits())
                })
                .collect();
            (fields, metrics)
        };
        let a = run();
        let b = run();
        assert!(a.0 == b.0, "fields diverged under an identical seed");
        assert!(a.1 == b.1, "herbivore metrics diverged under an identical seed");
    }
}
