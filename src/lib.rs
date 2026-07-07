use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hecs::Entity;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, Connection, mutate_genome};
use crate::grid::Grid;

pub mod agent;
pub mod eco;
pub mod grid;

/// A creature's grid position, kept as its own hecs component: spatial queries
/// (the decision phase, centroids, survival) touch it every step, far more often
/// than the brain, so it stays a small standalone component.
#[derive(Clone, Copy)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

/// A read-only per-agent snapshot for the viewer, materialized in stable `order`
/// sequence. Keeps the render code off the ECS internals (and out of hecs borrow
/// guards) on the hot crowd path — position plus the two color inputs.
pub struct AgentView {
    pub pos: (u32, u32),
    pub rgba: [u8; 4],
    pub lineage: u32,
}

/// Which simulation to run. `challenge` is the generational neural-net evolution
/// sim (the default, unchanged); `eco` is the continuous, generation-less
/// terrarium (rung 1: the nutrient + plant producer base, see [`crate::eco`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum Mode {
    #[default]
    Challenge,
    Eco,
}

/// Command-line configuration shared by both binaries (viewer + headless).
/// Defaults reproduce the original hardcoded values.
#[derive(Parser)]
#[command(about = "Genome-encoded neural-net evolution simulator")]
pub struct Cli {
    /// Which simulation to run: the generational `challenge` sim (default) or
    /// the continuous `eco` terrarium.
    #[arg(long, value_enum, default_value_t = Mode::Challenge)]
    pub mode: Mode,
    /// Eco mode only: stop after this many ticks (headless; default: run
    /// indefinitely). Ignored in challenge mode (which uses `--generations`).
    #[arg(long)]
    pub ticks: Option<u64>,
    /// Number of agents per generation.
    #[arg(long, default_value_t = 1000)]
    pub population: u32,
    /// Number of genes (connections) per genome.
    #[arg(long, default_value_t = 256)]
    pub genome_length: u32,
    /// Number of inner neurons in each brain.
    #[arg(long, default_value_t = 225)]
    pub inner_neurons: u32,
    /// Per-bit genome mutation probability.
    #[arg(long, default_value_t = 0.001)]
    pub mutation_rate: f32,
    /// Simulation steps per generation.
    #[arg(long, default_value_t = 200)]
    pub steps: u32,
    /// Stop after this many generations (default: run indefinitely).
    #[arg(long)]
    pub generations: Option<u32>,
    /// RNG seed. If omitted, one is drawn from entropy and printed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Write per-generation metrics as JSONL to this path.
    #[arg(long)]
    pub metrics: Option<PathBuf>,
    /// Save the best genome + config to this path (overwriting) periodically.
    #[arg(long)]
    pub save_champion: Option<PathBuf>,
    /// Generation interval for --save-champion.
    #[arg(long, default_value_t = 50)]
    pub champion_interval: u32,
    /// Seed the initial population with mutated copies of a saved champion's genome.
    #[arg(long)]
    pub load_genome: Option<PathBuf>,
    /// Selection environment (obstacle layout + survival predicate).
    #[arg(long, value_enum, default_value_t = Challenge::NorthBand)]
    pub challenge: Challenge,
    /// Side length of the (square) world grid. Defaults to 128; the challenges
    /// hardcode 128-grid coordinates, so scaling is not advertised yet — the
    /// representation is size-parametric but the survival zones are not.
    #[arg(long, default_value_t = 128)]
    pub world_size: u32,
    /// Headless only: run 50 generations and print gens/sec, then exit.
    #[arg(long)]
    pub bench: bool,
}

impl Cli {
    pub fn to_config(&self, seed: u64) -> SimConfig {
        SimConfig {
            population: self.population,
            genome_length: self.genome_length,
            amount_inners: self.inner_neurons,
            mutation_rate: self.mutation_rate,
            steps_per_generation: self.steps,
            seed,
            challenge: self.challenge,
            world_size: self.world_size,
        }
    }
}

/// Return the seed to use: the explicit one, or a fresh entropy seed that is
/// printed so the run can be reproduced later.
pub fn resolve_seed(seed: Option<u64>) -> u64 {
    match seed {
        Some(s) => s,
        None => {
            let s = rand::rng().random::<u64>();
            println!("seed: {s}");
            s
        }
    }
}

/// Build a fully-configured Simulator from parsed CLI args (metrics, champion
/// saving, and optional genome loading all wired up). Does not yet generate the
/// initial generation.
pub fn build_simulator(cli: &Cli) -> Simulator {
    let seed = resolve_seed(cli.seed);
    let mut sim = Simulator::new(cli.to_config(seed));

    if let Some(path) = &cli.load_genome {
        let data = std::fs::read_to_string(path).expect("failed to read --load-genome file");
        let champ: Champion =
            serde_json::from_str(&data).expect("failed to parse champion JSON");
        sim.set_seed_genome(champ.genome);
    }
    if let Some(path) = &cli.metrics {
        sim.set_metrics(path).expect("failed to open --metrics file");
    }
    if let Some(path) = &cli.save_champion {
        sim.set_champion(path.clone(), cli.champion_interval);
    }
    sim
}

/// Build a configured eco-mode [`EcoSim`] from parsed CLI args (world size,
/// seed, metrics). Reuses `resolve_seed` and `--world-size`; the meadow's
/// dynamics are the tuned [`eco::EcoParams`] defaults. Does not seed the initial
/// meadow — call [`eco::EcoSim::seed_initial`] after.
pub fn build_eco_sim(cli: &Cli) -> eco::EcoSim {
    let seed = resolve_seed(cli.seed);
    let size = cli.world_size as usize;
    let mut params = eco::EcoParams::default();
    apply_eco_env_overrides(&mut params);
    let mut sim = eco::EcoSim::new(eco::EcoConfig {
        width: size,
        height: size,
        seed,
        params,
    });
    if let Some(path) = &cli.metrics {
        sim.set_metrics(path).expect("failed to open --metrics file");
    }
    sim
}

/// Dev-only herbivore-tuning overrides read from the environment, applied only on
/// the binary path (never by `EcoSim::new`, so the determinism tests and any
/// unset run use the tuned [`eco::EcoParams`] defaults verbatim). Lets a headless
/// parameter sweep explore the coexistence balance without a recompile, e.g.
/// `ECO_GRAZE_CAP=0.3 ECO_METABOLISM=0.02 cargo run --bin headless -- --mode eco`.
fn apply_eco_env_overrides(p: &mut eco::EcoParams) {
    let f32v = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let usizev = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    if let Some(v) = f32v("ECO_GRAZE_CAP") {
        p.graze_cap = v;
    }
    if let Some(v) = f32v("ECO_GRAZE_EFF") {
        p.graze_efficiency = v;
    }
    if let Some(v) = f32v("ECO_METABOLISM") {
        p.metabolism = v;
    }
    if let Some(v) = f32v("ECO_MOVE_COST") {
        p.move_cost = v;
    }
    if let Some(v) = f32v("ECO_REPRO") {
        p.repro_threshold = v;
    }
    if let Some(v) = f32v("ECO_ENERGY_MAX") {
        p.energy_max = v;
    }
    if let Some(v) = f32v("ECO_INIT_ENERGY") {
        p.herb_init_energy = v;
    }
    if let Some(v) = f32v("ECO_CORPSE") {
        p.corpse_nutrient = v;
    }
    if let Some(v) = usizev("ECO_INIT_HERB") {
        p.init_herbivores = v;
    }
    if std::env::var("ECO_RESEED").is_ok() {
        p.reseed_on_extinction = true;
    }
}

/// Y coordinate an agent must exceed at generation end to survive and
/// reproduce. The barrier obstacle sits on this row.
pub const SURVIVAL_Y: u32 = 108;

/// Grid center used by the position-based pattern challenges (ring, heart,
/// orbit). The 128×128 grid's cells run 0..=127; 64 is one of the four central
/// cells (chosen and used consistently).
const CENTER: (i32, i32) = (64, 64);

/// `flock`: max distance from the population's final centroid to survive. Kept
/// tight (a single blob) yet ≥ ~18, the radius a disk needs to actually hold all
/// ~1000 collision-separated agents, so full convergence stays feasible.
const FLOCK_RADIUS: i32 = 20;
/// `ring`: inner/outer radii of the survival annulus about the center. A
/// hollow ring; the hole (r < RING_R_IN) is what forces the pattern.
const RING_R_IN: i32 = 22;
const RING_R_OUT: i32 = 42;
/// `heart`: scale (grid cells per normalized unit) of the embedded heart curve.
/// Larger = bigger heart. Tuned so the shape reads clearly and gen-0 survival
/// stays in a climbable range.
const HEART_SCALE: f32 = 34.0;
/// `orbit`: radius band the agent must end within, and the minimum absolute
/// swept angle required to count as having orbited. A half-turn (π) is the
/// natural target but leaves gen-0 near 0.2% (fragile); 2.0 rad (~115°) is still
/// clearly sustained rotation while lifting the gen-0 floor for a robust climb.
const ORBIT_R_IN: i32 = 14;
const ORBIT_R_OUT: i32 = 50;
const ORBIT_THETA: f32 = 2.0;

/// Squared distance from a cell to the grid center — the shared kernel for the
/// radius-based pattern challenges (avoids a sqrt).
fn dist2_center(pos: (u32, u32)) -> i32 {
    let dx = pos.0 as i32 - CENTER.0;
    let dy = pos.1 as i32 - CENTER.1;
    dx * dx + dy * dy
}

/// The signed angle (radians, wrapped to (−π, π]) swept about the grid center
/// as a cell moves from `old` to `new`. Single-cell moves usually subtend a
/// small angle; the wrap covers the rare large sweep close to the center. The
/// center is offset by 0.5 so no cell sits exactly on it (atan2 stays defined).
fn swept_angle(old: (u32, u32), new: (u32, u32)) -> f32 {
    let cx = CENTER.0 as f32 - 0.5;
    let cy = CENTER.1 as f32 - 0.5;
    let a0 = (old.1 as f32 - cy).atan2(old.0 as f32 - cx);
    let a1 = (new.1 as f32 - cy).atan2(new.0 as f32 - cx);
    let mut d = a1 - a0;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// `orbit` survival: a completed half-turn (by accumulated signed angle) and a
/// final radius inside the band. Kept as a free function so the unit test and
/// the simulator's dynamic-survival path share one definition.
fn orbit_survives(pos: (u32, u32), accumulated_angle: f32) -> bool {
    let r2 = dist2_center(pos);
    accumulated_angle.abs() >= ORBIT_THETA
        && (ORBIT_R_IN * ORBIT_R_IN..=ORBIT_R_OUT * ORBIT_R_OUT).contains(&r2)
}

/// A rectangle of obstacle cells: ((x0, y0), (x1, y1)), inclusive on both ends.
type Rect = ((u32, u32), (u32, u32));

/// The selection environment for a run. Each variant defines two things: the
/// obstacle layout stamped into the 128x128 world at the start of every
/// generation, and the survival predicate applied to each agent's final
/// position. Both are pure functions of the generation number (or constant), so
/// runs stay bit-for-bit reproducible. Selection reuses the existing positional
/// mechanism — no energy, food, or per-step scoring is involved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Challenge {
    /// The original behavior: survive iff final `y > 108`, with a single solid
    /// horizontal barrier spanning x∈[10,118] on row y=108. Trivial to solve
    /// (evolution just learns "go north"); kept as the default so existing runs
    /// are unchanged.
    #[default]
    NorthBand,
    /// Survive iff the final position lies within radius 10 of any of the four
    /// world corners. A 33x33 obstacle block fills the center (x,y∈[48,80]) to
    /// discourage clumping and give the directional sensors something to read.
    /// Selects for dispersal and a committed directional preference.
    Corners,
    /// Survive iff final `y > 108`, but the barrier on row y=108 spans the full
    /// width x∈[0,127] with only two narrow gaps (width 3, centered near x=40
    /// and x=88). Unlike NorthBand there are no open flanks: the only way north
    /// is through a gap, so an agent starting below must locate and thread one.
    /// Pure "go north" drift rarely lands on a gap within the step budget, so
    /// this selects for real navigation off the directional obstacle sensors.
    Gauntlet,
    /// Survive iff the final `y` is within 10 of a band center that oscillates
    /// with the generation number: `center = round(64 + 24*sin(0.25*gen))`,
    /// giving a height-21 safe band whose midline sweeps y∈[40,88]. No
    /// obstacles. The target moves between generations and agents have no
    /// absolute-position sense, so a hardcoded "go north" fails; it selects for
    /// robust centering strategies.
    MovingBand,
    /// Survive iff the final position is strictly inside a central walled box:
    /// walls of thickness 1 form the square x,y∈[44,84] with a width-15 entrance
    /// gap in the bottom wall (x∈[57,71], y=44). Safe interior is x,y∈[45,83].
    /// Selects for seeking the box and threading the single entrance, using the
    /// directional sensors to follow walls.
    Enclosure,
    /// Pattern-forming (dynamic): survive iff the final position lies within
    /// `FLOCK_RADIUS` of the population's own *final* centroid (the mean of all
    /// agents' end positions, computed once at generation end). There is no fixed
    /// zone — the target is wherever the crowd gathers — so it selects purely for
    /// convergence into a single tight blob. Needs the self-position sensor to
    /// steer toward the crowd rather than drift.
    Flock,
    /// Pattern-forming (static): survive iff the distance from the grid center is
    /// in the annulus `[RING_R_IN, RING_R_OUT]`. Selects the swarm into a hollow
    /// ring — agents must hold a target radius, neither collapsing to the center
    /// nor fleeing to the edge.
    Ring,
    /// Pattern-forming (static): survive iff the final position lies inside an
    /// embedded heart, defined analytically by the implicit curve
    /// `(nx²+ny²−1)³ − nx²·ny³ ≤ 0` over grid coordinates normalized about the
    /// center (y flipped so the heart sits upright). The showpiece: the swarm is
    /// sculpted into a filled heart. `HEART_SCALE` sets its size.
    Heart,
    /// Pattern-forming (dynamic): survive iff the agent has swept at least
    /// `ORBIT_THETA` radians of signed angle around the grid center over the
    /// generation *and* ends in the radius band `[ORBIT_R_IN, ORBIT_R_OUT]`.
    /// Rewards sustained rotation rather than a spiral in/out, selecting for a
    /// pinwheel. Uses each agent's per-generation accumulated-angle path state.
    Orbit,
}

/// Append the solid segments of a full-width barrier on row `y` (x∈[0,127])
/// with the given inclusive `gaps` left open, to `out`.
fn barrier_row(y: u32, gaps: &[(u32, u32)], out: &mut Vec<Rect>) {
    let mut x = 0u32;
    for &(g0, g1) in gaps {
        if x < g0 {
            out.push(((x, y), (g0 - 1, y)));
        }
        x = g1 + 1;
    }
    if x <= 127 {
        out.push(((x, y), (127, y)));
    }
}

impl Challenge {
    /// Human-readable name for the status panel.
    pub fn name(self) -> &'static str {
        match self {
            Challenge::NorthBand => "north-band",
            Challenge::Corners => "corners",
            Challenge::Gauntlet => "gauntlet",
            Challenge::MovingBand => "moving-band",
            Challenge::Enclosure => "enclosure",
            Challenge::Flock => "flock",
            Challenge::Ring => "ring",
            Challenge::Heart => "heart",
            Challenge::Orbit => "orbit",
        }
    }

    /// Obstacle rectangles present in the world for `generation`.
    fn obstacles(self, _generation: u32) -> Vec<Rect> {
        match self {
            Challenge::NorthBand => vec![((10, SURVIVAL_Y), (118, SURVIVAL_Y))],
            Challenge::Corners => vec![((48, 48), (80, 80))],
            Challenge::Gauntlet => {
                // Full-width barrier on row SURVIVAL_Y with two width-3 gaps
                // (centered near x=40 and x=88). The narrow openings are the
                // only way north, forcing navigation to a gap.
                let mut segs = Vec::new();
                barrier_row(SURVIVAL_Y, &[(39, 41), (87, 89)], &mut segs);
                segs
            }
            Challenge::MovingBand => vec![],
            Challenge::Enclosure => vec![
                ((44, 84), (84, 84)), // top wall
                ((44, 44), (44, 84)), // left wall
                ((84, 44), (84, 84)), // right wall
                ((44, 44), (56, 44)), // bottom wall, left of entrance
                ((72, 44), (84, 44)), // bottom wall, right of entrance
            ],
            // The pattern-forming challenges shape the swarm with the survival
            // predicate alone — no obstacles.
            Challenge::Flock | Challenge::Ring | Challenge::Heart | Challenge::Orbit => vec![],
        }
    }

    /// Midline of the MovingBand safe zone at `generation` (deterministic).
    fn moving_band_center(generation: u32) -> u32 {
        (64.0 + 24.0 * (generation as f64 * 0.25).sin()).round() as u32
    }

    /// Does an agent ending at `pos` in `generation` survive?
    fn survives(self, pos: (u32, u32), generation: u32) -> bool {
        match self {
            Challenge::NorthBand | Challenge::Gauntlet => pos.1 > SURVIVAL_Y,
            Challenge::Corners => {
                const R2: i32 = 10 * 10;
                let corners = [(0i32, 0i32), (0, 127), (127, 0), (127, 127)];
                corners.iter().any(|&(cx, cy)| {
                    let dx = pos.0 as i32 - cx;
                    let dy = pos.1 as i32 - cy;
                    dx * dx + dy * dy <= R2
                })
            }
            Challenge::MovingBand => {
                let center = Challenge::moving_band_center(generation) as i32;
                (pos.1 as i32 - center).abs() <= 10
            }
            Challenge::Enclosure => {
                (45..=83).contains(&pos.0) && (45..=83).contains(&pos.1)
            }
            Challenge::Ring => {
                let d2 = dist2_center(pos);
                (RING_R_IN * RING_R_IN..=RING_R_OUT * RING_R_OUT).contains(&d2)
            }
            Challenge::Heart => {
                // Normalized coordinates about the center, y flipped so the
                // heart sits upright (lobes up, point down) on screen. Inside
                // the implicit heart curve => survive (a filled heart).
                let nx = (pos.0 as f32 - CENTER.0 as f32) / HEART_SCALE;
                let ny = (CENTER.1 as f32 - pos.1 as f32) / HEART_SCALE;
                let t = nx * nx + ny * ny - 1.0;
                t * t * t - nx * nx * ny * ny * ny <= 0.0
            }
            // Dynamic challenges: survival depends on more than a single final
            // position (Flock on the crowd's centroid, Orbit on per-agent swept
            // angle), so it is resolved in `Simulator::eval_survival`, not here.
            // Returning false gives them no static zone — which is exactly right
            // for spawn-exclusion (nothing to exclude) and the viewer tint (no
            // fixed target to draw).
            Challenge::Flock | Challenge::Orbit => false,
        }
    }
}

/// Number of random agent pairs sampled when measuring genome diversity.
const DIVERSITY_SAMPLE_PAIRS: usize = 200;

/// The parameters that fully define a reproducible run: same config + same
/// build produce the same metrics byte-for-byte. Serialized alongside a saved
/// champion genome so a run can always be reconstructed.
#[derive(Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub population: u32,
    pub genome_length: u32,
    pub amount_inners: u32,
    pub mutation_rate: f32,
    pub steps_per_generation: u32,
    pub seed: u64,
    /// Selection environment. Defaults to `NorthBand` when absent from an older
    /// serialized champion, so those files still load.
    #[serde(default)]
    pub challenge: Challenge,
    /// Side length of the square world grid. Defaults to 128 for older champions
    /// (and the current default), which is what the challenge coordinates assume.
    #[serde(default = "default_world_size")]
    pub world_size: u32,
}

/// Default world side length (128), used for the `serde` default so champions
/// serialized before `world_size` existed still load.
fn default_world_size() -> u32 {
    128
}

/// A saved champion: the best agent's genome plus the config that produced it.
/// Neuron state is intentionally not serialized — the brain is a pure function
/// of the genome.
#[derive(Serialize, Deserialize)]
pub struct Champion {
    pub config: SimConfig,
    pub genome: Vec<u32>,
}

/// One JSONL record per generation.
///
/// `genome_diversity` is the mean pairwise Hamming distance (in bits, range
/// 0..genome_length*32) over `DIVERSITY_SAMPLE_PAIRS` randomly sampled agent
/// pairs, drawn from a dedicated RNG so measurement never perturbs the sim.
#[derive(Clone, Serialize)]
pub struct GenerationMetrics {
    pub generation: u32,
    pub survivors: u32,
    pub survival_rate: f64,
    pub extinction: bool,
    pub mean_final_y: f64,
    pub max_final_y: u32,
    pub genome_diversity: f64,
    pub wall_ms: f64,
}

/// SplitMix64 finalizer — mixes an integer into a well-distributed seed. Shared
/// with the eco module's per-cell RNG derivation.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Derive a deterministic per-agent seed from the run seed and the position in
/// the simulation (generation, step, agent index). This is what makes the
/// parallel decision and reproduction phases reproducible without a shared RNG.
fn agent_seed(master: u64, generation: u32, step: u32, index: usize) -> u64 {
    let mut s = mix(master);
    s = mix(s ^ generation as u64);
    s = mix(s ^ step as u64);
    mix(s ^ index as u64)
}

/// Step tag used to seed the reproduction phase, kept distinct from any real
/// step index (which is always < steps_per_generation).
const REPRO_TAG: u32 = u32::MAX;
/// Step tag used to seed diversity sampling, distinct from real steps.
const DIVERSITY_TAG: u32 = u32::MAX - 1;

pub struct Simulator {
    /// The spatial substrate: a typed cell grid (obstacles + occupancy) that
    /// replaced the old `Vec<u128>` column bitmask. Private; the viewer builds
    /// its terrain overlay from `obstacles` + `is_safe`, never from raw cells.
    grid: Grid,
    /// Creatures live here as entities: a `Position` component + an `Agent`
    /// bundle (brain/genome/...). hecs archetype iteration is *not* stable
    /// across despawns, so all determinism-sensitive iteration goes through
    /// `order` instead of querying the world directly.
    ecs: hecs::World,
    /// The stable birth-order of live creatures — the crown jewel of
    /// determinism. Position in `order` is the per-agent RNG index (identical to
    /// the old `Vec<Agent>` index): pushed on spawn, `retain`-culled so
    /// survivors keep their relative order, children appended in parent order.
    order: Vec<Entity>,
    pub generation: u32,
    current_steps: u32,
    config: SimConfig,
    move_vectors: Vec<(i32, i32)>,
    /// Obstacle rectangles currently stamped into the world (recomputed from the
    /// challenge each generation). Exposed so the viewer can render walls.
    pub obstacles: Vec<Rect>,
    master_rng: ChaCha8Rng,
    gen_start: Instant,
    /// Optional genome to seed the initial population from (mutated copies).
    seed_genome: Option<Vec<u32>>,
    metrics_out: Option<File>,
    /// In-memory per-generation metrics for the whole run so far, recorded
    /// unconditionally (independent of `--metrics`). Small structs; a full run's
    /// worth is a few KB. The viewer reads this for its live charts.
    history: Vec<GenerationMetrics>,
    champion_path: Option<PathBuf>,
    champion_interval: u32,
    /// The previous generation's final state: each agent's last position paired
    /// with whether it survived, captured at turnover *before* the losers are
    /// culled and the next generation spawns. Purely for the viewer's turnover
    /// pulse; does not affect the sim or determinism.
    last_final: Vec<((u32, u32), bool)>,
}

impl Simulator {
    pub fn new(config: SimConfig) -> Simulator {
        let master_rng = ChaCha8Rng::seed_from_u64(config.seed);
        let size = config.world_size as usize;
        Simulator {
            grid: Grid::new(size, size),
            ecs: hecs::World::new(),
            order: Vec::new(),
            generation: 0,
            current_steps: 0,
            move_vectors: vec![
                (0, 1),
                (0, -1),
                (1, 0),
                (1, 1),
                (1, -1),
                (-1, 0),
                (-1, 1),
                (-1, -1),
            ],
            // Obstacle layout for generation 0, per the configured challenge.
            obstacles: config.challenge.obstacles(0),
            master_rng,
            gen_start: Instant::now(),
            seed_genome: None,
            metrics_out: None,
            history: Vec::new(),
            champion_path: None,
            champion_interval: 0,
            last_final: Vec::new(),
            config,
        }
    }

    /// Open (truncating) a JSONL metrics file; one line is written per generation.
    pub fn set_metrics(&mut self, path: &Path) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        self.metrics_out = Some(file);
        Ok(())
    }

    /// Save the best genome + config to `path` every `interval` generations.
    pub fn set_champion(&mut self, path: PathBuf, interval: u32) {
        self.champion_path = Some(path);
        self.champion_interval = interval;
    }

    /// Seed the initial population from mutated copies of a loaded genome.
    pub fn set_seed_genome(&mut self, genome: Vec<u32>) {
        self.seed_genome = Some(genome);
    }

    /// Per-generation metrics recorded so far (always populated). The viewer's
    /// survival/diversity charts read from this.
    pub fn metrics_history(&self) -> &[GenerationMetrics] {
        &self.history
    }

    /// The previous generation's final `(position, survived)` pairs, captured at
    /// turnover before culling. Empty until the first generation completes. The
    /// viewer uses this to flash survivors/culled at their final positions.
    pub fn last_final(&self) -> &[((u32, u32), bool)] {
        &self.last_final
    }

    /// The active selection environment.
    pub fn challenge(&self) -> Challenge {
        self.config.challenge
    }

    /// Whether `pos` lies in the survival zone for the current generation — the
    /// pure survival predicate the viewer tints the world overlay from.
    pub fn is_safe(&self, pos: (u32, u32)) -> bool {
        self.config.challenge.survives(pos, self.generation)
    }

    // ----- viewer-facing accessors (the viewer never touches the ECS) -----

    /// The live creatures in stable birth order. A viewer selection index maps
    /// into this slice; entry `order()[i]` is the entity for order position `i`.
    pub fn order(&self) -> &[Entity] {
        &self.order
    }

    /// Current agent positions in `order` sequence (for motion trails / heat).
    pub fn positions(&self) -> Vec<(u32, u32)> {
        self.order.iter().map(|&e| self.pos_of(e)).collect()
    }

    /// A render snapshot (position + colors) per agent, in `order` sequence.
    pub fn agent_views(&self) -> Vec<AgentView> {
        self.order
            .iter()
            .map(|&e| {
                let pos = self.pos_of(e);
                let a = self.ecs.get::<&Agent>(e).expect("live entity has Agent");
                AgentView { pos, rgba: a.get_rgba(), lineage: a.lineage }
            })
            .collect()
    }

    /// The brain wiring of one entity (for the inspector's node-link diagram),
    /// copied out so the viewer holds no ECS borrow. `None` if it despawned.
    pub fn agent_connections(&self, e: Entity) -> Option<Vec<Connection>> {
        self.ecs.get::<&Agent>(e).ok().map(|a| a.brain_connections().to_vec())
    }

    /// The live neuron activations of one entity (read every frame for the
    /// selected agent), copied out so the viewer holds no ECS borrow.
    pub fn agent_neurons(&self, e: Entity) -> Option<Vec<Vec<f32>>> {
        self.ecs.get::<&Agent>(e).ok().map(|a| a.brain_neurons().to_vec())
    }

    /// The lineage id of one entity (for the brain panel subtitle).
    pub fn agent_lineage(&self, e: Entity) -> Option<u32> {
        self.ecs.get::<&Agent>(e).ok().map(|a| a.lineage)
    }

    // ----- internal ECS helpers -----

    /// Position of a live entity. Panics if the entity has no `Position`, which
    /// would be an internal bug (every creature is spawned with one).
    #[inline]
    fn pos_of(&self, e: Entity) -> (u32, u32) {
        let p = self.ecs.get::<&Position>(e).expect("live entity has Position");
        (p.x, p.y)
    }

    /// Overwrite a live entity's position component.
    #[inline]
    fn set_pos(&mut self, e: Entity, pos: (u32, u32)) {
        let mut p = self.ecs.get::<&mut Position>(e).expect("live entity has Position");
        p.x = pos.0;
        p.y = pos.1;
    }

    /// Attach a fully-built `Agent` to a fresh entity at a reserved unique cell,
    /// appending it to `order`. Used by the reproduction phase, whose children
    /// are decoded in parallel; here only the position reservation (one master
    /// RNG draw sequence) is serial. The entity is spawned with a placeholder
    /// position first so `rand_pos` can record the real occupant id.
    fn spawn_prebuilt(&mut self, agent: Agent) -> Entity {
        let e = self.ecs.spawn((Position { x: 0, y: 0 },));
        let pos = self.rand_pos(e);
        self.ecs.insert_one(e, agent).expect("just-spawned entity exists");
        self.set_pos(e, pos);
        self.order.push(e);
        e
    }

    /// Spawn one creature whose genome is drawn *after* its position, preserving
    /// the original master-RNG interleaving (position draws, then genome draws)
    /// for the serial initial-generation and extinction-reseed paths.
    /// `make_genome` runs after `rand_pos`, receiving `self` so it can pull from
    /// the master RNG (random founder) or a per-agent RNG (seeded from a loaded
    /// champion) as the caller chooses.
    fn spawn_drawn(&mut self, lineage: u32, make_genome: impl FnOnce(&mut Self) -> Vec<u32>) {
        let e = self.ecs.spawn((Position { x: 0, y: 0 },));
        let pos = self.rand_pos(e);
        let genome = make_genome(self);
        let agent = Agent::new(&genome, self.config.amount_inners as u8, lineage);
        self.ecs.insert_one(e, agent).expect("just-spawned entity exists");
        self.set_pos(e, pos);
        self.order.push(e);
    }

    fn random_genome(&mut self) -> Vec<u32> {
        (0..self.config.genome_length)
            .map(|_| self.master_rng.random::<u32>())
            .collect()
    }

    pub fn generate_initial_generation(&mut self) {
        let seed_genome = self.seed_genome.clone();
        let seed = self.config.seed;
        let mutation_rate = self.config.mutation_rate;
        for i in 0..self.config.population {
            // Lineage id = founder index at the initial generation. The genome is
            // drawn *after* the position (inside `spawn_drawn`) to preserve the
            // original master-RNG draw order.
            match &seed_genome {
                Some(g) => {
                    let g = g.clone();
                    self.spawn_drawn(i, move |_s| {
                        let mut rng = ChaCha8Rng::seed_from_u64(agent_seed(
                            seed,
                            0,
                            REPRO_TAG,
                            i as usize,
                        ));
                        mutate_genome(&g, mutation_rate, &mut rng)
                    });
                }
                None => self.spawn_drawn(i, |s| s.random_genome()),
            }
        }
        self.add_obstacles();
        self.gen_start = Instant::now();
    }

    fn spawn_next_generation(&mut self) {
        let lived_gen = self.generation;
        // Resolve survival once for the generation just finished — including the
        // dynamic challenges' whole-population context (flock's centroid, orbit's
        // per-agent swept angle) — and reuse the same flags for metrics, the
        // viewer snapshot, and culling. Order matches `self.order`.
        let survived = self.eval_survival(lived_gen);

        // Record metrics / champion for the generation that just finished.
        self.record_generation(&survived);

        // Snapshot final positions + survival before culling, for the viewer's
        // turnover pulse. Cheap: one (u32,u32,bool) per agent, overwritten each
        // generation. Independent of the RNG, so determinism is unaffected.
        self.last_final.clear();
        let finals: Vec<((u32, u32), bool)> = self
            .order
            .iter()
            .zip(&survived)
            .map(|(&e, &s)| (self.pos_of(e), s))
            .collect();
        self.last_final = finals;
        self.generation += 1;

        // Cull the losers (survivors keep their relative order), then reset the
        // grid to obstacles-only for the new generation.
        self.cull_to_survivors(&survived);

        // Extinction: no survivors means there is nothing to reproduce from.
        // Reseed the generation with fresh random genomes instead of crashing
        // on a modulo-by-zero. Serial (each genome pulls from the master RNG),
        // position drawn before genome as in the initial generation.
        if self.order.is_empty() {
            for i in 0..self.config.population {
                self.spawn_drawn(i, |s| s.random_genome());
            }
            self.gen_start = Instant::now();
            return;
        }

        // Reproduce. Gather the survivors' parent templates (in `order`), decode
        // the children in parallel (each keyed by its birth index `i`, exactly as
        // the old `par_iter` over positions), then despawn the parents and spawn
        // the children serially so the master-RNG position draws stay in order.
        let len = self.order.len();
        let parents: Vec<(Vec<u32>, u8, u32)> = self
            .order
            .iter()
            .map(|&e| {
                let a = self.ecs.get::<&Agent>(e).expect("survivor has Agent");
                (a.genome.clone(), a.amt_inners(), a.lineage)
            })
            .collect();

        let mutation_rate = self.config.mutation_rate;
        let seed = self.config.seed;
        let generation = self.generation;
        let amt_inners = self.config.amount_inners as u8;
        let children: Vec<Agent> = (0..self.config.population)
            .into_par_iter()
            .map(|i| {
                let (pgenome, _amt, lineage) = &parents[i as usize % len];
                let mut rng =
                    ChaCha8Rng::seed_from_u64(agent_seed(seed, generation, REPRO_TAG, i as usize));
                let genome = mutate_genome(pgenome, mutation_rate, &mut rng);
                Agent::new(&genome, amt_inners, *lineage)
            })
            .collect();

        // The next generation is entirely the children; the survivors were only
        // reproduction templates, so despawn them and rebuild `order`.
        for &e in &self.order {
            self.ecs.despawn(e).expect("despawning a parent template");
        }
        self.order.clear();
        for child in children {
            self.spawn_prebuilt(child);
        }
        self.gen_start = Instant::now();
    }

    /// Reserve a unique spawn cell for `occupant`: unoccupied AND outside the
    /// current challenge's survival zone. Excluding the survival zone stops
    /// agents from spawning on a free win — they must move to earn survival, so
    /// gen-0 numbers reflect behavior rather than lucky placement. Uses the
    /// master seeded RNG (deterministic) and the survival predicate for the
    /// generation the spawned agent will live through (`self.generation`), and
    /// records `occupant` in the reserved cell.
    fn rand_pos(&mut self, occupant: Entity) -> (u32, u32) {
        let challenge = self.config.challenge;
        let generation = self.generation;
        let (max_x, max_y) = (self.grid.max_x() as u32, self.grid.max_y() as u32);
        let mut pos: (u32, u32) = (
            self.master_rng.random_range(0..=max_x),
            self.master_rng.random_range(0..=max_y),
        );
        while self.get_pos(pos) || challenge.survives(pos, generation) {
            pos = (
                self.master_rng.random_range(0..=max_x),
                self.master_rng.random_range(0..=max_y),
            );
        }
        self.grid.set_occupant(pos, occupant);
        pos
    }

    /// Whether a cell is blocked (obstacle or another creature) — the occupancy
    /// test the spawn reservation and collision resolver read.
    fn get_pos(&self, coords: (u32, u32)) -> bool {
        self.grid.blocked(coords)
    }

    /// Despawn the creatures whose `survived[i]` is false, keeping survivors in
    /// their relative birth order — the determinism guarantee: `order` is
    /// filtered in place (never swap-removed), so a survivor's RNG index (its
    /// position in `order`) only ever decreases monotonically, matching the old
    /// `Vec::retain`. Then reset the grid to obstacles for the next generation.
    fn cull_to_survivors(&mut self, survived: &[bool]) {
        let mut survivors = Vec::with_capacity(self.order.len());
        for (&e, &s) in self.order.iter().zip(survived) {
            if s {
                survivors.push(e);
            } else {
                self.ecs.despawn(e).expect("culling a live entity");
            }
        }
        self.order = survivors;
        self.clear_world();
    }

    fn clear_world(&mut self) {
        self.grid.reset();
        self.add_obstacles();
    }

    /// Per-agent survival flags for the just-finished generation, in `self.order`
    /// order. Static challenges defer to the pure `survives` predicate; the
    /// dynamic ones resolve their whole-population context here, once: `flock`
    /// against the crowd's final centroid, `orbit` against each agent's
    /// accumulated swept angle plus final radius band.
    fn eval_survival(&self, generation: u32) -> Vec<bool> {
        match self.config.challenge {
            Challenge::Flock => {
                let c = self.agents_centroid();
                let r2 = FLOCK_RADIUS * FLOCK_RADIUS;
                self.order
                    .iter()
                    .map(|&e| {
                        let p = self.pos_of(e);
                        let dx = p.0 as i32 - c.0 as i32;
                        let dy = p.1 as i32 - c.1 as i32;
                        dx * dx + dy * dy <= r2
                    })
                    .collect()
            }
            Challenge::Orbit => self
                .order
                .iter()
                .map(|&e| {
                    let pos = self.pos_of(e);
                    let angle = self
                        .ecs
                        .get::<&Agent>(e)
                        .expect("live entity has Agent")
                        .accumulated_angle();
                    orbit_survives(pos, angle)
                })
                .collect(),
            challenge => self
                .order
                .iter()
                .map(|&e| challenge.survives(self.pos_of(e), generation))
                .collect(),
        }
    }

    /// Mean position (centroid) of the current agents. `flock`'s survival zone
    /// is a disk about this point, so the target is wherever the crowd gathers.
    /// Integer sum, so it is order-independent.
    fn agents_centroid(&self) -> (u32, u32) {
        let n = self.order.len().max(1) as u32;
        let sum = self.order.iter().fold((0u32, 0u32), |acc, &e| {
            let p = self.pos_of(e);
            (acc.0 + p.0, acc.1 + p.1)
        });
        (sum.0 / n, sum.1 / n)
    }

    /// Advance the simulation one step.
    ///
    /// Split into two phases for parallelism:
    ///   (a) decision phase — parallel over agents, reading the (unmutated) grid;
    ///       each agent computes its sensor inputs and runs `Brain::step` (with a
    ///       deterministic per-agent RNG) to produce a desired translation.
    ///   (b) application phase — serial, in `order` sequence, resolving
    ///       collisions against the live (mutating) grid.
    ///
    /// hecs archetype order is not stable, so the decision phase materializes the
    /// creatures and sorts them into `order` before iterating: the parallel index
    /// `i` is then the position in `order`, identical to the old `Vec<Agent>`
    /// index, which is what makes the per-agent RNG reproducible.
    pub fn step(&mut self) {
        if self.current_steps >= self.config.steps_per_generation {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();
        let move_vectors = self.move_vectors.clone();
        let seed = self.config.seed;
        let generation = self.generation;
        let step = self.current_steps;

        // (a) parallel decision phase. The grid is read (occupancy) but not
        // mutated here — mutation is confined to the serial application below —
        // so the closures can borrow it directly instead of cloning a snapshot.
        let translations: Vec<(i32, i32)> = {
            let index: HashMap<Entity, usize> =
                self.order.iter().enumerate().map(|(i, &e)| (e, i)).collect();
            let grid = &self.grid;
            let mut items: Vec<(Entity, &Position, &mut Agent)> = self
                .ecs
                .query_mut::<(Entity, &Position, &mut Agent)>()
                .into_iter()
                .collect();
            // Reorder into stable birth order so `i` is the RNG index.
            items.sort_by_key(|(e, _, _)| index[e]);
            items
                .par_iter_mut()
                .enumerate()
                .map(|(i, (_e, pos, agent))| {
                    let agent_pos = (pos.x, pos.y);
                    let used = agent.get_used_inputs();
                    let all_inputs =
                        calc_positional_inputs(grid, &move_vectors, agent_pos, &inputs, used);
                    let mut rng =
                        ChaCha8Rng::seed_from_u64(agent_seed(seed, generation, step, i));
                    agent.step(all_inputs, &mut rng)
                })
                .collect()
        };

        // (b) serial application phase, in `order` sequence. Pair each entity
        // (in order) with its translation up front, so the loop owns its data and
        // is free to mutate the grid + position components as it goes.
        // `orbit` is the only challenge that needs path state; accumulate swept
        // angle here (serial => deterministic), and only when it's active.
        let track_angle = self.config.challenge == Challenge::Orbit;
        let (max_x, max_y) = (self.grid.max_x(), self.grid.max_y());
        let moves: Vec<(Entity, (i32, i32))> =
            self.order.iter().copied().zip(translations).collect();
        for (entity, translation) in moves {
            let agent_pos = self.pos_of(entity);
            let pos: (u32, u32) = (
                (agent_pos.0 as i32 + translation.0).clamp(0, max_x) as u32,
                (agent_pos.1 as i32 + translation.1).clamp(0, max_y) as u32,
            );

            if !self.get_pos(pos) {
                self.grid.clear_occupant(agent_pos);
                self.set_pos(entity, pos);
                self.grid.set_occupant(pos, entity);
                if track_angle {
                    self.ecs
                        .get::<&mut Agent>(entity)
                        .expect("live entity has Agent")
                        .add_angle(swept_angle(agent_pos, pos));
                }
            }
        }

        self.current_steps += 1;
    }

    /* INPUTS
    0: always 0
    1: always 1
    2: oscillator
    3: age
    4: random
    5: ns population gradient
    6: ew population gradient
    7-14: can move in x direction
    15: self x (normalized, 0..1)
    16: self y (normalized, 0..1)
    */
    fn calc_step_inputs(&mut self) -> Vec<f32> {
        let mut inputs: Vec<f32> = vec![0.0; 7];

        // Population centroid — an integer sum, so it is order-independent and
        // can be read straight off a query without touching `order`.
        let (mut sx, mut sy, mut count) = (0u32, 0u32, 0u32);
        {
            let mut q = self.ecs.query::<&Position>();
            for p in q.iter() {
                sx += p.x;
                sy += p.y;
                count += 1;
            }
        }
        // max(1) guards the empty-population case without changing the result
        // (sums are then 0, so the average is 0 either way).
        let n = count.max(1);
        let av = (sx / n, sy / n);

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = self.current_steps as f32 / self.config.steps_per_generation as f32;
        inputs[4] = self.master_rng.random_range(0.0..1.0);
        inputs[5] = av.1 as f32 / self.grid.max_y() as f32;
        inputs[6] = av.0 as f32 / self.grid.max_x() as f32;

        inputs
    }

    /// Recompute the challenge's obstacle layout for the current generation and
    /// stamp it into the world.
    fn add_obstacles(&mut self) {
        let obstacles = self.config.challenge.obstacles(self.generation);
        let (w, h) = (self.grid.width, self.grid.height);
        for &((x0, y0), (x1, y1)) in &obstacles {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    // Challenge coordinates assume a 128 grid. On a differently
                    // sized (unadvertised) world, any off-grid obstacle cell is
                    // skipped rather than panicking; at the default 128 every cell
                    // is in bounds, so this guard is a no-op there.
                    if (x as usize) < w && (y as usize) < h {
                        self.grid.set_obstacle((x, y));
                    }
                }
            }
        }
        self.obstacles = obstacles;
    }

    /// Compute + emit metrics for the just-finished generation, and save the
    /// champion if it's on the save interval. Uses `self.generation` (not yet
    /// incremented) and the current agents.
    fn record_generation(&mut self, survived: &[bool]) {
        let wall_ms = self.gen_start.elapsed().as_secs_f64() * 1000.0;
        let n = self.order.len();

        let survivors = survived.iter().filter(|&&s| s).count() as u32;
        // max / sum over final y — order-independent aggregates.
        let max_final_y = self.order.iter().map(|&e| self.pos_of(e).1).max().unwrap_or(0);
        let sum_y: u64 = self.order.iter().map(|&e| self.pos_of(e).1 as u64).sum();
        let mean_final_y = if n > 0 { sum_y as f64 / n as f64 } else { 0.0 };
        let survival_rate = if n > 0 { survivors as f64 / n as f64 } else { 0.0 };

        // Always record into the in-memory history (drives the live viewer),
        // regardless of whether file output is enabled.
        let m = GenerationMetrics {
            generation: self.generation,
            survivors,
            survival_rate,
            extinction: survivors == 0,
            mean_final_y,
            max_final_y,
            genome_diversity: self.genome_diversity(),
            wall_ms,
        };
        if let Some(file) = self.metrics_out.as_mut() {
            // Best-effort: a metrics write failure shouldn't kill a long run.
            if let Ok(line) = serde_json::to_string(&m) {
                let _ = writeln!(file, "{}", line);
            }
        }
        self.history.push(m);

        self.maybe_save_champion();
    }

    /// Mean pairwise Hamming distance (bits) over a random sample of agent
    /// pairs, using a dedicated deterministic RNG so it never perturbs the sim.
    /// Pairs are sampled by `order` index (identical to the old `Vec<Agent>`
    /// index), so the measurement is reproducible for a given seed.
    fn genome_diversity(&self) -> f64 {
        let n = self.order.len();
        if n < 2 {
            return 0.0;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(agent_seed(
            self.config.seed,
            self.generation,
            DIVERSITY_TAG,
            0,
        ));
        let mut total: u64 = 0;
        for _ in 0..DIVERSITY_SAMPLE_PAIRS {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            let ga = self.ecs.get::<&Agent>(self.order[a]).expect("live entity has Agent");
            let gb = self.ecs.get::<&Agent>(self.order[b]).expect("live entity has Agent");
            total += ga
                .genome
                .iter()
                .zip(gb.genome.iter())
                .map(|(x, y)| (x ^ y).count_ones() as u64)
                .sum::<u64>();
        }
        total as f64 / DIVERSITY_SAMPLE_PAIRS as f64
    }

    fn maybe_save_champion(&self) {
        let Some(path) = &self.champion_path else {
            return;
        };
        if self.champion_interval == 0 || !self.generation.is_multiple_of(self.champion_interval) {
            return;
        }
        // Highest final y wins; iterating `order` keeps the tie-break stable.
        let Some(&best) = self.order.iter().max_by_key(|&&e| self.pos_of(e).1) else {
            return;
        };
        let genome = self
            .ecs
            .get::<&Agent>(best)
            .expect("live entity has Agent")
            .genome
            .clone();
        let champion = Champion {
            config: self.config.clone(),
            genome,
        };
        if let Ok(json) = serde_json::to_string_pretty(&champion) {
            let _ = std::fs::write(path, json);
        }
    }
}

/// Build an agent's full input vector: the shared base inputs (7), the 8
/// directional "can I move here" sensors, and the 2 self-position sensors, read
/// against the grid (occupancy is not mutated during the decision phase, so the
/// grid can be shared across the parallel closures). Total width 17.
fn calc_positional_inputs(
    grid: &Grid,
    move_vectors: &[(i32, i32)],
    pos: (u32, u32),
    base: &[f32],
    used: Vec<usize>,
) -> Vec<f32> {
    let mut copy = base.to_vec();
    copy.extend_from_slice(&[0.0; 10]);
    let (max_x, max_y) = (grid.max_x(), grid.max_y());

    // Obstacle sensors 7..=14 are computed lazily — only for the ids the brain
    // actually reads (`used`). The position sensors 15/16 can also appear in
    // `used`; they are filled unconditionally below, so skip them (and anything
    // else out of the directional range) here to keep the move_vectors index in
    // bounds.
    for i in used {
        if !(7..=14).contains(&i) {
            continue;
        }
        let vec: (i32, i32) = move_vectors[i - 7];
        let neighbor = (
            (pos.0 as i32 + vec.0).clamp(0, max_x) as u32,
            (pos.1 as i32 + vec.1).clamp(0, max_y) as u32,
        );
        copy[i] = if grid.blocked(neighbor) { 0.0 } else { 1.0 };
    }

    // Self-position sensors, always populated (position is always "used", unlike
    // the lazy obstacle sensors): normalized (x, y) so a brain can navigate to
    // absolute coordinates rather than only follow the crowd or walls. At the
    // default 128 grid `max_x`/`max_y` are 127, matching the original /127.0.
    copy[15] = pos.0 as f32 / max_x as f32;
    copy[16] = pos.1 as f32 / max_y as f32;

    copy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(seed: u64) -> SimConfig {
        SimConfig {
            population: 200,
            genome_length: 64,
            amount_inners: 32,
            mutation_rate: 0.001,
            steps_per_generation: 50,
            seed,
            challenge: Challenge::NorthBand,
            world_size: 128,
        }
    }

    /// A world with the given challenge's obstacles stamped in, used to check
    /// obstacle cells are marked occupied.
    fn world_for(challenge: Challenge) -> Simulator {
        let mut cfg = test_config(1);
        cfg.challenge = challenge;
        let mut sim = Simulator::new(cfg);
        sim.add_obstacles();
        sim
    }

    #[test]
    fn challenge_predicates_and_layouts_are_consistent() {
        // For each challenge: a point in the safe zone survives, a point outside
        // does not, and every obstacle cell is marked occupied in the world.
        // (challenge, inside/survives, outside/dies)
        type Case = (Challenge, (u32, u32), (u32, u32));
        let cases: &[Case] = &[
            (Challenge::NorthBand, (64, 120), (64, 50)),
            (Challenge::Corners, (2, 2), (64, 64)),
            (Challenge::Gauntlet, (64, 120), (64, 50)),
            (Challenge::MovingBand, (64, Challenge::moving_band_center(0)), (64, 5)),
            (Challenge::Enclosure, (64, 64), (5, 5)),
        ];
        for &(challenge, inside, outside) in cases {
            assert!(
                challenge.survives(inside, 0),
                "{challenge:?}: safe-zone point {inside:?} should survive"
            );
            assert!(
                !challenge.survives(outside, 0),
                "{challenge:?}: outside point {outside:?} should not survive"
            );
            let sim = world_for(challenge);
            for ((x0, y0), (x1, y1)) in challenge.obstacles(0) {
                for x in x0..=x1 {
                    for y in y0..=y1 {
                        assert!(
                            sim.get_pos((x, y)),
                            "{challenge:?}: obstacle cell ({x},{y}) not occupied"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gauntlet_gaps_are_open_and_barrier_blocks() {
        // The three gap centers on the barrier row are passable; the barrier is
        // solid everywhere else, including the flanks (x=0 and x=127) that
        // NorthBand leaves open. This is what forces navigation to a gap.
        let sim = world_for(Challenge::Gauntlet);
        for gap_x in [40u32, 88] {
            assert!(!sim.get_pos((gap_x, SURVIVAL_Y)), "gap at x={gap_x} is blocked");
        }
        assert!(sim.get_pos((10, SURVIVAL_Y)), "barrier between gaps should be solid");
        assert!(sim.get_pos((64, SURVIVAL_Y)), "center should be solid between the two gaps");
        assert!(sim.get_pos((0, SURVIVAL_Y)), "west flank should be solid (no open edge)");
        assert!(sim.get_pos((127, SURVIVAL_Y)), "east flank should be solid (no open edge)");
    }

    #[test]
    fn enclosure_entrance_is_open() {
        // The bottom-wall entrance gap is passable; the rest of the bottom wall
        // is solid, so agents must thread the single opening.
        let sim = world_for(Challenge::Enclosure);
        assert!(!sim.get_pos((64, 44)), "entrance cell should be open");
        assert!(sim.get_pos((50, 44)), "bottom wall beside entrance should be solid");
    }

    #[test]
    fn champion_json_round_trips_challenge() {
        // A saved champion carries its challenge through serialization, and an
        // older champion file with no `challenge` field loads as NorthBand.
        let mut cfg = test_config(3);
        cfg.challenge = Challenge::Gauntlet;
        let champ = Champion { config: cfg, genome: vec![1, 2, 3] };
        let json = serde_json::to_string(&champ).unwrap();
        assert!(json.contains("\"challenge\":\"gauntlet\""), "kebab-case challenge in JSON: {json}");
        let back: Champion = serde_json::from_str(&json).unwrap();
        assert_eq!(back.config.challenge, Challenge::Gauntlet);

        let legacy = r#"{"config":{"population":1,"genome_length":1,"amount_inners":1,"mutation_rate":0.001,"steps_per_generation":1,"seed":1},"genome":[0]}"#;
        let back: Champion = serde_json::from_str(legacy).unwrap();
        assert_eq!(back.config.challenge, Challenge::NorthBand, "missing challenge defaults to NorthBand");
    }

    #[test]
    fn spawns_never_land_in_survival_zone() {
        // The spawn-exclusion fix: no agent may start already satisfying the
        // survival predicate for the generation it will live through. Checked
        // for every challenge across the initial generation.
        for challenge in [
            Challenge::NorthBand,
            Challenge::Corners,
            Challenge::Gauntlet,
            Challenge::MovingBand,
            Challenge::Enclosure,
            Challenge::Flock,
            Challenge::Ring,
            Challenge::Heart,
            Challenge::Orbit,
        ] {
            let mut cfg = test_config(7);
            cfg.challenge = challenge;
            let mut sim = Simulator::new(cfg);
            sim.generate_initial_generation();
            for pos in sim.positions() {
                assert!(
                    !challenge.survives(pos, 0),
                    "{challenge:?}: agent spawned inside survival zone at {pos:?}"
                );
            }
        }
    }

    #[test]
    fn gauntlet_north_sensor_reads_barrier() {
        // In the actual Gauntlet world, an agent just below the barrier reads
        // its north directional sensor (id 7 -> move_vectors[0] = (0,1)) as 0.0
        // where the barrier is solid and 1.0 where a gap sits above — confirming
        // the directional obstacle sensors fire against the challenge's walls.
        let sim = world_for(Challenge::Gauntlet);
        let move_vectors = sim.move_vectors.clone();
        let base = vec![0.0f32; 7];
        // (10,107): solid barrier cell directly north -> blocked (0.0).
        let blocked =
            calc_positional_inputs(&sim.grid, &move_vectors, (10, 107), &base, vec![7]);
        assert_eq!(blocked[7], 0.0, "north sensor should read blocked below solid barrier");
        // (40,107): a gap sits directly north -> open (1.0).
        let open =
            calc_positional_inputs(&sim.grid, &move_vectors, (40, 107), &base, vec![7]);
        assert_eq!(open[7], 1.0, "north sensor should read open below a gap");
    }

    #[test]
    fn moving_band_center_tracks_generation() {
        // The band midline is a deterministic, non-constant function of the
        // generation (so "go north" can't be hardcoded).
        let c0 = Challenge::moving_band_center(0);
        let c_shifted = Challenge::moving_band_center(6);
        assert_eq!(c0, 64, "gen-0 band should be centered");
        assert_ne!(c0, c_shifted, "band center should move across generations");
    }

    #[test]
    fn collision_blocks_movement_onto_occupied_cells() {
        // Every spawned position is marked in the world; an agent can never be
        // placed on top of another, and obstacle cells stay occupied.
        let mut sim = Simulator::new(test_config(1));
        sim.generate_initial_generation();

        // No two agents share a cell.
        let mut seen = std::collections::HashSet::new();
        for pos in sim.positions() {
            assert!(seen.insert(pos), "duplicate agent position");
        }
        // The barrier row cells are occupied (obstacle present).
        for x in 10..=118u32 {
            assert!(sim.get_pos((x, SURVIVAL_Y)), "obstacle cell not set");
        }

        // Run a bunch of steps; the invariant "occupied cells are unique" must
        // hold — the application phase only moves onto empty cells.
        for _ in 0..30 {
            sim.step();
            let mut seen = std::collections::HashSet::new();
            for pos in sim.positions() {
                assert!(seen.insert(pos), "two agents on one cell after step");
            }
        }
    }

    #[test]
    fn directional_sensors_respond_to_walls() {
        // Sensor id 9 maps to move_vectors[2] = (1, 0) (east neighbor). It reads
        // 1.0 in open space and 0.0 when a wall (or agent) occupies that cell —
        // confirming the directional obstacle sensors fire near walls, which is
        // what Gauntlet/Enclosure rely on for navigation.
        let move_vectors = vec![
            (0, 1),
            (0, -1),
            (1, 0),
            (1, 1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1),
        ];
        let base = vec![0.0f32; 7];
        let pos = (64u32, 64u32);

        let open = Grid::new(128, 128);
        let inputs = calc_positional_inputs(&open, &move_vectors, pos, &base, vec![9]);
        assert_eq!(inputs[9], 1.0, "east sensor should read 1.0 in open space");

        let mut walled = Grid::new(128, 128);
        walled.set_obstacle((65, 64)); // wall at (65, 64), the east neighbor
        let inputs = calc_positional_inputs(&walled, &move_vectors, pos, &base, vec![9]);
        assert_eq!(inputs[9], 0.0, "east sensor should read 0.0 against a wall");
    }

    #[test]
    fn lineage_assigned_at_founding_and_inherited() {
        // Founders get their index as lineage id; a child inherits its parent's
        // lineage verbatim (unchanged by mutation).
        let mut sim = Simulator::new(test_config(99));
        sim.generate_initial_generation();
        for (i, &e) in sim.order.iter().enumerate() {
            let lineage = sim.ecs.get::<&Agent>(e).unwrap().lineage;
            assert_eq!(lineage, i as u32, "founder {i} should have lineage {i}");
        }
        let parent = sim.ecs.get::<&Agent>(sim.order[7]).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let child = parent.produce_child(0.5, &mut rng);
        assert_eq!(child.lineage, parent.lineage, "child must inherit parent lineage");
    }

    #[test]
    fn metrics_history_records_without_file_output() {
        // History is populated every generation even with no --metrics file.
        let mut sim = Simulator::new(test_config(5));
        sim.generate_initial_generation();
        while sim.generation < 3 {
            sim.step();
        }
        assert_eq!(sim.metrics_history().len(), 3, "one record per completed generation");
        assert_eq!(sim.metrics_history()[0].generation, 0);
    }

    #[test]
    fn same_seed_is_deterministic() {
        let mut a = Simulator::new(test_config(12345));
        let mut b = Simulator::new(test_config(12345));
        a.generate_initial_generation();
        b.generate_initial_generation();
        for _ in 0..120 {
            a.step();
            b.step();
        }
        let pa = a.positions();
        let pb = b.positions();
        assert_eq!(pa, pb, "same seed diverged");
    }

    #[test]
    fn different_seed_differs() {
        let mut a = Simulator::new(test_config(1));
        let mut b = Simulator::new(test_config(2));
        a.generate_initial_generation();
        b.generate_initial_generation();
        for _ in 0..120 {
            a.step();
            b.step();
        }
        let pa = a.positions();
        let pb = b.positions();
        assert_ne!(pa, pb, "different seeds produced identical state");
    }

    #[test]
    fn position_sensors_report_normalized_pos() {
        // Widening added inputs 15/16 = self (x, y) / 127, filled unconditionally
        // (independent of `used`). The eight obstacle sensors (ids 7..=14) are
        // unaffected and still keyed off `used`.
        let move_vectors = vec![
            (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1),
        ];
        let base = vec![0.0f32; 7];
        let world = Grid::new(128, 128);
        let pos = (32u32, 96u32);
        let inputs = calc_positional_inputs(&world, &move_vectors, pos, &base, vec![]);
        assert_eq!(inputs.len(), 17, "input vector should be widened to 17");
        assert_eq!(inputs[15], 32.0 / 127.0, "input 15 = self x / 127");
        assert_eq!(inputs[16], 96.0 / 127.0, "input 16 = self y / 127");
    }

    #[test]
    fn ring_survives_only_in_the_annulus() {
        // A point mid-annulus survives; the hollow center and the region past
        // the outer radius both die.
        let c = Challenge::Ring;
        let mid = (RING_R_IN + RING_R_OUT) / 2;
        assert!(
            c.survives(((CENTER.0 + mid) as u32, CENTER.1 as u32), 0),
            "mid-annulus point should survive"
        );
        assert!(
            !c.survives((CENTER.0 as u32, CENTER.1 as u32), 0),
            "the hollow center should die"
        );
        assert!(
            !c.survives(((CENTER.0 + RING_R_OUT + 6) as u32, CENTER.1 as u32), 0),
            "outside the outer radius should die"
        );
    }

    #[test]
    fn heart_contains_center_and_excludes_far_corner() {
        // The implicit heart is filled and contains the grid center; a far
        // corner lies well outside it.
        let c = Challenge::Heart;
        assert!(
            c.survives((CENTER.0 as u32, CENTER.1 as u32), 0),
            "grid center should be inside the heart"
        );
        assert!(!c.survives((2, 2), 0), "far corner should be outside the heart");
    }

    #[test]
    fn orbit_requires_half_turn_and_radius_band() {
        // In the radius band with a completed half-turn (either sign) survives;
        // too little swept angle fails; outside the band fails even if orbited.
        let mid_r = (ORBIT_R_IN + ORBIT_R_OUT) / 2;
        let in_band = ((CENTER.0 + mid_r) as u32, CENTER.1 as u32);
        assert!(orbit_survives(in_band, ORBIT_THETA + 0.1), "half-turn in band survives");
        assert!(orbit_survives(in_band, -(ORBIT_THETA + 0.1)), "sign of the sweep shouldn't matter");
        assert!(!orbit_survives(in_band, ORBIT_THETA - 0.5), "insufficient swept angle fails");
        let outside = ((CENTER.0 + ORBIT_R_OUT + 5) as u32, CENTER.1 as u32);
        assert!(!orbit_survives(outside, ORBIT_THETA + 1.0), "outside the band fails even if well-orbited");
    }

    #[test]
    fn flock_survival_tracks_the_crowd_centroid() {
        // Flock's survival zone is a disk about the population's own centroid.
        // A dense crowd at the center pins the centroid there: an agent in the
        // blob survives, one in the far corner does not.
        let mut cfg = test_config(1);
        cfg.challenge = Challenge::Flock;
        let mut sim = Simulator::new(cfg);
        let genome = vec![0u32; 8];
        // 21 agents in the blob at the center (indices 0..=20), one far away
        // (index 21). Spawn straight into the ECS + order; eval_survival reads
        // positions and the crowd centroid, not the grid, so occupancy is moot.
        for i in 0..22u32 {
            let p = if i == 21 { (120, 120) } else { (64, 64) };
            let e = sim.ecs.spawn((Position { x: p.0, y: p.1 }, Agent::new(&genome, 8, 0)));
            sim.order.push(e);
        }
        let survived = sim.eval_survival(0);
        assert!(survived[20], "an agent inside the crowd blob should survive flock");
        assert!(!survived[21], "an agent far from the crowd centroid should not");
    }

    #[test]
    fn orbit_run_is_deterministic() {
        // The new per-agent swept-angle path state is updated in the serial
        // phase, so two identically seeded orbit runs must match bit-for-bit,
        // positions and accumulated angles alike.
        let mk = || {
            let mut c = test_config(2024);
            c.challenge = Challenge::Orbit;
            Simulator::new(c)
        };
        let mut a = mk();
        let mut b = mk();
        a.generate_initial_generation();
        b.generate_initial_generation();
        for _ in 0..120 {
            a.step();
            b.step();
        }
        let finals = |s: &Simulator| -> Vec<((u32, u32), u32)> {
            s.order
                .iter()
                .map(|&e| {
                    let pos = s.pos_of(e);
                    let ang = s.ecs.get::<&Agent>(e).unwrap().accumulated_angle().to_bits();
                    (pos, ang)
                })
                .collect()
        };
        assert_eq!(finals(&a), finals(&b), "orbit run diverged under an identical seed");
    }

    #[test]
    fn order_is_stable_across_a_cull() {
        // The cull must preserve survivors' relative birth order (so the RNG
        // index stays put) and despawn exactly the losers. Keep evens, drop odds.
        let mut sim = Simulator::new(test_config(3));
        sim.generate_initial_generation();
        let before: Vec<Entity> = sim.order.clone();
        let survived: Vec<bool> = (0..before.len()).map(|i| i % 2 == 0).collect();

        sim.cull_to_survivors(&survived);

        let expected: Vec<Entity> = before
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, &e)| e)
            .collect();
        assert_eq!(sim.order, expected, "cull must preserve survivors' relative order");
        // Every survivor is still live; every culled entity is gone.
        for (i, &e) in before.iter().enumerate() {
            assert_eq!(
                sim.ecs.contains(e),
                i % 2 == 0,
                "entity at index {i} has wrong liveness after cull"
            );
        }
        assert_eq!(sim.ecs.len() as usize, expected.len(), "world holds untracked entities");
    }
}
