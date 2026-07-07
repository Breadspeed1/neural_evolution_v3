use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, mutate_genome};

pub mod agent;

/// Command-line configuration shared by both binaries (viewer + headless).
/// Defaults reproduce the original hardcoded values.
#[derive(Parser)]
#[command(about = "Genome-encoded neural-net evolution simulator")]
pub struct Cli {
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

/// Y coordinate an agent must exceed at generation end to survive and
/// reproduce. The barrier obstacle sits on this row.
pub const SURVIVAL_Y: u32 = 108;

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
#[derive(Serialize)]
struct GenerationMetrics {
    generation: u32,
    survivors: u32,
    survival_rate: f64,
    extinction: bool,
    mean_final_y: f64,
    max_final_y: u32,
    genome_diversity: f64,
    wall_ms: f64,
}

/// SplitMix64 finalizer — mixes an integer into a well-distributed seed.
fn mix(mut z: u64) -> u64 {
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
    pub world: Vec<u128>,
    pub agents: Vec<Agent>,
    pub generation: u32,
    current_steps: u32,
    config: SimConfig,
    move_vectors: Vec<(i32, i32)>,
    obstacles: Vec<((u32, u32), (u32, u32))>,
    master_rng: ChaCha8Rng,
    gen_start: Instant,
    /// Optional genome to seed the initial population from (mutated copies).
    seed_genome: Option<Vec<u32>>,
    metrics_out: Option<File>,
    champion_path: Option<PathBuf>,
    champion_interval: u32,
}

impl Simulator {
    pub fn new(config: SimConfig) -> Simulator {
        let master_rng = ChaCha8Rng::seed_from_u64(config.seed);
        Simulator {
            world: vec![0; 128],
            agents: Vec::new(),
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
            // Single horizontal barrier on the survival row, derived from SURVIVAL_Y.
            obstacles: vec![((10, SURVIVAL_Y), (118, SURVIVAL_Y))],
            master_rng,
            gen_start: Instant::now(),
            seed_genome: None,
            metrics_out: None,
            champion_path: None,
            champion_interval: 0,
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

    fn random_genome(&mut self) -> Vec<u32> {
        (0..self.config.genome_length)
            .map(|_| self.master_rng.random::<u32>())
            .collect()
    }

    pub fn generate_initial_generation(&mut self) {
        let seed_genome = self.seed_genome.clone();
        for i in 0..self.config.population {
            let pos = self.rand_pos();
            let genome = match &seed_genome {
                Some(g) => {
                    let mut rng = ChaCha8Rng::seed_from_u64(agent_seed(
                        self.config.seed,
                        0,
                        REPRO_TAG,
                        i as usize,
                    ));
                    mutate_genome(g, self.config.mutation_rate, &mut rng)
                }
                None => self.random_genome(),
            };
            self.agents
                .push(Agent::new(&genome, self.config.amount_inners as u8, pos));
        }
        self.add_obstacles();
        self.gen_start = Instant::now();
    }

    fn spawn_next_generation(&mut self) {
        // Record metrics / champion for the generation that just finished.
        self.record_generation();

        self.generation += 1;
        self.remove_losers();

        // Extinction: no survivors means there is nothing to reproduce from.
        // Reseed the generation with fresh random genomes instead of crashing
        // on a modulo-by-zero.
        if self.agents.is_empty() {
            let mut new_generation: Vec<Agent> = Vec::new();
            for _ in 0..self.config.population {
                let pos = self.rand_pos();
                let genome = self.random_genome();
                new_generation.push(Agent::new(
                    &genome,
                    self.config.amount_inners as u8,
                    pos,
                ));
            }
            self.agents = new_generation;
            self.gen_start = Instant::now();
            return;
        }

        // Reserve unique spawn positions serially (mutates the world), then
        // produce the children (genome mutation + Brain decode) in parallel.
        let positions: Vec<(u32, u32)> =
            (0..self.config.population).map(|_| self.rand_pos()).collect();
        let len = self.agents.len();
        let mutation_rate = self.config.mutation_rate;
        let seed = self.config.seed;
        let generation = self.generation;
        let parents = &self.agents;

        let new_generation: Vec<Agent> = positions
            .par_iter()
            .enumerate()
            .map(|(i, &pos)| {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(agent_seed(seed, generation, REPRO_TAG, i));
                parents[i % len].produce_child(mutation_rate, pos, &mut rng)
            })
            .collect();

        self.agents = new_generation;
        self.gen_start = Instant::now();
    }

    fn rand_pos(&mut self) -> (u32, u32) {
        let mut pos: (u32, u32) = (
            self.master_rng.random_range(0..=127),
            self.master_rng.random_range(0..=127),
        );
        while self.get_pos(pos) {
            pos = (
                self.master_rng.random_range(0..=127),
                self.master_rng.random_range(0..=127),
            );
        }
        self.toggle_pos(pos);
        pos
    }

    fn toggle_pos(&mut self, coords: (u32, u32)) {
        let mask = 2_u128.pow(coords.1);
        self.world[coords.0 as usize] ^= mask;
    }

    fn get_pos(&self, coords: (u32, u32)) -> bool {
        (self.world[coords.0 as usize] >> coords.1) & 1 == 1
    }

    fn remove_losers(&mut self) {
        self.agents.retain(|a| a.get_pos().1 > SURVIVAL_Y);
        self.clear_world();
    }

    fn clear_world(&mut self) {
        self.world = vec![0; 128];
        self.add_obstacles();
    }

    /// Advance the simulation one step.
    ///
    /// Split into two phases for parallelism:
    ///   (a) decision phase — parallel over agents against a start-of-tick
    ///       snapshot of the world; each agent computes its sensor inputs and
    ///       runs `Brain::step` (with a deterministic per-agent RNG) to produce
    ///       a desired translation.
    ///   (b) application phase — serial, in agent order, resolving collisions
    ///       against the live (mutating) world.
    pub fn step(&mut self) {
        if self.current_steps >= self.config.steps_per_generation {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();
        let world_snapshot = self.world.clone();
        let move_vectors = self.move_vectors.clone();
        let seed = self.config.seed;
        let generation = self.generation;
        let step = self.current_steps;

        // (a) parallel decision phase
        let translations: Vec<(i32, i32)> = self
            .agents
            .par_iter_mut()
            .enumerate()
            .map(|(i, agent)| {
                let agent_pos = agent.get_pos();
                let used = agent.get_used_inputs();
                let all_inputs =
                    calc_positional_inputs(&world_snapshot, &move_vectors, agent_pos, &inputs, used);
                let mut rng = ChaCha8Rng::seed_from_u64(agent_seed(seed, generation, step, i));
                agent.step(all_inputs, &mut rng)
            })
            .collect();

        // (b) serial application phase
        for (i, &translation) in translations.iter().enumerate() {
            let agent_pos: (u32, u32) = self.agents[i].get_pos();
            let pos: (u32, u32) = (
                (agent_pos.0 as i32 + translation.0).clamp(0, 127) as u32,
                (agent_pos.1 as i32 + translation.1).clamp(0, 127) as u32,
            );

            if !self.get_pos(pos) {
                self.toggle_pos(agent_pos);
                self.agents[i].set_pos(pos);
                self.toggle_pos(pos);
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
    */
    fn calc_step_inputs(&mut self) -> Vec<f32> {
        let mut inputs: Vec<f32> = vec![0.0; 7];
        let mut av: (u32, u32) = (0, 0);

        for a in &self.agents {
            av = (av.0 + a.get_pos().0, av.1 + a.get_pos().1);
        }

        // max(1) guards the empty-population case without changing the result
        // (sums are then 0, so the average is 0 either way).
        let n = self.agents.len().max(1) as u32;
        av = (av.0 / n, av.1 / n);

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = self.current_steps as f32 / self.config.steps_per_generation as f32;
        inputs[4] = self.master_rng.random_range(0.0..1.0);
        inputs[5] = av.1 as f32 / 127.0;
        inputs[6] = av.0 as f32 / 127.0;

        inputs
    }

    fn add_obstacles(&mut self) {
        for obstacle in &self.obstacles {
            let mut mask = 0u128;
            (obstacle.0.1..=obstacle.1.1).for_each(|x| mask += 2_u128.pow(x));

            for x in obstacle.0.0..=obstacle.1.0 {
                self.world[x as usize] |= mask;
            }
        }
    }

    /// Compute + emit metrics for the just-finished generation, and save the
    /// champion if it's on the save interval. Uses `self.generation` (not yet
    /// incremented) and the current agents.
    fn record_generation(&mut self) {
        let wall_ms = self.gen_start.elapsed().as_secs_f64() * 1000.0;
        let n = self.agents.len();

        let survivors = self
            .agents
            .iter()
            .filter(|a| a.get_pos().1 > SURVIVAL_Y)
            .count() as u32;
        let max_final_y = self.agents.iter().map(|a| a.get_pos().1).max().unwrap_or(0);
        let sum_y: u64 = self.agents.iter().map(|a| a.get_pos().1 as u64).sum();
        let mean_final_y = if n > 0 { sum_y as f64 / n as f64 } else { 0.0 };
        let survival_rate = if n > 0 { survivors as f64 / n as f64 } else { 0.0 };

        let metrics = self.metrics_out.is_some().then(|| GenerationMetrics {
            generation: self.generation,
            survivors,
            survival_rate,
            extinction: survivors == 0,
            mean_final_y,
            max_final_y,
            genome_diversity: self.genome_diversity(),
            wall_ms,
        });
        if let (Some(file), Some(m)) = (self.metrics_out.as_mut(), metrics) {
            // Best-effort: a metrics write failure shouldn't kill a long run.
            if let Ok(line) = serde_json::to_string(&m) {
                let _ = writeln!(file, "{}", line);
            }
        }

        self.maybe_save_champion();
    }

    /// Mean pairwise Hamming distance (bits) over a random sample of agent
    /// pairs, using a dedicated deterministic RNG so it never perturbs the sim.
    fn genome_diversity(&self) -> f64 {
        let n = self.agents.len();
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
            let ga = &self.agents[a].genome;
            let gb = &self.agents[b].genome;
            total += ga
                .iter()
                .zip(gb.iter())
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
        let Some(best) = self
            .agents
            .iter()
            .max_by_key(|a| a.get_pos().1)
        else {
            return;
        };
        let champion = Champion {
            config: self.config.clone(),
            genome: best.genome.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&champion) {
            let _ = std::fs::write(path, json);
        }
    }
}

/// Read a single world cell from a snapshot (column-bitmask) world without
/// needing `&mut Simulator`, so the decision phase can run in parallel.
#[inline]
fn world_get(world: &[u128], coords: (u32, u32)) -> bool {
    (world[coords.0 as usize] >> coords.1) & 1 == 1
}

/// Build an agent's full input vector: the shared base inputs plus the 8
/// directional "can I move here" sensors, read against a world snapshot.
fn calc_positional_inputs(
    world: &[u128],
    move_vectors: &[(i32, i32)],
    pos: (u32, u32),
    base: &[f32],
    used: Vec<usize>,
) -> Vec<f32> {
    let mut copy = base.to_vec();
    copy.extend_from_slice(&[0.0; 8]);

    for i in used {
        let vec: (i32, i32) = move_vectors[i - 7];
        let neighbor = (
            (pos.0 as i32 + vec.0).clamp(0, 127) as u32,
            (pos.1 as i32 + vec.1).clamp(0, 127) as u32,
        );
        copy[i] = if world_get(world, neighbor) { 0.0 } else { 1.0 };
    }

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
        }
    }

    #[test]
    fn collision_blocks_movement_onto_occupied_cells() {
        // Every spawned position is marked in the world; an agent can never be
        // placed on top of another, and obstacle cells stay occupied.
        let mut sim = Simulator::new(test_config(1));
        sim.generate_initial_generation();

        // No two agents share a cell.
        let mut seen = std::collections::HashSet::new();
        for a in &sim.agents {
            assert!(seen.insert(a.get_pos()), "duplicate agent position");
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
            for a in &sim.agents {
                assert!(seen.insert(a.get_pos()), "two agents on one cell after step");
            }
        }
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
        let pa: Vec<_> = a.agents.iter().map(|x| x.get_pos()).collect();
        let pb: Vec<_> = b.agents.iter().map(|x| x.get_pos()).collect();
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
        let pa: Vec<_> = a.agents.iter().map(|x| x.get_pos()).collect();
        let pb: Vec<_> = b.agents.iter().map(|x| x.get_pos()).collect();
        assert_ne!(pa, pb, "different seeds produced identical state");
    }
}
