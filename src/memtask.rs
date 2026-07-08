//! A minimal **delayed-recall benchmark** — the clean, isolated proof that the
//! CTRNN substrate buys memory that a feed-forward brain provably cannot have.
//!
//! ## Why this exists
//!
//! The eco terrarium was built on the belief that foraging/hunting with local
//! senses *demands* memory. Measured honestly (see DESIGN.md), it does not:
//! reactive gradient-following + straight-line search suffice, so the memoryless
//! feed-forward brain matches or beats the CTRNN there (whose integration lag and
//! untamed-from-random recurrence are a net cost), and the head-to-head
//! mixed-competition share goes *to feed-forward*. That is the ctrnn-sim lesson
//! recurring: substrate without a task that rewards it doesn't help.
//!
//! This benchmark is the inversion — a task where memory is **provably
//! necessary** — so the substrate's payoff can actually be measured:
//!
//! - A single cue bit `c ∈ {0,1}` is shown on input 1 (as ±1) only for the first
//!   `CUE_STEPS` ticks of an episode, then removed (input 1 = 0).
//! - After the cue is gone a "report" flag (input 2) turns on. The brain must push
//!   its two decision outputs (north vs south) to report `c`.
//!
//! At report time the feed-forward brain's inputs are `[bias, 0, report, 0]` —
//! **identical whether `c` was 0 or 1** — so its output is identical too: it is
//! pinned at chance (fitness ≈ 0.5), no matter how it is wired. A CTRNN can latch
//! `c` into persistent state during the cue window (an inner→inner loop) and read
//! it back after — so evolution can climb it to near-perfect recall. Same sparse
//! connection genome, same `mutate_genome`, same everything but the neuron
//! dynamics; the only variable is memory.
//!
//! The benchmark runs a small mu+lambda evolutionary loop for each brain kind and
//! returns the two mean-fitness curves, so the gap is a measured A/B, not an
//! assertion. It is deterministic (seeded), so two runs match.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::agent::{Brain, BrainKind, mutate_genome};

/// Inputs fed to the recall brain: `[bias, cue, report, pad]` (width 4).
const MEM_INPUTS: usize = 4;
/// Inner neurons available to build a latch + readout. Small — a latch needs only
/// one recurrent loop — but enough headroom for evolution to find one.
const MEM_INNERS: u8 = 6;
/// Ticks per episode.
const EPISODE_STEPS: u32 = 12;
/// How many opening ticks the cue is visible before it is removed (the delay the
/// brain must bridge is `EPISODE_STEPS - CUE_STEPS`).
const CUE_STEPS: u32 = 4;

/// Genes (connections) per recall genome. Enough to encode input→inner (latch the
/// cue), inner→inner (hold it), and inner→output (read it out).
const MEM_GENOME_LEN: usize = 24;

/// The two mean-fitness-over-generations curves from a paired recall run. Fitness
/// is in 0..1, where 0.5 is chance (guessing) and 1.0 is perfect recall.
pub struct MemResult {
    pub ff_curve: Vec<f32>,
    pub ctrnn_curve: Vec<f32>,
}

/// Evaluate one genome on the recall task: the mean, over both cue values and the
/// post-cue report window, of the decision margin *aligned* with the true cue,
/// mapped to 0..1 (0.5 = chance). Deterministic — the only RNG is the movement
/// decoder's jitter draw, which never touches the decision outputs read here.
fn recall_fitness(genome: &[u32], kind: BrainKind) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut total = 0.0f32;
    for c in [0u32, 1u32] {
        // Fresh brain per episode so a CTRNN starts each episode with no memory
        // (the cue must be learned *within* the episode, not leaked across).
        let mut brain = Brain::from(genome.to_vec(), MEM_INPUTS, MEM_INNERS, kind);
        let cue = if c == 1 { 1.0 } else { -1.0 };
        let mut episode = 0.0f32;
        let mut reports = 0.0f32;
        for step in 0..EPISODE_STEPS {
            let in_cue = step < CUE_STEPS;
            let inputs = vec![
                1.0,
                if in_cue { cue } else { 0.0 },
                if in_cue { 0.0 } else { 1.0 },
                0.0,
            ];
            brain.step(inputs, &mut rng);
            if !in_cue {
                // Decision margin: north output minus south output. Aligned with
                // the cue (+ for c=1, − for c=0); a correct, confident report → +1.
                let out = brain.neurons();
                let margin = out[2][1] - out[2][2];
                episode += if c == 1 { margin } else { -margin };
                reports += 1.0;
            }
        }
        total += (episode / reports.max(1.0)).clamp(-1.0, 1.0);
    }
    // Mean aligned margin over the two cues, mapped from −1..1 to 0..1.
    ((total / 2.0) + 1.0) / 2.0
}

/// Run the paired recall benchmark: a small mu+lambda loop (keep the top half,
/// refill with mutated children) for each brain kind, returning the two
/// mean-fitness curves over `generations`. `population` genomes per generation.
pub fn run_recall_benchmark(seed: u64, generations: u32, population: usize) -> MemResult {
    MemResult {
        ff_curve: evolve(seed, generations, population, BrainKind::Feedforward),
        ctrnn_curve: evolve(seed, generations, population, BrainKind::Ctrnn),
    }
}

/// One evolutionary run for a single brain kind; returns the per-generation mean
/// fitness of the surviving (top-half) parents — the learning curve.
fn evolve(seed: u64, generations: u32, population: usize, kind: BrainKind) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Founders: random genomes (identical draw order for both kinds at a given
    // seed, so the only difference between the two curves is the brain dynamics).
    let mut genomes: Vec<Vec<u32>> = (0..population)
        .map(|_| (0..MEM_GENOME_LEN).map(|_| rng.random::<u32>()).collect())
        .collect();

    let mut curve = Vec::with_capacity(generations as usize);
    let keep = (population / 2).max(1);
    for generation in 0..generations {
        // Score, then sort descending by fitness (ties by genome, deterministic).
        let mut scored: Vec<(f32, Vec<u32>)> = genomes
            .into_iter()
            .map(|g| (recall_fitness(&g, kind), g))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));

        let parents_mean = scored[..keep].iter().map(|(f, _)| *f).sum::<f32>() / keep as f32;
        curve.push(parents_mean);

        // Reproduce: keep the top half, refill with mutated children of parents.
        let parents: Vec<Vec<u32>> = scored.into_iter().take(keep).map(|(_, g)| g).collect();
        let mut next = parents.clone();
        let mut child_rng = ChaCha8Rng::seed_from_u64(crate::mix(seed ^ generation as u64));
        let mut i = 0usize;
        while next.len() < population {
            let child = mutate_genome(&parents[i % keep], 0.03, &mut child_rng);
            next.push(child);
            i += 1;
        }
        genomes = next;
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedforward_is_stuck_at_chance_on_recall() {
        // A feed-forward brain's inputs at report time are identical for both cue
        // values, so no wiring can distinguish them: fitness cannot beat ~chance
        // (0.5), however long it evolves. This is what makes the task a clean
        // memory test.
        let curve = evolve(7, 40, 80, BrainKind::Feedforward);
        let best = curve.iter().cloned().fold(0.0f32, f32::max);
        assert!(best < 0.60, "feed-forward should stay near chance on recall, got {best}");
    }

    #[test]
    fn ctrnn_learns_to_recall_the_cue() {
        // The CTRNN can latch the cue into persistent state and read it back after
        // the cue is gone, so evolution climbs it well above chance — the payoff
        // of recurrence on a task that actually needs memory.
        let curve = evolve(7, 60, 120, BrainKind::Ctrnn);
        let best = curve.iter().cloned().fold(0.0f32, f32::max);
        assert!(best > 0.75, "CTRNN should learn to recall well above chance, got {best}");
    }

    #[test]
    fn recall_benchmark_is_deterministic() {
        let a = run_recall_benchmark(42, 20, 60);
        let b = run_recall_benchmark(42, 20, 60);
        assert_eq!(a.ff_curve, b.ff_curve, "FF recall curve not deterministic");
        assert_eq!(a.ctrnn_curve, b.ctrnn_curve, "CTRNN recall curve not deterministic");
    }
}
