use std::time::Instant;

use clap::Parser;
use neural_evolution::{Cli, build_simulator};

/// Headless runner: no window. Runs the sim to the requested generation limit
/// (or indefinitely), writing metrics/champion files as configured. `--bench`
/// runs a fixed 50-generation benchmark and prints gens/sec.
fn main() {
    let cli = Cli::parse();
    let mut sim = build_simulator(&cli);
    sim.generate_initial_generation();

    if cli.bench {
        let target = sim.generation + 50;
        let start = Instant::now();
        while sim.generation < target {
            sim.step();
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "BENCH: 50 gens in {:.3}s = {:.3} gens/sec",
            elapsed,
            50.0 / elapsed
        );
        return;
    }

    let generations = cli.generations.unwrap_or(u32::MAX);
    let mut last_gen = sim.generation;
    while sim.generation < generations {
        sim.step();
        if sim.generation != last_gen {
            last_gen = sim.generation;
            if sim.generation.is_multiple_of(10) {
                println!("generation {}", sim.generation);
            }
        }
    }
}
