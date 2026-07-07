use std::time::Instant;

use clap::Parser;
use neural_evolution::{Cli, Mode, build_eco_sim, build_simulator};

/// Headless runner: no window. In `challenge` mode runs the generational sim to
/// the requested generation limit (or indefinitely); in `eco` mode advances the
/// continuous terrarium for `--ticks` and prints a coverage-over-time summary.
/// `--bench` runs a fixed 50-generation challenge benchmark and prints gens/sec.
fn main() {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Challenge => run_challenge(&cli),
        Mode::Eco => run_eco(&cli),
    }
}

fn run_challenge(cli: &Cli) {
    let mut sim = build_simulator(cli);
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

fn run_eco(cli: &Cli) {
    let mut sim = build_eco_sim(cli);
    sim.seed_initial();

    let ticks = cli.ticks.unwrap_or(u32::MAX as u64);
    let start = Instant::now();
    while sim.tick_count() < ticks {
        sim.tick();
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Coverage / biomass trajectory: print the meadow at a spread of ticks so
    // the self-regulation curve (climb → plateau) is visible in one glance.
    println!(
        "eco: {} ticks in {:.3}s = {:.0} ticks/sec ({}x{} grid)",
        sim.tick_count(),
        elapsed,
        sim.tick_count() as f64 / elapsed.max(1e-9),
        sim.width(),
        sim.height(),
    );
    println!("  tick   coverage   biomass    mean_nutrient");
    let hist = sim.metrics_history();
    let last = hist.len().saturating_sub(1);
    let marks = [0usize, 25, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000];
    for &t in marks.iter().filter(|&&t| t <= last) {
        let m = &hist[t];
        println!(
            "  {:>5}   {:>7.2}%   {:>7.1}    {:>7.3}",
            m.tick,
            m.coverage * 100.0,
            m.total_biomass,
            m.mean_nutrient,
        );
    }
    if let Some(m) = hist.get(last) {
        println!(
            "  final @ {}: coverage {:.2}%  biomass {:.1}  mean_nutrient {:.3}",
            m.tick,
            m.coverage * 100.0,
            m.total_biomass,
            m.mean_nutrient,
        );
    }
}
