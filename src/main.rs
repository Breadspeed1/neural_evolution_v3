use std::collections::HashSet;
use crate::agent::Agent;

mod agent;

fn main() {
    agent::test();
}

struct Simulator {
    world: Vec<Vec<u32>>,
    agents: HashSet<Agent>,
    generation: u32,
    total_age: u32,
    current_steps: u32,
    genome_length: u32,
    amount_inners: u32,
    mutation_rate: f32,
    steps_per_generation: u32,
    population: u32,
    world_size: (u32, u32)
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, world_size: (u32, u32)) -> Simulator {
        Simulator{
            world: Vec::new(),
            agents: HashSet::new(),
            generation: 0,
            total_age: 0,
            current_steps: 0,
            genome_length,
            amount_inners,
            mutation_rate,
            steps_per_generation,
            population,
            world_size
        }
    }

    fn step(&mut self) {
        todo!("balls")
    }

    fn generate_initial_generation(&mut self) {
        todo!("a")
    }
}