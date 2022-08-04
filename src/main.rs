use std::collections::HashSet;
use rand::{random, Rng};
use crate::agent::Agent;

mod agent;

fn main() {
    let genome_length: u32 = 0;
    let amount_inners: u32 = 0;
    let mutation_rate: f32 = 0.0;
    let steps_per_generation: u32 = 0;
    let population: u32 = 0;
    let world_size: (u32, u32) = (0, 0);

    let mut simulator = Simulator::new(
        genome_length,
        amount_inners,
        mutation_rate,
        steps_per_generation,
        population,
        world_size
    );

    simulator.generate_initial_generation();

    loop {
        simulator.step();
    }
}

struct Simulator {
    world: Vec<Vec<u64>>,
    agents: HashSet<Agent>,
    generation: u32,
    total_age: u32,
    current_steps: u32,
    genome_length: u32,
    amount_inners: u32,
    mutation_rate: f32,
    steps_per_generation: u32,
    population: u32,
    world_size: (u32, u32),
    move_vectors: Vec<(u32, u32)>
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, world_size: (u32, u32)) -> Simulator {
        Simulator{
            world: vec![vec![0; world_size.1 as usize]; world_size.0 as usize],
            agents: HashSet::new(),
            generation: 0,
            total_age: 0,
            current_steps: 0,
            genome_length,
            amount_inners,
            mutation_rate,
            steps_per_generation,
            population,
            world_size,
            move_vectors: vec![
            (0, 1),
            (0, -1),
            (1, 0),
            (1, 1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1)
            ]
        }
    }

    fn step(&mut self) {
        todo!("balls")
    }

    /* INPUTS
    0: always 0
    1: always 1
    2: oscillator
    3: age
    4: random
    5-12: can move in x direction
    */
    fn calc_inputs(&mut self, pos: (u32, u32)) {
        let mut inputs: Vec<f32> = vec![0.0; 13];

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = (self.current_steps/self.steps_per_generation) as f32;
        inputs[4] = rand::thread_rng().gen_range(0.0..1.0);

        for i in 0..self.move_vectors.len() {

        }
    }

    fn pos_free(&mut self, pos: (u32, u32)) -> bool {
        if self.world[pos.0][pos.1] == 0 {
            true
        }
        false
    }

    fn generate_initial_generation(&mut self) {
        let mut taken_pos: HashSet<(u32, u32)> = HashSet::new();
        let mut rng = rand::thread_rng();

        for i in 0..self.population {
            let mut  pos: (u32, u32) = (rng.gen_range(0..self.world_size.0), rng.gen_range(0..self.world_size.1));
            while !taken_pos.insert(pos) {
                pos = (rng.gen_range(0..self.world_size.0), rng.gen_range(0..self.world_size.1));
            }
            let mut genome: Vec<u32> = Vec::new();

            for i in 0..self.genome_length {
                genome.push(rng.gen::<u32>());
            }

            let mut agent: Agent = Agent::new(
                &genome,
                self.amount_inners as u8
            );

            self.world[pos.0 as usize][pos.1 as usize] = agent.id;
            self.agents.insert(agent);
        }
    }
}