use std::collections::HashSet;
use rand::{random, Rng};
use crate::agent::Agent;

mod agent;

fn main() {
    let genome_length: u32 = 32;
    let amount_inners: u32 = 20;
    let mutation_rate: f32 = 0.001;
    let steps_per_generation: u32 = 200;
    let population: u32 = 1;
    let world_size: (u16, u16) = (128, 128);

    let mut simulator = Simulator::new(
        genome_length,
        amount_inners,
        mutation_rate,
        steps_per_generation,
        population,
        world_size
    );

    //agent::test();
    run_sim(&mut simulator);
}

fn run_sim(simulator: &mut Simulator) {
    simulator.generate_initial_generation();

    for i in 0..5 {
        simulator.step();
    }

    /*loop {
        simulator.step();
    }*/
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
    world_size: (u16, u16),
    move_vectors: Vec<(i8, i8)>
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, world_size: (u16, u16)) -> Simulator {
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
        let inputs = self.calc_step_inputs();

        for i in 0..self.world_size.0 {
            for j in 0..self.world_size.1 {
                let id = self.world[i as usize][j as usize];
                if id != 0 {
                    let mut a = self.get_agent(id);
                    a.step(self.calc_positional_inputs((i as i32, j as i32), inputs.clone()));
                    self.add_agent(a);
                }
            }
        }

        self.current_steps += 1;
    }

    fn add_agent(&mut self, agent: Agent) {
        self.agents.insert(agent);
    }

    fn get_agent(&mut self, id: u64) -> Agent {
        self.agents.take(&id).unwrap()
    }

    fn calc_positional_inputs(&mut self, pos: (i32, i32), base: Vec<f32>) -> Vec<f32> {
        let mut copy = base.clone();

        for i in 0..self.move_vectors.len() {
            let vec: (i8, i8) = self.move_vectors[i];
            let mut out: f32 = 0.0;
            if self.pos_free((pos.0 + vec.0 as i32, pos.1 + vec.1 as i32)) {
                out = 1.0;
            }

           copy.push(out);
        }

        copy
    }

    /* INPUTS
    0: always 0
    1: always 1
    2: oscillator
    3: age
    4: random
    5-12: can move in x direction
    */
    fn calc_step_inputs(&mut self) -> Vec<f32> {
        let mut inputs: Vec<f32> = vec![0.0; 5];

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = (self.current_steps as f32/self.steps_per_generation as f32);
        inputs[4] = rand::thread_rng().gen_range(0.0..1.0);

        inputs
    }

    fn pos_free(&mut self, pos: (i32, i32)) -> bool {
        if pos.0 >= self.world_size.0 as i32 || pos.0 < 0 {
            return false;
        }
        if pos.1 >= self.world_size.1 as i32 || pos.1 < 0 {
            return false;
        }
        if self.world[pos.0 as usize][pos.1 as usize] != 0 {
            return false;
        }
        return true;
    }

    fn generate_initial_generation(&mut self) {
        let mut taken_pos: HashSet<(u32, u32)> = HashSet::new();
        let mut rng = rand::thread_rng();

        for i in 0..self.population {
            let mut  pos: (u32, u32) = (rng.gen_range(0..self.world_size.0) as u32, rng.gen_range(0..self.world_size.1) as u32);
            while !taken_pos.insert(pos) {
                pos = (rng.gen_range(0..self.world_size.0) as u32, rng.gen_range(0..self.world_size.1) as u32);
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