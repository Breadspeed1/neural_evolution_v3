use std::collections::HashSet;
use std::fs;
use rand::{Rng};
use serde::{Serialize};
use crate::agent::Agent;

mod agent;

fn main() {
    let genome_length: u32 = 32;
    let amount_inners: u32 = 20;
    let mutation_rate: f32 = 0.001;
    let steps_per_generation: u32 = 200;
    let population: u32 = 200;
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

    loop {
        simulator.step();
    }
}

struct Simulator {
    world: Vec<Vec<u64>>,
    agents: HashSet<Agent>,
    generation: u32,
    current_steps: u32,
    genome_length: u32,
    amount_inners: u32,
    mutation_rate: f32,
    steps_per_generation: u32,
    population: u32,
    world_size: (u16, u16),
    move_vectors: Vec<(i8, i8)>,
    states: Generation
}

#[derive(Serialize)]
struct Generation {
    states: Vec<Vec<(u64, (u32, u32))>>,
    agents: Vec<(u64, Vec<u32>)>,
}

impl Generation {
    fn new() -> Generation {
        Generation {
            states: Vec::new(),
            agents: Vec::new(),
        }
    }
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, world_size: (u16, u16)) -> Simulator {
        Simulator {
            world: vec![vec![0; world_size.1 as usize]; world_size.0 as usize],
            agents: HashSet::new(),
            generation: 0,
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
            ],
            states: Generation::new()
        }
    }

    fn spawn_next_generation(&mut self) {
        self.generation += 1;
        self.remove_losers();
        let mut new_generation: HashSet<Agent> = HashSet::new();
        let mut ids: Vec<u64> = Vec::new();
        let mut rand = rand::thread_rng();

        let mut i: u64 = 0;
        self.agents.iter().for_each(|x| {ids.push(x.id)});
        while new_generation.len() != self.population as usize {
            let mut a: Agent = self.agents.take(&ids[rand.gen_range(0..ids.len())]).unwrap();
            new_generation.insert(a.produce_child(self.mutation_rate, i));
            self.add_agent(a);
            i += 1;
        }

        self.agents = new_generation;
        self.populate_world();

        self.generation_to_json();
        self.reset_generation();

        println!("on generation {}", self.generation);
    }

    fn reset_generation(&mut self) {
        let mut new_agents: Vec<(u64, Vec<u32>)> = Vec::new();
        self.agents.iter().for_each(|x| new_agents.push((x.id, x.genome.clone())));

        self.states = Generation {
            states: Vec::new(),
            agents: new_agents
        };
    }

    fn generation_to_json(&mut self) {
        let j = serde_json::to_string(&self.states);
        fs::write(format!("output\\{}.json", self.generation), j.unwrap().to_string()).expect("error writing file");
    }

    fn populate_world(&mut self) {
        let mut taken_pos: HashSet<(u32, u32)> = HashSet::new();
        let mut rng = rand::thread_rng();

        for a in self.agents.iter() {
            let mut  pos: (u32, u32) = (rng.gen_range(0..self.world_size.0) as u32, rng.gen_range(0..self.world_size.1) as u32);
            while !taken_pos.insert(pos) {
                pos = (rng.gen_range(0..self.world_size.0) as u32, rng.gen_range(0..self.world_size.1) as u32);
            }

            self.world[pos.0 as usize][pos.1 as usize] = a.id;
        }
    }

    fn remove_losers(&mut self) {
        for i in 0..self.world_size.0 {
            for j in 0..self.world_size.1 {
                let id = self.world[i as usize][j as usize];
                if id != 0 {
                    if j < 64 {
                        self.remove_agent(id);
                    }
                }
            }
        }

        self.clear_world();
    }

    fn clear_world(&mut self) {
        self.world = vec![vec![0; self.world_size.1 as usize]; self.world_size.0 as usize];
    }

    fn remove_agent(&mut self, id: u64) {
        self.agents.remove(&id);
    }

    fn step(&mut self) {
        if self.current_steps >= 200 {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();
        let mut requested_moves: Vec<(u64, (i32, i32))> = Vec::new();

        for i in 0..self.world_size.0 {
            for j in 0..self.world_size.1 {
                let id = self.world[i as usize][j as usize];
                if id != 0 {
                    let mut a = self.get_agent(id);
                    let translation: (i32, i32) = a.step(self.calc_positional_inputs((i as i32, j as i32), inputs.clone()));
                    let pos: (i32, i32) = (translation.0 + i as i32, translation.1 + j as i32);

                    self.add_state((id, (i as u32, j as u32)));

                    if self.pos_free(pos) && !requested_moves.contains(&(id, pos)) {
                        requested_moves.push((
                            id,
                            pos
                        ));
                    }
                    else {
                        requested_moves.push((
                            id,
                            (i as i32, j as i32)
                        ));
                    }

                    self.add_agent(a);

                    self.world[i as usize][j as usize] = 0;
                }
            }
        }

        self.states.states.push(Vec::new());

        for pos in requested_moves {
            self.world[pos.1.0 as usize][pos.1.1 as usize] = pos.0;
        }

        self.current_steps += 1;
    }

    fn add_state(&mut self, s: (u64, (u32, u32))) {
        let l = self.state_len();
        if l > 0 {
            self.states.states[l - 1].push(s);
        }
        else {
            self.states.states.push(vec![s])
        }
    }

    fn state_len(&mut self) -> usize {
        self.states.states.len()
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

            let agent: Agent = Agent::new(
                &genome,
                self.amount_inners as u8,
                i as u64
            );

            self.world[pos.0 as usize][pos.1 as usize] = agent.id;
            self.agents.insert(agent);
        }

        self.reset_generation();
    }
}