use rand::{Rng};
use crate::agent::Agent;

mod agent;

fn main() {
    let genome_length: u32 = 32;
    let amount_inners: u32 = 20;
    let mutation_rate: f32 = 0.001;
    let steps_per_generation: u32 = 200;
    let population: u32 = 1000;

    let mut simulator = Simulator::new(
        genome_length,
        amount_inners,
        mutation_rate,
        steps_per_generation,
        population
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
    world: Vec<u128>,
    agents: Vec<Agent>,
    generation: u32,
    current_steps: u32,
    genome_length: u32,
    amount_inners: u32,
    mutation_rate: f32,
    steps_per_generation: u32,
    population: u32,
    move_vectors: Vec<(i32, i32)>
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32) -> Simulator {
        Simulator {
            world: vec![0; 128],
            agents: Vec::new(),
            generation: 0,
            current_steps: 0,
            genome_length,
            amount_inners,
            mutation_rate,
            steps_per_generation,
            population,
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

    fn spawn_next_generation(&mut self) {
        self.generation += 1;
        self.remove_losers();
        let mut new_generation: Vec<Agent> = Vec::new();

        for i in 0..self.population {
            let mut a: Agent = self.agents[i as usize % self.agents.len()].clone();

            new_generation.push(a.produce_child(
                self.mutation_rate,
                self.rand_pos()
            ));
        }

        self.agents = new_generation;

        println!("on generation {}", self.generation);
    }

    fn rand_pos(&mut self) -> (u32, u32) {
        let mut rand = rand::thread_rng();
        let mut pos: (u32, u32) = (rand.gen_range(0..127), rand.gen_range(0..127));
        while self.get_pos(pos) {
            pos = (rand.gen_range(0..127), rand.gen_range(0..127));
        }
        self.toggle_pos(pos);

        pos
    }

    fn toggle_pos(&mut self, coords: (u32, u32)) {
        let mask = (2 as u128).pow(coords.1);
        self.world[coords.0 as usize] ^= mask;
    }

    fn get_pos(&mut self, coords: (u32, u32)) -> bool {
        let mask = (2 as u128).pow(coords.1);
        (self.world[coords.0 as usize] & mask)/mask == 1
    }

    fn remove_losers(&mut self) {
        let mut winners: Vec<Agent> = Vec::new();

        for agent in &mut *self.agents {
            let pos = agent.get_pos();
            if pos.1 > 63 {
                winners.push(agent.clone());
            }
        }

        self.agents = winners;
        self.clear_world();
    }

    fn clear_world(&mut self) {
        self.world = vec![0; 128];
    }

    fn step(&mut self) {
        if self.current_steps >= 200 {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();

        for i in 0..self.agents.len() {
            let agent: Agent = self.agents[i].clone();
            let all_inputs = self.calc_positional_inputs(agent.get_pos(), &inputs);
            let translation: (i32, i32) = self.agents[i].step(all_inputs);
            let pos: (u32, u32) = ((agent.pos.0 as i32 + translation.0).clamp(0, 127) as u32, (agent.pos.1 as i32 + translation.1).clamp(0, 127) as u32);

            if !self.get_pos(pos) {
                self.toggle_pos(agent.get_pos());
                self.agents[i].set_pos(pos);
                self.toggle_pos(pos);
            }
        }

        self.current_steps += 1;
    }

    /*fn add_state(&mut self, s: (u64, (u32, u32))) {
        let l = self.state_len();
        if l > 0 {
            self.states.states[l - 1].push(s);
        }
        else {
            self.states.states.push(vec![s])
        }
    }*/

    /*fn state_len(&mut self) -> usize {
        self.states.states.len()
    }*/

    fn calc_positional_inputs(&mut self, pos: (u32, u32), base: &Vec<f32>) -> Vec<f32> {
        let mut copy = base.clone();

        for i in 0..self.move_vectors.len() {
            let vec: (i32, i32) = self.move_vectors[i];
            let mut out: f32 = 1.0;
            if self.get_pos(((pos.0 as i32 + vec.0).clamp(0, 127) as u32, (pos.1 as i32 + vec.1).clamp(0, 127) as u32)) {
                out = 0.0;
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

    fn random_genome(&self) -> Vec<u32> {
        let mut genome: Vec<u32> = Vec::new();
        let mut rand = rand::thread_rng();

        (0..self.genome_length).for_each(|_| {genome.push(rand.gen::<u32>())});

        genome
    }

    fn generate_initial_generation(&mut self) {
        for _ in 0..self.population {
            let pos: (u32, u32) = self.rand_pos();

            self.agents.push(Agent::new(
                &self.random_genome(),
                self.amount_inners as u8,
                pos
            ));
        }
    }
}