use std::fs::File;
use std::{mem, thread};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use image::{Frame, ImageBuffer, RgbaImage};
use image::codecs::gif::{GifEncoder};
use rand::{Rng};
use crate::agent::Agent;

mod agent;

fn main() {
    let genome_length: u32 = 32;
    let amount_inners: u32 = 20;
    let mutation_rate: f32 = 0.001;
    let steps_per_generation: u32 = 300;
    let population: u32 = 1000;
    let generate_gifs: bool = false;
    let obstacles: Vec<((u32, u32), (u32, u32))> = vec![
        ((20, 64), (108, 64))
    ];

    let mut simulator = Simulator::new(
        genome_length,
        amount_inners,
        mutation_rate,
        steps_per_generation,
        population,
        obstacles,
        generate_gifs
    );

    run_sim(&mut simulator);
}

fn run_sim(simulator: &mut Simulator) {
    simulator.generate_initial_generation();
    let generations = 60;

    let now = Instant::now();
    while simulator.generation < generations {

        step(simulator);

        if simulator.current_steps == 0 {
            println!("on generation {}", simulator.generation);
        }
    }
    println!("{} gens took {} minutes", generations, now.elapsed().as_secs_f32()/60.0);
}

fn step(simulator: &mut Simulator) {
    if simulator.step_or_generation() {
        let inputs: Vec<f32> = simulator.calc_step_inputs();
        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        for i in 0 as usize..=10 {
            handles.push(thread::spawn( move || {
                simulator.handle_agents((i * 100)..((i + 1) * 100), &inputs);
            }));
        }
    }
    else {
        simulator.spawn_next_generation();
    }
}

struct GenerationOutput {
    frames: Vec<Frame>
}

impl GenerationOutput {
    fn new() -> GenerationOutput {
        GenerationOutput {
            frames: Vec::new()
        }
    }

    fn add_step(&mut self, state: Vec<u128>) {
        let mut im: RgbaImage = ImageBuffer::new(128, 128);
        im.fill(u8::MAX);

        for i in 0..state.len() {
            for j in 0..128 as u32 {
                let mask = (2 as u128).pow(j);
                if (state[i] & mask)/mask == 1 {
                    im.get_pixel_mut(i as u32, j).0 = [255, 0, 0, 255];
                }
            }
        }

        self.frames.push(Frame::new(im));
    }

    fn save(&mut self, path: String) {
        let file_out = File::create(path).unwrap();
        let mut encoder = GifEncoder::new(file_out);

        for i in 0..self.frames.len() {
            encoder.encode_frame(self.frames[i].clone()).unwrap();
        }
    }
}

struct Simulator {
    world: Arc<Mutex<Vec<u128>>>,
    agents: Arc<Mutex<Vec<Agent>>>,
    generation: u32,
    current_steps: u32,
    genome_length: u32,
    amount_inners: u32,
    mutation_rate: f32,
    steps_per_generation: u32,
    population: u32,
    move_vectors: Vec<(i32, i32)>,
    output: GenerationOutput,
    obstacles: Vec<((u32, u32), (u32, u32))>,
    use_output: bool
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, obstacles: Vec<((u32, u32), (u32, u32))>, use_output: bool) -> Simulator {
        Simulator {
            world: Arc::new(Mutex::new(vec![0; 128])),
            agents: Arc::new(Mutex::new(Vec::new())),
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
            ],
            output: GenerationOutput::new(),
            obstacles,
            use_output
        }
    }

    fn step_or_generation(&mut self) -> bool {
        self.current_steps < self.steps_per_generation
    }

    fn handle_agents(&mut self, rows: Range<usize>, inputs: &Vec<f32>) {
        for i in rows {
            let agent: &Agent = &self.agents.lock().unwrap()[i];
            let agent_pos: (u32, u32) = agent.get_pos();
            let used: Vec<usize> = agent.get_used_inputs();
            let all_inputs = self.calc_positional_inputs(agent_pos, &inputs, used);
            let translation: (i32, i32) = self.agents.lock().unwrap()[i].step(all_inputs);
            let pos: (u32, u32) = ((agent_pos.0 as i32 + translation.0).clamp(0, 127) as u32, (agent_pos.1 as i32 + translation.1).clamp(0, 127) as u32);

            if !self.get_pos(pos) {
                self.toggle_pos(agent_pos);
                self.agents.lock().unwrap()[i].set_pos(pos);
                self.toggle_pos(pos);
            }
        }
    }

    fn end_step(&mut self) {
        if self.use_output {
            self.update_output();
        }
        self.current_steps += 1;
    }

    fn spawn_next_generation(&mut self) {
        if self.use_output {
            self.reset_output()
        }

        self.generation += 1;
        self.remove_losers();
        let mut new_generation: Vec<Agent> = Vec::new();

        for i in 0..self.population {
            let mut a: Agent = self.agents.lock().unwrap()[i as usize % self.agents.lock().unwrap().len()].clone();
            new_generation.push(a.produce_child(
                self.mutation_rate,
                self.rand_pos()
            ));
        }

        let mut state = self.agents.lock().unwrap();
        mem::replace(&mut *state, Vec::new());
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

    fn update_output(&mut self) {
        self.output.add_step(self.world.lock().unwrap().clone());
    }

    fn reset_output(&mut self) {
        self.output.save(format!("G:\\\\Visualizer_Output\\generation-{}.gif", self.generation));
        self.output = GenerationOutput::new();
    }

    fn toggle_pos(&mut self, coords: (u32, u32)) {
        let mask = (2 as u128).pow(coords.1);
        self.world.lock().unwrap()[coords.0 as usize] ^= mask;
    }

    fn get_pos(&mut self, coords: (u32, u32)) -> bool {
        let mask = (2 as u128).pow(coords.1);
        (self.world.lock().unwrap()[coords.0 as usize] & mask)/mask == 1
    }

    fn remove_losers(&mut self) {
        let mut winners: Vec<Agent> = Vec::new();

        for agent in &mut *self.agents.lock().unwrap() {
            let pos = agent.get_pos();
            if pos.1 > 63 {
                winners.push(agent.clone());
            }
        }

        let mut state = self.agents.lock().unwrap();
        mem::replace(&mut *state, winners);

        self.clear_world();
    }

    fn clear_world(&mut self) {
        let mut state = self.world.lock().unwrap();
        mem::replace(&mut *state, vec![0; 128]);

        self.add_obstacles();
    }

    /*fn step(&mut self) {
        if self.current_steps >= self.steps_per_generation {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();

        for i in 0..self.agents.len() {
            let agent_pos: (u32, u32) = self.agents[i].get_pos();
            let used: Vec<usize> = self.agents[i].get_used_inputs();
            let all_inputs = self.calc_positional_inputs(agent_pos, &inputs, used);
            let translation: (i32, i32) = self.agents[i].step(all_inputs);
            let pos: (u32, u32) = ((agent_pos.0 as i32 + translation.0).clamp(0, 127) as u32, (agent_pos.1 as i32 + translation.1).clamp(0, 127) as u32);

            if !self.get_pos(pos) {
                self.toggle_pos(agent_pos);
                self.agents[i].set_pos(pos);
                self.toggle_pos(pos);
            }
        }

        if self.use_output {
            self.update_output();
        }
        self.current_steps += 1;
    }*/


    fn calc_positional_inputs(&mut self, pos: (u32, u32), base: &Vec<f32>, used: Vec<usize>) -> Vec<f32> {
        let mut copy = base.clone();
        copy.append(&mut vec![0.0; 8]);

        for i in used {
            let vec: (i32, i32) = self.move_vectors[i - 7];
            let mut out: f32 = 1.0;
            if self.get_pos(((pos.0 as i32 + vec.0).clamp(0, 127) as u32, (pos.1 as i32 + vec.1).clamp(0, 127) as u32)) {
                out = 0.0;
            }

           copy[i] = out;
        }

        copy
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

        for a in &mut *self.agents.lock().unwrap() {
            av = (av.0 + a.get_pos().0, av.0 + a.get_pos().0);
        }

        let len = self.agents.lock().unwrap().len();
        av = (av.0 / len as u32, av.1 / len as u32);

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = self.current_steps as f32/self.steps_per_generation as f32;
        inputs[4] = rand::thread_rng().gen_range(0.0..1.0);
        inputs[5] = av.1 as f32 / 127.0;
        inputs[6] = av.0 as f32 / 127.0;

        inputs
    }

    fn random_genome(&self) -> Vec<u32> {
        let mut genome: Vec<u32> = Vec::new();
        let mut rand = rand::thread_rng();

        (0..self.genome_length).for_each(|_| {genome.push(rand.gen::<u32>())});

        genome
    }

    fn add_obstacles(&mut self) {
        for obstacle in &mut *self.obstacles {
            let mut mask = 0;
            (obstacle.0.1..=obstacle.1.1).for_each(|x| mask += (2 as u128).pow(x));

            for x in obstacle.0.0..=obstacle.1.0 {
                self.world.lock().unwrap()[x as usize] |= mask;
                //println!("{}", self.world[x as usize]);
            }
        }
    }

    fn generate_initial_generation(&mut self) {
        for _ in 0..self.population {
            let pos: (u32, u32) = self.rand_pos();

            self.agents.lock().unwrap().push(Agent::new(
                &self.random_genome(),
                self.amount_inners as u8,
                pos
            ));
        }

        self.add_obstacles();
    }
}