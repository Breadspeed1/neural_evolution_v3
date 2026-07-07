use std::fs::File;
use std::time::{Duration, Instant};
use image::{Frame, ImageBuffer, RgbaImage};
use image::codecs::gif::{GifEncoder};
use macroquad::prelude::*;
use ::rand::RngExt;
use crate::agent::Agent;

mod agent;

/// Y coordinate an agent must exceed at generation end to survive and
/// reproduce. The barrier obstacle sits on this row.
const SURVIVAL_Y: u32 = 108;

fn window_conf() -> Conf {
    Conf {
        window_title: "Neural Evolution".to_owned(),
        window_width: 1080,
        window_height: 1080,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let genome_length: u32 = 256;
    let amount_inners: u32 = 225;
    let mutation_rate: f32 = 0.001;
    let steps_per_generation: u32 = 200;
    let population: u32 = 1000;
    let generate_gifs: bool = false;
    let obstacles: Vec<((u32, u32), (u32, u32))> = vec![
        ((10, SURVIVAL_Y), (118, SURVIVAL_Y)),
        /*((10, 107), (10, 20)),
        ((118, 107), (118, 20))*/
    ];

    let mut simulator = Simulator::new(
        genome_length,
        amount_inners,
        mutation_rate,
        steps_per_generation,
        population,
        obstacles,
        generate_gifs,
    );

    run_sim(&mut simulator).await;
}

async fn run_sim(simulator: &mut Simulator) {
    simulator.generate_initial_generation();
    let generations: u32 = 723048912;
    // step delay in milliseconds (q lowers, e raises, floor 0)
    let mut step_delay: u64 = 5;
    // whether the world is rendered each frame
    let mut display: bool = true;
    // when display is off, how many sim steps to run per frame
    let steps_per_frame_headless: u32 = 1000;

    let now = Instant::now();

    while simulator.generation < generations {
        // --- input handling ---
        if is_key_pressed(KeyCode::Q) {
            step_delay = step_delay.saturating_sub(5);
            println!("step delay changed to {}ms", step_delay);
        }
        if is_key_pressed(KeyCode::E) {
            step_delay += 5;
            println!("step delay changed to {}ms", step_delay);
        }
        if is_key_pressed(KeyCode::V) {
            display = !display;
            println!("toggling display");
        }

        if display {
            let frame = simulator.step(true);
            if simulator.current_steps == 0 {
                println!("on generation {}", simulator.generation);
            }

            clear_background(WHITE);
            if let Some(image) = frame {
                draw_world(&image);
            }

            if step_delay > 0 {
                std::thread::sleep(Duration::from_millis(step_delay));
            }
            next_frame().await;
        } else {
            // headless burst: run many steps without drawing, then yield a frame
            for _ in 0..steps_per_frame_headless {
                simulator.step(false);
                if simulator.current_steps == 0 {
                    println!("on generation {}", simulator.generation);
                }
                if simulator.generation >= generations {
                    break;
                }
            }
            clear_background(WHITE);
            next_frame().await;
        }
    }

    println!("{} gens took {} minutes", generations, now.elapsed().as_secs_f32() / 60.0);
}

/// Draw the 128x128 world scaled up to fill the window, preserving aspect
/// ratio, with nearest-neighbour filtering for crisp pixels.
fn draw_world(image: &RgbaImage) {
    let texture = Texture2D::from_rgba8(128, 128, image.as_raw());
    texture.set_filter(FilterMode::Nearest);

    let size = screen_width().min(screen_height());
    let x = (screen_width() - size) / 2.0;
    let y = (screen_height() - size) / 2.0;

    draw_texture_ex(
        &texture,
        x,
        y,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(size, size)),
            ..Default::default()
        },
    );
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
    world: Vec<u128>,
    agents: Vec<Agent>,
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
    use_output: bool,
}

impl Simulator {
    fn new(genome_length: u32, amount_inners: u32, mutation_rate: f32, steps_per_generation: u32, population: u32, obstacles: Vec<((u32, u32), (u32, u32))>, use_output: bool) -> Simulator {
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
            ],
            output: GenerationOutput::new(),
            obstacles,
            use_output,
        }
    }

    fn spawn_next_generation(&mut self) {
        if self.use_output {
            self.reset_output()
        }

        self.generation += 1;
        self.remove_losers();

        // Extinction: no survivors means there is nothing to reproduce from.
        // Reseed the generation with fresh random genomes instead of crashing
        // on a modulo-by-zero.
        if self.agents.is_empty() {
            println!("extinction at generation {} - reseeding with random genomes", self.generation);
            let mut new_generation: Vec<Agent> = Vec::new();
            for _ in 0..self.population {
                let pos: (u32, u32) = self.rand_pos();
                new_generation.push(Agent::new(
                    &self.random_genome(),
                    self.amount_inners as u8,
                    pos
                ));
            }
            self.agents = new_generation;
            return;
        }

        let mut new_generation: Vec<Agent> = Vec::new();

        for i in 0..self.population {
            let mut a: Agent = self.agents[i as usize % self.agents.len()].clone();
            new_generation.push(a.produce_child(
                self.mutation_rate,
                self.rand_pos()
            ));
        }

        self.agents = new_generation;
    }

    fn rand_pos(&mut self) -> (u32, u32) {
        let mut rand = ::rand::rng();
        let mut pos: (u32, u32) = (rand.random_range(0..=127), rand.random_range(0..=127));
        while self.get_pos(pos) {
            pos = (rand.random_range(0..=127), rand.random_range(0..=127));
        }
        self.toggle_pos(pos);

        pos
    }

    fn update_output(&mut self) {
        self.output.add_step(self.world.clone());
    }

    fn reset_output(&mut self) {
        std::fs::create_dir_all("output").unwrap();
        self.output.save(format!("output/generation-{}.gif", self.generation));
        self.output = GenerationOutput::new();
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
            if pos.1 > SURVIVAL_Y {
                winners.push(agent.clone());
            }
        }

        self.agents = winners;
        self.clear_world();
    }

    fn clear_world(&mut self) {
        self.world = vec![0; 128];
        self.add_obstacles();
    }

    /// Advance the simulation one step. When `show_image` is true, returns the
    /// rendered 128x128 world for this step; otherwise returns `None`.
    fn step(&mut self, show_image: bool) -> Option<RgbaImage> {
        if self.current_steps >= self.steps_per_generation {
            self.spawn_next_generation();
            self.current_steps = 0;
            return None;
        }

        let mut image: RgbaImage = RgbaImage::default();
        if show_image {
            image = ImageBuffer::new(128, 128);
            image.fill(u8::MAX);
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

            if show_image {
                let pix = image.get_pixel_mut(pos.0, pos.1);
                pix.0 = self.agents[i].get_rgba();
            }
        }

        self.current_steps += 1;

        if self.use_output {
            self.update_output();
        }

        if show_image {
            Some(image)
        } else {
            None
        }
    }


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

        for a in &mut *self.agents {
            av = (av.0 + a.get_pos().0, av.1 + a.get_pos().1);
        }

        let n = self.agents.len() as u32;
        if n > 0 {
            av = (av.0 / n, av.1 / n);
        }

        inputs[0] = 0.0;
        inputs[1] = 1.0;
        inputs[2] = (self.current_steps % 2) as f32;
        inputs[3] = self.current_steps as f32/self.steps_per_generation as f32;
        inputs[4] = ::rand::rng().random_range(0.0..1.0);
        inputs[5] = av.1 as f32 / 127.0;
        inputs[6] = av.0 as f32 / 127.0;

        inputs
    }

    fn random_genome(&self) -> Vec<u32> {
        let mut genome: Vec<u32> = Vec::new();
        let mut rand = ::rand::rng();

        (0..self.genome_length).for_each(|_| {genome.push(rand.random::<u32>())});

        genome
    }

    fn add_obstacles(&mut self) {
        for obstacle in &mut *self.obstacles {
            let mut mask = 0;
            (obstacle.0.1..=obstacle.1.1).for_each(|x| mask += (2 as u128).pow(x));

            for x in obstacle.0.0..=obstacle.1.0 {
                self.world[x as usize] |= mask;
            }
        }
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

        self.add_obstacles();
    }
}
