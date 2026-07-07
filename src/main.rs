use std::fs::File;
use std::time::{Duration, Instant};
use image::{Frame, ImageBuffer, RgbaImage};
use image::codecs::gif::{GifEncoder};
use macroquad::prelude::*;
use ::rand::RngExt;
use rayon::prelude::*;
use crate::agent::Agent;

mod agent;

/// Read a single world cell from a snapshot (column-bitmask) world without
/// needing `&mut Simulator`, so the decision phase can run in parallel.
#[inline]
fn world_get(world: &[u128], coords: (u32, u32)) -> bool {
    (world[coords.0 as usize] >> coords.1) & 1 == 1
}

/// Build an agent's full input vector: the shared base inputs plus the 8
/// directional "can I move here" sensors, read against a world snapshot.
fn calc_positional_inputs(
    world: &[u128],
    move_vectors: &[(i32, i32)],
    pos: (u32, u32),
    base: &[f32],
    used: Vec<usize>,
) -> Vec<f32> {
    let mut copy = base.to_vec();
    copy.extend_from_slice(&[0.0; 8]);

    for i in used {
        let vec: (i32, i32) = move_vectors[i - 7];
        let neighbor = (
            (pos.0 as i32 + vec.0).clamp(0, 127) as u32,
            (pos.1 as i32 + vec.1).clamp(0, 127) as u32,
        );
        copy[i] = if world_get(world, neighbor) { 0.0 } else { 1.0 };
    }

    copy
}

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

    if std::env::args().any(|a| a == "--bench") {
        simulator.generate_initial_generation();
        let target = simulator.generation + 50;
        let start = Instant::now();
        while simulator.generation < target {
            simulator.step();
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("BENCH: 50 gens in {:.3}s = {:.3} gens/sec", elapsed, 50.0 / elapsed);
        return;
    }

    run_sim(&mut simulator).await;
}

/// Speed model: a target number of sim steps per second. `q` halves it, `e`
/// doubles it; above `UNLIMITED_THRESHOLD` the sim runs a full frame-time
/// budget of steps ("unlimited").
const MIN_SPS: f64 = 1.0;
const UNLIMITED_THRESHOLD: f64 = 100_000.0;
/// Wall-clock budget spent stepping per frame when running unlimited/headless.
const DISPLAY_BUDGET: Duration = Duration::from_millis(6);
const HEADLESS_BUDGET: Duration = Duration::from_millis(250);

async fn run_sim(simulator: &mut Simulator) {
    simulator.generate_initial_generation();
    let generations: u32 = 723048912;

    // target sim steps per second; `unlimited` runs as fast as the budget allows
    let mut target_sps: f64 = 200.0;
    let mut unlimited: bool = false;
    // whether the world is rendered each frame
    let mut display: bool = true;

    // rate-limiter accumulator (fractional steps carried between frames)
    let mut step_accum: f64 = 0.0;

    // persistent texture we update in place instead of recreating each frame
    let mut frame_image = Image::gen_image_color(128, 128, WHITE);
    let texture = Texture2D::from_image(&frame_image);
    texture.set_filter(FilterMode::Nearest);

    // stats
    let mut last_gen = simulator.generation;
    let mut steps_since_report: u64 = 0;
    let mut gens_since_report: u64 = 0;
    let mut last_report = Instant::now();
    let mut last_frame = Instant::now();

    let now = Instant::now();

    while simulator.generation < generations {
        // --- input handling ---
        if is_key_pressed(KeyCode::Q) {
            if unlimited {
                unlimited = false;
                target_sps = UNLIMITED_THRESHOLD;
            }
            target_sps = (target_sps * 0.5).max(MIN_SPS);
            println!("target {:.0} steps/sec", target_sps);
        }
        if is_key_pressed(KeyCode::E) {
            target_sps *= 2.0;
            if target_sps >= UNLIMITED_THRESHOLD {
                unlimited = true;
                println!("target steps/sec: unlimited");
            } else {
                println!("target {:.0} steps/sec", target_sps);
            }
        }
        if is_key_pressed(KeyCode::V) {
            display = !display;
            println!("display {}", if display { "on" } else { "off" });
        }

        // --- run sim steps for this frame ---
        let frame_start = Instant::now();
        let dt = frame_start.duration_since(last_frame).as_secs_f64();
        last_frame = frame_start;

        let mut steps_this_frame: u64 = 0;
        if !display || unlimited {
            // time-budgeted burst
            let budget = if display { DISPLAY_BUDGET } else { HEADLESS_BUDGET };
            loop {
                for _ in 0..64 {
                    simulator.step();
                    steps_this_frame += 1;
                }
                if frame_start.elapsed() >= budget {
                    break;
                }
            }
        } else {
            // rate-limited by target_sps
            step_accum += target_sps * dt;
            // cap to avoid a spiral of death after a hitch
            let cap = (target_sps * 0.25).ceil().max(1.0);
            let n = step_accum.floor().min(cap);
            step_accum -= n;
            for _ in 0..(n as u64) {
                simulator.step();
                steps_this_frame += 1;
            }
        }

        // track generation progress
        if simulator.generation != last_gen {
            gens_since_report += simulator.generation.wrapping_sub(last_gen) as u64;
            last_gen = simulator.generation;
            println!("on generation {}", simulator.generation);
        }
        steps_since_report += steps_this_frame;

        // periodic speed report
        if last_report.elapsed() >= Duration::from_secs(2) {
            let secs = last_report.elapsed().as_secs_f64();
            println!(
                "~{:.0} steps/sec, {:.1} gens/min",
                steps_since_report as f64 / secs,
                gens_since_report as f64 / secs * 60.0,
            );
            steps_since_report = 0;
            gens_since_report = 0;
            last_report = Instant::now();
        }

        // --- draw ---
        if display {
            clear_background(WHITE);
            draw_world(simulator, &mut frame_image, &texture);
        } else {
            clear_background(BLACK);
        }
        next_frame().await;
    }

    println!("{} gens took {} minutes", generations, now.elapsed().as_secs_f32() / 60.0);
}

/// Draw the 128x128 world scaled up to fill the window, preserving aspect
/// ratio, with nearest-neighbour filtering. The pixel buffer is built straight
/// from the agent list into a persistent `Image`/`Texture2D` (updated in place)
/// rather than allocating a new texture each frame.
fn draw_world(simulator: &Simulator, image: &mut Image, texture: &Texture2D) {
    let pixels = image.get_image_data_mut();
    pixels.fill([255, 255, 255, 255]);
    for agent in &simulator.agents {
        let (x, y) = agent.get_pos();
        pixels[(y * 128 + x) as usize] = agent.get_rgba();
    }
    texture.update(image);

    let size = screen_width().min(screen_height());
    let x = (screen_width() - size) / 2.0;
    let y = (screen_height() - size) / 2.0;

    draw_texture_ex(
        texture,
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

        // Reserve unique spawn positions serially (mutates the world), then
        // produce the children (genome mutation + Brain decode) in parallel.
        let positions: Vec<(u32, u32)> =
            (0..self.population).map(|_| self.rand_pos()).collect();
        let len = self.agents.len();
        let mutation_rate = self.mutation_rate;
        let parents = &self.agents;

        let new_generation: Vec<Agent> = positions
            .par_iter()
            .enumerate()
            .map(|(i, &pos)| parents[i % len].produce_child(mutation_rate, pos))
            .collect();

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

    /// Advance the simulation one step.
    ///
    /// Split into two phases for parallelism:
    ///   (a) decision phase — parallel over agents against a start-of-tick
    ///       snapshot of the world; each agent computes its sensor inputs and
    ///       runs `Brain::step` to produce a desired translation.
    ///   (b) application phase — serial, in agent order, resolving collisions
    ///       against the live (mutating) world exactly as before.
    ///
    /// Semantics differ slightly from the fully-serial version: agents now see
    /// the world as it was at the start of the tick rather than partially
    /// updated by earlier agents in the same tick.
    fn step(&mut self) {
        if self.current_steps >= self.steps_per_generation {
            self.spawn_next_generation();
            self.current_steps = 0;
            return;
        }

        let inputs = self.calc_step_inputs();
        let world_snapshot = self.world.clone();
        let move_vectors = self.move_vectors.clone();

        // (a) parallel decision phase
        let translations: Vec<(i32, i32)> = self
            .agents
            .par_iter_mut()
            .map(|agent| {
                let agent_pos = agent.get_pos();
                let used = agent.get_used_inputs();
                let all_inputs =
                    calc_positional_inputs(&world_snapshot, &move_vectors, agent_pos, &inputs, used);
                agent.step(all_inputs)
            })
            .collect();

        // (b) serial application phase
        for i in 0..self.agents.len() {
            let agent_pos: (u32, u32) = self.agents[i].get_pos();
            let translation = translations[i];
            let pos: (u32, u32) = (
                (agent_pos.0 as i32 + translation.0).clamp(0, 127) as u32,
                (agent_pos.1 as i32 + translation.1).clamp(0, 127) as u32,
            );

            if !self.get_pos(pos) {
                self.toggle_pos(agent_pos);
                self.agents[i].set_pos(pos);
                self.toggle_pos(pos);
            }
        }

        self.current_steps += 1;

        if self.use_output {
            self.update_output();
        }
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
