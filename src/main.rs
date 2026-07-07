use std::time::{Duration, Instant};

use clap::Parser;
use macroquad::prelude::*;
use neural_evolution::{Cli, Simulator, build_simulator};

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
    let cli = Cli::parse();
    let mut simulator = build_simulator(&cli);
    let generations = cli.generations.unwrap_or(u32::MAX);
    run_sim(&mut simulator, generations).await;
}

/// Speed model: a target number of sim steps per second. `q` halves it, `e`
/// doubles it; above `UNLIMITED_THRESHOLD` the sim runs a full frame-time
/// budget of steps ("unlimited").
const MIN_SPS: f64 = 1.0;
const UNLIMITED_THRESHOLD: f64 = 100_000.0;
/// Wall-clock budget spent stepping per frame when running unlimited/headless.
const DISPLAY_BUDGET: Duration = Duration::from_millis(6);
const HEADLESS_BUDGET: Duration = Duration::from_millis(250);

async fn run_sim(simulator: &mut Simulator, generations: u32) {
    simulator.generate_initial_generation();

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

    println!(
        "{} gens took {} minutes",
        generations,
        now.elapsed().as_secs_f32() / 60.0
    );
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
