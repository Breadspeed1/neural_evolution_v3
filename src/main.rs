use std::collections::HashSet;
use std::time::{Duration, Instant};

use clap::Parser;
use macroquad::prelude::*;
use neural_evolution::agent::Agent;
use neural_evolution::{Cli, Simulator, build_simulator};

fn window_conf() -> Conf {
    Conf {
        window_title: "Neural Evolution".to_owned(),
        window_width: 1720,
        window_height: 1080,
        window_resizable: true,
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

// ---------------------------------------------------------------------------
// Dark palette (verbatim from the validated design spec). Series hues (blue /
// aqua / edge red) are reserved for data; text always uses the ink colors.
// ---------------------------------------------------------------------------
const BG_PAGE: Color = Color::new(13.0 / 255.0, 13.0 / 255.0, 13.0 / 255.0, 1.0); // #0d0d0d
const PANEL: Color = Color::new(26.0 / 255.0, 26.0 / 255.0, 25.0 / 255.0, 1.0); // #1a1a19
const TEXT_PRIMARY: Color = Color::new(1.0, 1.0, 1.0, 1.0); // #ffffff
const TEXT_SECONDARY: Color = Color::new(195.0 / 255.0, 194.0 / 255.0, 183.0 / 255.0, 1.0); // #c3c2b7
const MUTED: Color = Color::new(137.0 / 255.0, 135.0 / 255.0, 129.0 / 255.0, 1.0); // #898781
const HAIRLINE: Color = Color::new(1.0, 1.0, 1.0, 0.10); // rgba(255,255,255,0.10)
const BASELINE: Color = Color::new(56.0 / 255.0, 56.0 / 255.0, 53.0 / 255.0, 1.0); // #383835
const SURVIVAL_HUE: Color = Color::new(57.0 / 255.0, 135.0 / 255.0, 229.0 / 255.0, 1.0); // #3987e5
const DIVERSITY_HUE: Color = Color::new(25.0 / 255.0, 158.0 / 255.0, 112.0 / 255.0, 1.0); // #199e70
const EDGE_NEG: Color = Color::new(230.0 / 255.0, 103.0 / 255.0, 103.0 / 255.0, 1.0); // #e66767
const GOOD: Color = Color::new(12.0 / 255.0, 163.0 / 255.0, 12.0 / 255.0, 1.0); // #0ca30c
const CRITICAL: Color = Color::new(208.0 / 255.0, 59.0 / 255.0, 59.0 / 255.0, 1.0); // #d03b3b

// World-terrain colors, as premultiplied [u8;4] for the 128x128 texture.
const WORLD_BG_PX: [u8; 4] = [17, 17, 16, 255]; // #111110
const SAFE_PX: [u8; 4] = [15, 39, 15, 255]; // GOOD @ ~15% over world bg
const OBSTACLE_PX: [u8; 4] = [56, 56, 53, 255]; // #383835

const MARGIN: f32 = 14.0;
/// How many recent positions each agent's motion trail retains.
const TRAIL_LEN: usize = 18;
/// Turnover pulse duration (wall-clock, independent of sim speed).
const PULSE_SECS: f32 = 0.5;

/// One agent's final `(position, survived)` pair, snapshotted at turnover.
type FinalState = ((u32, u32), bool);
/// An active generation-turnover pulse: its start time and the finals to flash.
type Pulse = (Instant, Vec<FinalState>);

/// How the agents are tinted in the world view.
#[derive(Clone, Copy, PartialEq)]
enum ColorMode {
    Genome,
    Lineage,
}

impl ColorMode {
    fn label(self) -> &'static str {
        match self {
            ColorMode::Genome => "genome",
            ColorMode::Lineage => "lineage",
        }
    }
}

async fn run_sim(simulator: &mut Simulator, generations: u32) {
    simulator.generate_initial_generation();

    // target sim steps per second; `unlimited` runs as fast as the budget allows
    let mut target_sps: f64 = 200.0;
    let mut unlimited: bool = false;
    // whether the world is rendered each frame
    let mut display: bool = true;
    // occupancy heatmap toggle (key `h`)
    let mut heatmap: bool = false;

    // rate-limiter accumulator (fractional steps carried between frames)
    let mut step_accum: f64 = 0.0;

    // persistent texture we update in place instead of recreating each frame
    let mut frame_image = Image::gen_image_color(128, 128, WHITE);
    let texture = Texture2D::from_image(&frame_image);
    texture.set_filter(FilterMode::Nearest);

    // --- dashboard state (all skipped while display is off) ---
    // Challenge-aware terrain overlay, cached per generation.
    let mut overlay: Vec<[u8; 4]> = vec![WORLD_BG_PX; 128 * 128];
    let mut overlay_gen: Option<u32> = None;
    let mut color_mode = ColorMode::Genome;
    // Selected agent for the brain inspector, tracked by (index, generation).
    let mut selected: Option<usize> = None;
    let mut selected_gen: u32 = 0;
    let mut graph: Option<BrainGraph> = None;

    // --- motion-legibility state (viewer-side only) ---
    // Per-agent motion trails (index-keyed within a generation, cleared at
    // turnover). Oldest position first, newest last.
    let mut trails: Vec<Vec<(u8, u8)>> = Vec::new();
    // Occupancy accumulator for the current generation (visited-cell counts).
    let mut heat: Vec<u32> = vec![0; 128 * 128];
    // Active turnover pulse: (start time, final (pos, survived) snapshot).
    let mut pulse: Option<Pulse> = None;

    // stats
    let mut last_gen = simulator.generation;
    let mut steps_since_report: u64 = 0;
    let mut gens_since_report: u64 = 0;
    let mut last_report = Instant::now();
    let mut last_frame = Instant::now();
    let mut fps_smooth: f64 = 60.0;
    let mut sps_smooth: f64 = 0.0;

    let now = Instant::now();

    // Optional smoke-test screenshot: if NEURAL_SCREENSHOT=<path> is set, grab
    // one frame after warmup and write it, then keep running normally.
    let screenshot_path = std::env::var("NEURAL_SCREENSHOT").ok();
    let mut frame_no: u64 = 0;

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
        if display && is_key_pressed(KeyCode::C) {
            color_mode = match color_mode {
                ColorMode::Genome => ColorMode::Lineage,
                ColorMode::Lineage => ColorMode::Genome,
            };
            println!("color mode: {}", color_mode.label());
        }
        if display && is_key_pressed(KeyCode::H) {
            heatmap = !heatmap;
            for c in heat.iter_mut() {
                *c = 0;
            }
            println!("heatmap {}", if heatmap { "on" } else { "off" });
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

        // track generation progress + detect turnover for the pulse/trail reset
        let turned = simulator.generation != last_gen;
        if turned {
            gens_since_report += simulator.generation.wrapping_sub(last_gen) as u64;
            // Capture the just-ended generation's final state for the pulse.
            if display {
                pulse = Some((Instant::now(), simulator.last_final().to_vec()));
                trails.clear();
                for c in heat.iter_mut() {
                    *c = 0;
                }
            }
            last_gen = simulator.generation;
            println!("on generation {}", simulator.generation);
        }
        steps_since_report += steps_this_frame;

        // smoothed readouts for the corner HUD
        if dt > 0.0 {
            fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1;
            sps_smooth = sps_smooth * 0.9 + (steps_this_frame as f64 / dt) * 0.1;
        }

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
            clear_background(BG_PAGE);

            let world_size = screen_height();

            // (Re)build the challenge overlay only when the generation changes.
            if overlay_gen != Some(simulator.generation) {
                build_overlay(simulator, &mut overlay);
                overlay_gen = Some(simulator.generation);
                // Agents were replaced on turnover: reselect via the heuristic.
                let idx = best_agent(simulator);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }
            if selected.is_none() {
                let idx = best_agent(simulator);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }

            // Mouse click selects the nearest agent within a few cells.
            if is_mouse_button_pressed(MouseButton::Left) {
                let (mx, my) = mouse_position();
                if mx < world_size && my < world_size {
                    let cx = (mx / world_size * 128.0) as i32;
                    let cy = (my / world_size * 128.0) as i32;
                    if let Some(idx) = nearest_agent(simulator, cx, cy, 4) {
                        set_selection(simulator, Some(idx), &mut selected, &mut selected_gen, &mut graph);
                    }
                }
            }
            // `b` reselects the heuristic "best" agent.
            if is_key_pressed(KeyCode::B) {
                let idx = best_agent(simulator);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }

            // Record one trail sample + heatmap sample per frame (correct across
            // turnover, which was handled above, and cheap regardless of speed).
            record_trails(&mut trails, &simulator.agents);
            if heatmap {
                accumulate_heat(&mut heat, &simulator.agents);
            }

            draw_world(
                simulator, &overlay, &heat, heatmap, color_mode, selected, &trails, pulse.as_ref(),
                world_size, &mut frame_image, &texture,
            );
            draw_dashboard(
                simulator, world_size, target_sps, unlimited, display, heatmap, color_mode,
                selected, graph.as_ref(), fps_smooth, sps_smooth,
            );
        } else {
            clear_background(BG_PAGE);
        }

        frame_no += 1;
        if let Some(path) = &screenshot_path
            && frame_no == 200
        {
            get_screen_data().export_png(path);
            println!("screenshot saved to {path}");
        }
        next_frame().await;
    }

    println!(
        "{} gens took {} minutes",
        generations,
        now.elapsed().as_secs_f32() / 60.0
    );
}

/// Append the current agent positions as the newest trail sample, capped to
/// `TRAIL_LEN`. Consecutive duplicates are skipped so a stationary agent keeps a
/// short trail rather than a stack of identical points. Trails are index-keyed
/// and rebuilt whenever the population size changes (belt-and-suspenders with the
/// explicit clear at turnover).
fn record_trails(trails: &mut Vec<Vec<(u8, u8)>>, agents: &[Agent]) {
    if trails.len() != agents.len() {
        trails.clear();
        trails.resize(agents.len(), Vec::new());
    }
    for (i, a) in agents.iter().enumerate() {
        let (x, y) = a.get_pos();
        let p = (x as u8, y as u8);
        let t = &mut trails[i];
        if t.last() == Some(&p) {
            continue;
        }
        t.push(p);
        if t.len() > TRAIL_LEN {
            t.remove(0);
        }
    }
}

/// Accumulate one occupancy sample (current agent cells) into the heat grid.
fn accumulate_heat(heat: &mut [u32], agents: &[Agent]) {
    for a in agents {
        let (x, y) = a.get_pos();
        heat[(y * 128 + x) as usize] += 1;
    }
}

/// Heuristic "best" agent for auto-selection: prefer an agent currently in the
/// survival zone, breaking ties (and the no-survivor case) by highest `y`. This
/// is exactly right for the band/gauntlet challenges (whose goal is high `y`)
/// and picks a live winner for the others when one exists. Returns None only for
/// an empty population.
fn best_agent(sim: &Simulator) -> Option<usize> {
    sim.agents
        .iter()
        .enumerate()
        .max_by_key(|(_, a)| (sim.is_safe(a.get_pos()) as u8, a.get_pos().1))
        .map(|(i, _)| i)
}

/// Nearest agent to cell (cx, cy) within `radius` cells (Chebyshev), if any.
fn nearest_agent(sim: &Simulator, cx: i32, cy: i32, radius: i32) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    for (i, a) in sim.agents.iter().enumerate() {
        let (ax, ay) = a.get_pos();
        let d2 = (ax as i32 - cx).pow(2) + (ay as i32 - cy).pow(2);
        if d2 <= radius * radius && best.is_none_or(|(bd, _)| d2 < bd) {
            best = Some((d2, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Assign the selection and rebuild the (cached) brain layout for it.
fn set_selection(
    sim: &Simulator,
    idx: Option<usize>,
    selected: &mut Option<usize>,
    selected_gen: &mut u32,
    graph: &mut Option<BrainGraph>,
) {
    *selected = idx;
    *selected_gen = sim.generation;
    *graph = idx
        .and_then(|i| sim.agents.get(i))
        .map(BrainGraph::build);
}

/// Build the 128x128 terrain overlay: survival zone tinted, obstacles solid,
/// everything else the world background. Pure function of the current
/// generation, so it is cached per generation by the caller.
fn build_overlay(sim: &Simulator, out: &mut [[u8; 4]]) {
    for y in 0..128u32 {
        for x in 0..128u32 {
            let c = if sim.is_safe((x, y)) { SAFE_PX } else { WORLD_BG_PX };
            out[(y * 128 + x) as usize] = c;
        }
    }
    for &((x0, y0), (x1, y1)) in &sim.obstacles {
        for x in x0..=x1 {
            for y in y0..=y1 {
                out[(y * 128 + x) as usize] = OBSTACLE_PX;
            }
        }
    }
}

/// Alpha-blend `src` (with alpha `a`, 0..1) over the opaque `dst` pixel.
fn blend_px(dst: [u8; 4], src: [u8; 3], a: f32) -> [u8; 4] {
    let mix = |d: u8, s: u8| ((d as f32) * (1.0 - a) + (s as f32) * a) as u8;
    [mix(dst[0], src[0]), mix(dst[1], src[1]), mix(dst[2], src[2]), 255]
}

/// Draw the world square (terrain + heatmap + trails + oriented agents + pulse)
/// at the top-left, scaled up with nearest-neighbour filtering.
#[allow(clippy::too_many_arguments)]
fn draw_world(
    sim: &Simulator,
    overlay: &[[u8; 4]],
    heat: &[u32],
    heatmap: bool,
    color_mode: ColorMode,
    selected: Option<usize>,
    trails: &[Vec<(u8, u8)>],
    pulse: Option<&Pulse>,
    world_size: f32,
    image: &mut Image,
    texture: &Texture2D,
) {
    // Composite terrain (+ optional heat wash) into the texture.
    let pixels = image.get_image_data_mut();
    pixels.copy_from_slice(overlay);
    if heatmap {
        let maxc = heat.iter().copied().max().unwrap_or(0).max(1);
        let lmax = ((maxc + 1) as f32).ln();
        for (i, &c) in heat.iter().enumerate() {
            if c > 0 {
                let t = ((c + 1) as f32).ln() / lmax;
                pixels[i] = blend_px(pixels[i], [57, 135, 229], 0.55 * t);
            }
        }
    }
    texture.update(image);
    draw_texture_ex(
        texture,
        0.0,
        0.0,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(world_size, world_size)),
            ..Default::default()
        },
    );

    let cell = world_size / 128.0;
    let sx = |cx: f32| (cx + 0.5) * cell;

    // Motion trails, under the agents. Newer samples brighter; the selected
    // agent gets a stronger, thicker trail so it stands out.
    for (i, trail) in trails.iter().enumerate() {
        if trail.len() < 2 {
            continue;
        }
        let Some(a) = sim.agents.get(i) else { continue };
        let col = agent_color(a, color_mode);
        let is_sel = selected == Some(i);
        let n = trail.len();
        for k in 1..n {
            let (x0, y0) = trail[k - 1];
            let (x1, y1) = trail[k];
            let f = k as f32 / n as f32; // 0 (old) .. 1 (new)
            let alpha = if is_sel { 0.30 + 0.60 * f } else { 0.10 + 0.40 * f };
            let thick = if is_sel { 2.2 } else { 1.3 };
            let c = Color::new(col.r, col.g, col.b, alpha);
            draw_line(sx(x0 as f32), sx(y0 as f32), sx(x1 as f32), sx(y1 as f32), thick, c);
        }
    }

    // Agents: an oriented chevron along the last-move direction, or a dot if it
    // didn't move this frame.
    for (i, a) in sim.agents.iter().enumerate() {
        let (gx, gy) = a.get_pos();
        let (px, py) = (sx(gx as f32), sx(gy as f32));
        let col = agent_color(a, color_mode);
        let dir = trails.get(i).and_then(|t| {
            (t.len() >= 2).then(|| {
                let a = t[t.len() - 2];
                let b = t[t.len() - 1];
                (b.0 as f32 - a.0 as f32, b.1 as f32 - a.1 as f32)
            })
        });
        match dir {
            Some((dx, dy)) if dx != 0.0 || dy != 0.0 => {
                let len = (dx * dx + dy * dy).sqrt();
                let (fx, fy) = (dx / len, dy / len);
                let (perp_x, perp_y) = (-fy, fx);
                let fwd = cell * 1.15;
                let back = cell * 0.55;
                let half = cell * 0.62;
                let tip = vec2(px + fx * fwd, py + fy * fwd);
                let l = vec2(px - fx * back + perp_x * half, py - fy * back + perp_y * half);
                let r = vec2(px - fx * back - perp_x * half, py - fy * back - perp_y * half);
                draw_triangle(tip, l, r, col);
            }
            _ => draw_circle(px, py, cell * 0.5, col),
        }
    }

    // Selection ring (bright, always legible on dark).
    if let Some(i) = selected
        && let Some(a) = sim.agents.get(i)
    {
        let (x, y) = a.get_pos();
        let r = cell * 3.0;
        draw_circle_lines(sx(x as f32), sx(y as f32), r, 2.0, Color::new(1.0, 1.0, 1.0, 0.95));
        draw_circle_lines(sx(x as f32), sx(y as f32), r + 2.0, 1.0, Color::new(1.0, 1.0, 1.0, 0.35));
    }

    // Generation punctuation: expanding rings at last generation's final
    // positions — green for survivors, red for the culled — fading over ~0.5s.
    if let Some((start, finals)) = pulse {
        let t = start.elapsed().as_secs_f32() / PULSE_SECS;
        if t < 1.0 {
            let ease = 1.0 - t;
            for &((x, y), survived) in finals {
                let base = if survived { GOOD } else { CRITICAL };
                let r = cell * (1.0 + 3.0 * t);
                draw_circle_lines(sx(x as f32), sx(y as f32), r, 1.6, Color::new(base.r, base.g, base.b, ease));
                if survived {
                    // Faint glow; kept low so dense survivor clusters flash
                    // rather than saturate to a solid wash.
                    draw_circle(sx(x as f32), sx(y as f32), r, Color::new(base.r, base.g, base.b, 0.05 * ease));
                }
            }
        }
    }
}

/// The color an agent is drawn with, constrained to a bright band so it pops on
/// the dark world. Both color modes route through the same HSL constraint: the
/// hue carries the identity (genome-hash or lineage-hash), saturation/lightness
/// are pinned into a vivid range.
fn agent_color(agent: &Agent, mode: ColorMode) -> Color {
    let hue = match mode {
        ColorMode::Genome => rgb_to_hue(agent.get_rgba()),
        ColorMode::Lineage => (agent.lineage as f32 * 0.618_034).fract(),
    };
    let (r, g, b) = hsl_to_rgb(hue, 0.75, 0.62);
    Color::new(r, g, b, 1.0)
}

/// Hue (0..1) of an RGB triple, for re-vivifying genome-hash colors.
fn rgb_to_hue(rgba: [u8; 4]) -> f32 {
    let r = rgba[0] as f32 / 255.0;
    let g = rgba[1] as f32 / 255.0;
    let b = rgba[2] as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    if d < 1e-6 {
        return 0.0;
    }
    let h = if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0).rem_euclid(1.0)
}

/// HSL (h,s,l all 0..1) to linear-ish RGB floats 0..1.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h * 6.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c * 0.5;
    (r1 + m, g1 + m, b1 + m)
}

// ---------------------------------------------------------------------------
// Dashboard (right column)
// ---------------------------------------------------------------------------

/// Fill a panel surface with a hairline border.
fn draw_panel(x: f32, y: f32, w: f32, h: f32) {
    draw_rectangle(x, y, w, h, PANEL);
    draw_rectangle_lines(x, y, w, h, 1.0, HAIRLINE);
}

/// A muted, upper-cased panel header.
fn draw_header(label: &str, x: f32, y: f32) {
    draw_text(label.to_uppercase(), x, y, 15.0, MUTED);
}

#[allow(clippy::too_many_arguments)]
fn draw_dashboard(
    sim: &Simulator,
    world_size: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    heatmap: bool,
    color_mode: ColorMode,
    selected: Option<usize>,
    graph: Option<&BrainGraph>,
    fps: f64,
    sps: f64,
) {
    // Hairline divider between the world and the panel column.
    draw_line(world_size, 0.0, world_size, screen_height(), 1.0, HAIRLINE);

    let col_x = world_size + MARGIN;
    let col_w = (screen_width() - world_size - 2.0 * MARGIN).max(120.0);
    let mut y = MARGIN;

    // Corner HUD (top-right of the world): fps + steps/sec, small and muted.
    let hud = format!("{:.0} fps   {:.0} steps/s", fps, sps);
    let dim = measure_text(&hud, None, 15, 1.0);
    draw_text(&hud, world_size - dim.width - 12.0, 20.0, 15.0, MUTED);

    let status_h = 176.0;
    draw_status_panel(sim, col_x, y, col_w, status_h, target_sps, unlimited, display, heatmap, color_mode, selected);
    y += status_h + MARGIN;

    let spark_h = 116.0;
    let survival: Vec<f32> = sim
        .metrics_history()
        .iter()
        .map(|m| m.survival_rate as f32)
        .collect();
    draw_sparkline(col_x, y, col_w, spark_h, &survival, Some((0.0, 1.0)), "survival", SURVIVAL_HUE, true, "{:.0}%", 100.0);
    y += spark_h + MARGIN;

    let diversity: Vec<f32> = sim
        .metrics_history()
        .iter()
        .map(|m| m.genome_diversity as f32)
        .collect();
    draw_sparkline(col_x, y, col_w, spark_h, &diversity, None, "diversity", DIVERSITY_HUE, false, "{:.0}", 1.0);
    y += spark_h + MARGIN;

    // Legend strip pinned to the bottom of the column.
    let legend = "q/e speed  ·  v display  ·  c colors  ·  b best  ·  h heat  ·  click select";
    let legend_h = 26.0;
    let legend_y = screen_height() - MARGIN - legend_h;
    draw_text(legend, col_x, legend_y + 17.0, 14.0, MUTED);

    let brain_h = (legend_y - MARGIN - y).max(120.0);
    draw_brain_panel(sim, col_x, y, col_w, brain_h, selected, graph);
}

#[allow(clippy::too_many_arguments)]
fn draw_status_panel(
    sim: &Simulator,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    heatmap: bool,
    color_mode: ColorMode,
    selected: Option<usize>,
) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;
    draw_header("status", x + pad, y + 22.0);

    let pop = sim.agents.len();
    let live_safe = sim.agents.iter().filter(|a| sim.is_safe(a.get_pos())).count();
    let cur_pct = if pop > 0 { live_safe as f64 / pop as f64 * 100.0 } else { 0.0 };
    let hist = sim.metrics_history();
    let last = hist.last();

    let speed = if unlimited { "unlimited".to_string() } else { format!("{:.0}/s", target_sps) };
    let last_str = match last {
        Some(m) => format!("gen {}  ·  {} survived ({:.0}%)", m.generation, m.survivors, m.survival_rate * 100.0),
        None => "gen 0 (in progress)".to_string(),
    };
    let sel_str = match selected {
        Some(i) => format!("#{i}"),
        None => "none".to_string(),
    };

    // label / value rows, value column aligned.
    let label_x = x + pad;
    let value_x = x + pad + 118.0;
    let row_h = 20.0;
    let mut ly = y + 46.0;
    let row = |label: &str, value: &str, vcol: Color, ly: &mut f32| {
        draw_text(label, label_x, *ly, 17.0, MUTED);
        draw_text(value, value_x, *ly, 17.0, vcol);
        *ly += row_h;
    };
    row("generation", &sim.generation.to_string(), TEXT_PRIMARY, &mut ly);
    row("challenge", sim.challenge().name(), TEXT_SECONDARY, &mut ly);
    row("population", &pop.to_string(), TEXT_SECONDARY, &mut ly);
    row("last gen", &last_str, TEXT_SECONDARY, &mut ly);

    // Current survival with a delta indicator vs the previous completed gen.
    draw_text("survival", label_x, ly, 17.0, MUTED);
    draw_text(format!("{cur_pct:.1}%"), value_x, ly, 17.0, TEXT_PRIMARY);
    if hist.len() >= 2 {
        let cur = hist[hist.len() - 1].survival_rate;
        let prev = hist[hist.len() - 2].survival_rate;
        let delta = (cur - prev) * 100.0;
        let dc = if cur > prev { GOOD } else if cur < prev { CRITICAL } else { MUTED };
        let vw = measure_text(format!("{cur_pct:.1}%"), None, 17, 1.0).width;
        let tx = value_x + vw + 12.0;
        // A drawn direction triangle (the ▲/▼ glyphs aren't in the default
        // font), so the up/down signal reads without relying on color alone.
        if cur > prev {
            draw_triangle(vec2(tx + 5.0, ly - 11.0), vec2(tx, ly - 2.0), vec2(tx + 10.0, ly - 2.0), dc);
        } else if cur < prev {
            draw_triangle(vec2(tx + 5.0, ly - 2.0), vec2(tx, ly - 11.0), vec2(tx + 10.0, ly - 11.0), dc);
        }
        draw_text(format!("{delta:+.1}"), tx + 16.0, ly, 15.0, dc);
    }
    ly += row_h;

    row("speed", &format!("{speed}   display {}", if display { "on" } else { "off" }), TEXT_SECONDARY, &mut ly);
    row("color", &format!("{}   heat {}   sel {}", color_mode.label(), if heatmap { "on" } else { "off" }, sel_str), TEXT_SECONDARY, &mut ly);
}

/// A minimal filled sparkline on a panel surface: muted single-hue fill under a
/// thin line, faint baseline, optional max marker, min/max value labels. `fixed`
/// pins the y-axis (e.g. 0..1 for a rate); otherwise it auto-scales.
#[allow(clippy::too_many_arguments)]
fn draw_sparkline(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    values: &[f32],
    fixed: Option<(f32, f32)>,
    title: &str,
    hue: Color,
    mark_max: bool,
    fmt: &str,
    scale: f32,
) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;
    draw_header(title, x + pad, y + 20.0);

    let plot_x = x + pad;
    let plot_w = (w - 2.0 * pad).max(1.0);
    let plot_y = y + 32.0;
    let plot_h = h - 44.0;
    let baseline = plot_y + plot_h;

    if values.is_empty() {
        draw_line(plot_x, baseline, plot_x + plot_w, baseline, 1.0, BASELINE);
        draw_text("(no data yet)", plot_x, plot_y + plot_h * 0.5, 15.0, MUTED);
        return;
    }

    let pts = downsample(values, plot_w as usize);
    let (lo, mut hi) = match fixed {
        Some((a, b)) => (a, b),
        None => {
            let lo = pts.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = pts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (lo, hi)
        }
    };
    if (hi - lo).abs() < 1e-6 {
        hi = lo + 1.0;
    }
    let map_y = |v: f32| baseline - ((v - lo) / (hi - lo)).clamp(0.0, 1.0) * plot_h;

    let fill = Color::new(hue.r, hue.g, hue.b, 0.16);
    let stroke = Color::new(hue.r, hue.g, hue.b, 0.95);
    let n = pts.len();
    let dx = if n > 1 { plot_w / (n - 1) as f32 } else { 0.0 };

    // Fill under the curve.
    for (i, &v) in pts.iter().enumerate() {
        let px = plot_x + i as f32 * dx;
        draw_line(px, map_y(v), px, baseline, dx.max(1.0), fill);
    }
    // Thin line on top.
    for i in 1..n {
        let x0 = plot_x + (i - 1) as f32 * dx;
        let x1 = plot_x + i as f32 * dx;
        draw_line(x0, map_y(pts[i - 1]), x1, map_y(pts[i]), 2.0, stroke);
    }
    // Faint baseline.
    draw_line(plot_x, baseline, plot_x + plot_w, baseline, 1.0, BASELINE);

    // Max marker (over the *full* series) — a small white tick, not red.
    if mark_max
        && let Some((mi, &mv)) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        let frac = if values.len() > 1 { mi as f32 / (values.len() - 1) as f32 } else { 0.0 };
        let px = plot_x + frac * plot_w;
        draw_circle(px, map_y(mv), 2.6, TEXT_PRIMARY);
    }

    // Min/max value labels.
    let hi_lbl = fmt.replace("{:.0}", &format!("{:.0}", hi * scale));
    let lo_lbl = fmt.replace("{:.0}", &format!("{:.0}", lo * scale));
    draw_text(&hi_lbl, plot_x + plot_w - 44.0, plot_y + 12.0, 14.0, MUTED);
    draw_text(&lo_lbl, plot_x + plot_w - 44.0, baseline - 2.0, 14.0, MUTED);
}

/// Bucket-mean downsample of `values` to at most `target` points.
fn downsample(values: &[f32], target: usize) -> Vec<f32> {
    let n = values.len();
    if target == 0 || n <= target {
        return values.to_vec();
    }
    (0..target)
        .map(|i| {
            let a = i * n / target;
            let b = ((i + 1) * n / target).max(a + 1).min(n);
            let slice = &values[a..b];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Brain inspector
// ---------------------------------------------------------------------------

const INPUT_LABELS: [&str; 15] = [
    "const0", "const1", "osc", "age", "rand", "grad-ns", "grad-ew", // 0..6
    "N", "S", "E", "NE", "SE", "W", "NW", "SW", // 7..14 (move_vectors order)
];
const OUTPUT_LABELS: [&str; 5] = ["rand", "N", "S", "E", "W"];
/// Cap on edges drawn in the brain panel; above this we keep the top-|weight|.
const MAX_EDGES: usize = 120;

struct Edge {
    src_layer: u8,
    src_id: u8,
    sink_layer: u8,
    sink_id: u8,
    weight: f32,
}

/// The pruned, drawable topology of one agent's brain. Recomputed only on
/// (re)selection or generation turnover; activations are read live at draw time.
struct BrainGraph {
    inner_ids: Vec<u8>,
    edges: Vec<Edge>,
    /// Active edges before the MAX_EDGES cap (so we can print "showing 120/N").
    active_total: usize,
}

impl BrainGraph {
    /// Prune to the subgraph that can influence an output (backward reachability
    /// from the output layer), then cap to the top MAX_EDGES by |weight|.
    fn build(agent: &Agent) -> BrainGraph {
        let conns = agent.brain_connections();

        // reaches: inner nodes with a path forward to an output. Fixpoint over
        // connections (sink type 2 = output is terminal-true).
        let mut reaches: HashSet<u8> = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for c in conns {
                let sink_ok = c.sink_type == 2 || reaches.contains(&c.sink_id);
                if sink_ok && c.source_type == 1 && reaches.insert(c.source_id) {
                    changed = true;
                }
            }
        }

        // Keep connections whose sink can reach an output.
        let mut edges: Vec<Edge> = conns
            .iter()
            .filter(|c| c.sink_type == 2 || reaches.contains(&c.sink_id))
            .map(|c| Edge {
                src_layer: c.source_type,
                src_id: c.source_id,
                sink_layer: c.sink_type,
                sink_id: c.sink_id,
                weight: c.weight,
            })
            .collect();
        let active_total = edges.len();

        if edges.len() > MAX_EDGES {
            edges.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap());
            edges.truncate(MAX_EDGES);
        }

        // Inner neurons that appear in the drawn edges.
        let mut inner: HashSet<u8> = HashSet::new();
        for e in &edges {
            if e.src_layer == 1 {
                inner.insert(e.src_id);
            }
            if e.sink_layer == 1 {
                inner.insert(e.sink_id);
            }
        }
        let mut inner_ids: Vec<u8> = inner.into_iter().collect();
        inner_ids.sort_unstable();

        BrainGraph { inner_ids, edges, active_total }
    }
}

fn draw_brain_panel(
    sim: &Simulator,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    selected: Option<usize>,
    graph: Option<&BrainGraph>,
) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;

    let title = match selected {
        Some(i) => format!("brain - agent #{i}"),
        None => "brain".to_string(),
    };
    draw_header(&title, x + pad, y + 22.0);

    let (Some(idx), Some(g)) = (selected, graph) else {
        draw_text("(no agent selected)", x + pad, y + 50.0, 15.0, MUTED);
        return;
    };
    let Some(agent) = sim.agents.get(idx) else {
        draw_text("(selection lost)", x + pad, y + 50.0, 15.0, MUTED);
        return;
    };

    let subtitle = if g.active_total > g.edges.len() {
        format!("lineage {}   ·   showing {}/{} edges", agent.lineage, g.edges.len(), g.active_total)
    } else {
        format!("lineage {}   ·   {} edges, {} inner", agent.lineage, g.edges.len(), g.inner_ids.len())
    };
    draw_text(&subtitle, x + pad, y + 42.0, 14.0, MUTED);

    // Plot area for the node-link diagram.
    let gx = x + pad;
    let gy = y + 58.0;
    let gw = w - 2.0 * pad;
    let gh = h - 66.0;
    if gh < 40.0 {
        return;
    }

    let neurons = agent.brain_neurons();
    let x_in = gx + 46.0;
    let x_out = gx + gw - 46.0;
    let x_mid = (x_in + x_out) * 0.5;

    // Vertical layout helpers: evenly spread `count` nodes across the plot.
    let pos_y = |i: usize, count: usize| -> f32 {
        if count <= 1 {
            gy + gh * 0.5
        } else {
            gy + gh * (i as f32 / (count - 1) as f32)
        }
    };
    let input_y = |id: u8| pos_y(id as usize, 15);
    let output_y = |id: u8| pos_y(id as usize, 5);
    let inner_index: std::collections::HashMap<u8, usize> =
        g.inner_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let inner_y = |id: u8| pos_y(*inner_index.get(&id).unwrap_or(&0), g.inner_ids.len());

    let node_x = |layer: u8| match layer {
        0 => x_in,
        2 => x_out,
        _ => x_mid,
    };
    let node_y = |layer: u8, id: u8| match layer {
        0 => input_y(id),
        2 => output_y(id),
        _ => inner_y(id),
    };

    // Edges first (under the nodes). Blue = positive, red = negative, fading
    // toward the baseline gray as |weight| -> 0; thickness scales with |weight|.
    for e in &g.edges {
        let (x0, y0) = (node_x(e.src_layer), node_y(e.src_layer, e.src_id));
        let (x1, y1) = (node_x(e.sink_layer), node_y(e.sink_layer, e.sink_id));
        let mag = (e.weight.abs().min(2.0) / 2.0).clamp(0.0, 1.0);
        let thick = 0.6 + mag * 2.2;
        let target = if e.weight >= 0.0 { SURVIVAL_HUE } else { EDGE_NEG };
        // lerp baseline(383835) -> target by mag, with alpha rising too.
        let col = Color::new(
            lerp(BASELINE.r, target.r, mag),
            lerp(BASELINE.g, target.g, mag),
            lerp(BASELINE.b, target.b, mag),
            0.22 + mag * 0.63,
        );
        draw_line(x0, y0, x1, y1, thick, col);
    }

    // Inner nodes (brightness by current activation).
    for &id in &g.inner_ids {
        let a = neurons.get(1).and_then(|l| l.get(id as usize)).copied().unwrap_or(0.0);
        draw_circle(x_mid, inner_y(id), 3.0, activation_color(a));
    }

    // Input nodes + labels (left).
    for id in 0..15u8 {
        let a = neurons.first().and_then(|l| l.get(id as usize)).copied().unwrap_or(0.0);
        let ny = input_y(id);
        draw_circle(x_in, ny, 4.0, activation_color(a));
        draw_text(INPUT_LABELS[id as usize], gx, ny + 4.0, 13.0, MUTED);
    }
    // Output nodes + labels (right).
    for id in 0..5u8 {
        let a = neurons.get(2).and_then(|l| l.get(id as usize)).copied().unwrap_or(0.0);
        let ny = output_y(id);
        draw_circle(x_out, ny, 5.0, activation_color(a));
        draw_text(OUTPUT_LABELS[id as usize], x_out + 10.0, ny + 4.0, 13.0, TEXT_SECONDARY);
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Node fill: baseline gray when quiet, ramping toward white as |activation|
/// grows (dark = quiet, bright = firing).
fn activation_color(a: f32) -> Color {
    let m = a.abs().clamp(0.0, 1.0);
    Color::new(
        lerp(BASELINE.r, 1.0, m),
        lerp(BASELINE.g, 1.0, m),
        lerp(BASELINE.b, 1.0, m),
        1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_evolution::{Cli, build_simulator};

    #[test]
    fn brain_graph_prunes_and_stays_under_cap() {
        // The active subgraph is a real subset of the genome's connections
        // (some connections can't reach an output), and the drawn set never
        // exceeds the edge cap.
        let cli = Cli::parse_from([
            "x", "--seed", "1", "--population", "100", "--genome-length", "256",
        ]);
        let mut sim = build_simulator(&cli);
        sim.generate_initial_generation();
        for _ in 0..50 {
            sim.step();
        }
        let i = best_agent(&sim).unwrap();
        let g = BrainGraph::build(&sim.agents[i]);
        assert!(g.edges.len() <= MAX_EDGES, "drawn edges exceed cap");
        assert!(
            g.active_total <= sim.agents[i].brain_connections().len(),
            "pruned set should not exceed the full connection set"
        );
    }

    #[test]
    fn agent_colors_land_in_the_bright_band() {
        // Both color modes route through the HSL constraint, so every produced
        // color is vivid on dark: not near-black, not near-white, with real
        // chroma. Checked across a spread of hues.
        for i in 0..64u32 {
            let hue = i as f32 / 64.0;
            let (r, g, b) = hsl_to_rgb(hue, 0.75, 0.62);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            assert!(max > 0.35, "hue {hue}: too dark (max {max})");
            assert!(min < 0.95, "hue {hue}: washed out (min {min})");
            assert!(max - min > 0.15, "hue {hue}: insufficient chroma");
        }
    }
}
