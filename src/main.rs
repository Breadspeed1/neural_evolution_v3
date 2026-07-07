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

    // rate-limiter accumulator (fractional steps carried between frames)
    let mut step_accum: f64 = 0.0;

    // persistent texture we update in place instead of recreating each frame
    let mut frame_image = Image::gen_image_color(128, 128, WHITE);
    let texture = Texture2D::from_image(&frame_image);
    texture.set_filter(FilterMode::Nearest);

    // --- dashboard state (all skipped while display is off) ---
    // Challenge-aware background overlay, cached per generation.
    let mut overlay: Vec<[u8; 4]> = vec![[255, 255, 255, 255]; 128 * 128];
    let mut overlay_gen: Option<u32> = None;
    let mut color_mode = ColorMode::Genome;
    // Selected agent for the brain inspector, tracked by (index, generation).
    let mut selected: Option<usize> = None;
    let mut selected_gen: u32 = 0;
    let mut graph: Option<BrainGraph> = None;

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
        if display && is_key_pressed(KeyCode::C) {
            color_mode = match color_mode {
                ColorMode::Genome => ColorMode::Lineage,
                ColorMode::Lineage => ColorMode::Genome,
            };
            println!("color mode: {}", color_mode.label());
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

            draw_world(simulator, &overlay, color_mode, selected, world_size, &mut frame_image, &texture);
            draw_dashboard(
                simulator,
                world_size,
                target_sps,
                unlimited,
                display,
                color_mode,
                selected,
                graph.as_ref(),
            );
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

/// Build the 128x128 challenge background: obstacle cells dark gray, survival
/// zone subtly green-tinted, everything else white. Pure function of the current
/// generation, so it is cached per generation by the caller.
fn build_overlay(sim: &Simulator, out: &mut [[u8; 4]]) {
    for y in 0..128u32 {
        for x in 0..128u32 {
            let c = if sim.is_safe((x, y)) {
                [225, 245, 225, 255]
            } else {
                [255, 255, 255, 255]
            };
            out[(y * 128 + x) as usize] = c;
        }
    }
    for &((x0, y0), (x1, y1)) in &sim.obstacles {
        for x in x0..=x1 {
            for y in y0..=y1 {
                out[(y * 128 + x) as usize] = [70, 70, 70, 255];
            }
        }
    }
}

/// Draw the world square (overlay + agents) at the top-left, plus the selection
/// ring, scaled up with nearest-neighbour filtering.
fn draw_world(
    sim: &Simulator,
    overlay: &[[u8; 4]],
    color_mode: ColorMode,
    selected: Option<usize>,
    world_size: f32,
    image: &mut Image,
    texture: &Texture2D,
) {
    let pixels = image.get_image_data_mut();
    pixels.copy_from_slice(overlay);
    for agent in &sim.agents {
        let (x, y) = agent.get_pos();
        pixels[(y * 128 + x) as usize] = agent_color(agent, color_mode);
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

    // Selection marker ring, drawn in screen space over the texture.
    if let Some(i) = selected
        && let Some(a) = sim.agents.get(i)
    {
        let (x, y) = a.get_pos();
        let sx = (x as f32 + 0.5) / 128.0 * world_size;
        let sy = (y as f32 + 0.5) / 128.0 * world_size;
        let r = (world_size / 128.0) * 3.0;
        draw_circle_lines(sx, sy, r, 2.0, Color::new(0.05, 0.05, 0.05, 0.9));
    }
}

/// The color an agent is drawn with under the active mode.
fn agent_color(agent: &Agent, mode: ColorMode) -> [u8; 4] {
    match mode {
        ColorMode::Genome => agent.get_rgba(),
        ColorMode::Lineage => lineage_color(agent.lineage),
    }
}

/// Map a lineage id to a bright, stable, well-spread color (golden-ratio hue).
fn lineage_color(lineage: u32) -> [u8; 4] {
    let hue = (lineage as f32 * 0.618_034).fract();
    let (r, g, b) = hsv_to_rgb(hue, 0.75, 0.95);
    [r, g, b, 255]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// ---------------------------------------------------------------------------
// Dashboard (right column)
// ---------------------------------------------------------------------------

const INK: Color = Color::new(0.12, 0.12, 0.14, 1.0);
const MUTED: Color = Color::new(0.45, 0.45, 0.48, 1.0);
const MARGIN: f32 = 14.0;

#[allow(clippy::too_many_arguments)]
fn draw_dashboard(
    sim: &Simulator,
    world_size: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    color_mode: ColorMode,
    selected: Option<usize>,
    graph: Option<&BrainGraph>,
) {
    // Faint divider between the world and the panel column.
    draw_line(world_size, 0.0, world_size, screen_height(), 1.0, Color::new(0.85, 0.85, 0.85, 1.0));

    let col_x = world_size + MARGIN;
    let col_w = (screen_width() - world_size - 2.0 * MARGIN).max(120.0);
    let mut y = MARGIN;

    let status_h = 168.0;
    draw_status_panel(sim, col_x, y, col_w, target_sps, unlimited, display, color_mode, selected);
    y += status_h + MARGIN;

    let spark_h = 118.0;
    let survival: Vec<f32> = sim
        .metrics_history()
        .iter()
        .map(|m| m.survival_rate as f32)
        .collect();
    draw_sparkline(col_x, y, col_w, spark_h, &survival, Some((0.0, 1.0)), "survival rate", true, "{:.0}%", 100.0);
    y += spark_h + MARGIN;

    let diversity: Vec<f32> = sim
        .metrics_history()
        .iter()
        .map(|m| m.genome_diversity as f32)
        .collect();
    draw_sparkline(col_x, y, col_w, spark_h, &diversity, None, "genome diversity", false, "{:.0}", 1.0);
    y += spark_h + MARGIN;

    let brain_h = (screen_height() - y - MARGIN).max(120.0);
    draw_brain_panel(sim, col_x, y, col_w, brain_h, selected, graph);
}

#[allow(clippy::too_many_arguments)]
fn draw_status_panel(
    sim: &Simulator,
    x: f32,
    y: f32,
    _w: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    color_mode: ColorMode,
    selected: Option<usize>,
) {
    draw_text("STATUS", x, y + 16.0, 22.0, INK);
    let pop = sim.agents.len();
    let live_safe = sim.agents.iter().filter(|a| sim.is_safe(a.get_pos())).count();
    let cur_pct = if pop > 0 { live_safe as f64 / pop as f64 * 100.0 } else { 0.0 };
    let last = sim.metrics_history().last();

    let speed = if unlimited {
        "unlimited".to_string()
    } else {
        format!("{:.0}/s", target_sps)
    };
    let last_str = match last {
        Some(m) => format!("gen {}: {} survived ({:.0}%)", m.generation, m.survivors, m.survival_rate * 100.0),
        None => "gen 0: (in progress)".to_string(),
    };
    let sel_str = match selected {
        Some(i) => format!("selected: agent #{i}"),
        None => "selected: none".to_string(),
    };

    let lines = [
        format!("generation {}", sim.generation),
        format!("challenge: {}", sim.challenge().name()),
        format!("population: {pop}"),
        last_str,
        format!("current survival: {:.1}%", cur_pct),
        format!("speed: {}   display: {}", speed, if display { "on" } else { "off" }),
        format!("color: {}   {}", color_mode.label(), sel_str),
    ];
    let mut ly = y + 44.0;
    for line in &lines {
        draw_text(line, x, ly, 18.0, INK);
        ly += 18.0;
    }
}

/// A minimal filled sparkline: muted single-hue fill under a thin line, faint
/// baseline, optional max marker, min/max value labels. `fixed` pins the y-axis
/// (e.g. 0..1 for a rate); otherwise it auto-scales. `fmt`/`scale` format the
/// value labels (value*scale).
#[allow(clippy::too_many_arguments)]
fn draw_sparkline(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    values: &[f32],
    fixed: Option<(f32, f32)>,
    title: &str,
    mark_max: bool,
    fmt: &str,
    scale: f32,
) {
    draw_text(title, x, y + 14.0, 18.0, INK);
    let plot_y = y + 24.0;
    let plot_h = h - 24.0;
    let baseline = plot_y + plot_h;

    if values.is_empty() {
        draw_line(x, baseline, x + w, baseline, 1.0, Color::new(0.85, 0.85, 0.85, 1.0));
        draw_text("(no data yet)", x, plot_y + plot_h * 0.5, 16.0, MUTED);
        return;
    }

    // Downsample to at most `w` points by bucket-mean so long runs still fit.
    let plot_w = w.max(1.0);
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

    let fill = Color::new(0.30, 0.52, 0.75, 0.18);
    let stroke = Color::new(0.20, 0.42, 0.68, 0.95);
    let n = pts.len();
    let dx = if n > 1 { plot_w / (n - 1) as f32 } else { 0.0 };

    // Fill under the curve.
    for (i, &v) in pts.iter().enumerate() {
        let px = x + i as f32 * dx;
        draw_line(px, map_y(v), px, baseline, dx.max(1.0), fill);
    }
    // Thin line on top.
    for i in 1..n {
        let x0 = x + (i - 1) as f32 * dx;
        let x1 = x + i as f32 * dx;
        draw_line(x0, map_y(pts[i - 1]), x1, map_y(pts[i]), 1.5, stroke);
    }
    // Faint baseline.
    draw_line(x, baseline, x + w, baseline, 1.0, Color::new(0.85, 0.85, 0.85, 1.0));

    // Max marker (over the *full* series, not the downsample).
    if mark_max
        && let Some((mi, &mv)) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        let frac = if values.len() > 1 { mi as f32 / (values.len() - 1) as f32 } else { 0.0 };
        let px = x + frac * plot_w;
        draw_circle(px, map_y(mv), 3.0, Color::new(0.85, 0.30, 0.25, 1.0));
    }

    // Min/max value labels (top-right / just above baseline).
    let hi_lbl = fmt.replace("{:.0}", &format!("{:.0}", hi * scale));
    let lo_lbl = fmt.replace("{:.0}", &format!("{:.0}", lo * scale));
    draw_text(&hi_lbl, x + w - 46.0, plot_y + 12.0, 15.0, MUTED);
    draw_text(&lo_lbl, x + w - 46.0, baseline - 2.0, 15.0, MUTED);
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
    draw_text("BRAIN INSPECTOR", x, y + 18.0, 22.0, INK);

    let (Some(idx), Some(g)) = (selected, graph) else {
        draw_text("(no agent selected)", x, y + 48.0, 16.0, MUTED);
        return;
    };
    let Some(agent) = sim.agents.get(idx) else {
        draw_text("(selection lost)", x, y + 48.0, 16.0, MUTED);
        return;
    };

    let subtitle = if g.active_total > g.edges.len() {
        format!("agent #{idx}  (lineage {})   showing {}/{} edges", agent.lineage, g.edges.len(), g.active_total)
    } else {
        format!("agent #{idx}  (lineage {})   {} edges, {} inner", agent.lineage, g.edges.len(), g.inner_ids.len())
    };
    draw_text(&subtitle, x, y + 40.0, 15.0, MUTED);

    // Plot area for the node-link diagram.
    let gx = x + 8.0;
    let gy = y + 54.0;
    let gw = w - 16.0;
    let gh = h - 62.0;
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

    // Edges first (under the nodes). Blue = positive, red = negative; thickness
    // and alpha scale with |weight|.
    for e in &g.edges {
        let (x0, y0) = (node_x(e.src_layer), node_y(e.src_layer, e.src_id));
        let (x1, y1) = (node_x(e.sink_layer), node_y(e.sink_layer, e.sink_id));
        let mag = e.weight.abs().min(2.0) / 2.0;
        let thick = 0.6 + mag * 2.2;
        let alpha = 0.20 + mag * 0.65;
        let col = if e.weight >= 0.0 {
            Color::new(0.20, 0.45, 0.85, alpha)
        } else {
            Color::new(0.85, 0.28, 0.25, alpha)
        };
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
        draw_text(OUTPUT_LABELS[id as usize], x_out + 10.0, ny + 4.0, 13.0, INK);
    }
}

/// Node fill: brightness scales with |activation| (dark = quiet, bright = firing).
fn activation_color(a: f32) -> Color {
    let m = a.abs().clamp(0.0, 1.0);
    let v = 0.15 + 0.75 * m;
    Color::new(v, v, v * 0.9 + 0.1 * m, 1.0)
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
}
