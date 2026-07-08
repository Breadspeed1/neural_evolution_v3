use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use clap::Parser;
use hecs::Entity;
use macroquad::prelude::*;
use neural_evolution::agent::{BrainKind, Connection};
use neural_evolution::eco::{CreatureView, EcoMetrics, EcoSim, Species};
use neural_evolution::grid::Cell;
use neural_evolution::{AgentView, Cli, Mode, Simulator, build_eco_sim, build_simulator};

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
    match cli.mode {
        Mode::Challenge => {
            let mut simulator = build_simulator(&cli);
            let generations = cli.generations.unwrap_or(u32::MAX);
            run_sim(&mut simulator, generations).await;
        }
        Mode::Eco => {
            let mut eco = build_eco_sim(&cli);
            let ticks = cli.ticks.unwrap_or(u64::MAX);
            run_eco(&mut eco, ticks).await;
        }
    }
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

// Eco-mode diorama tones. Nutrient is a subtle warm soil wash; biomass is a
// green intensity. Both are blended over the dark world background per cell.
const SOIL_RGB: [u8; 3] = [104, 74, 48]; // warm earthy brown
const PLANT_RGB: [u8; 3] = [76, 196, 96]; // living green
const PLANT_HUE: Color = Color::new(76.0 / 255.0, 196.0 / 255.0, 96.0 / 255.0, 1.0);
const NUTRIENT_HUE: Color = Color::new(176.0 / 255.0, 138.0 / 255.0, 96.0 / 255.0, 1.0);
// Herbivore accent (a warm amber, distinct from the plant/soil greens & browns)
// for the population sparkline and the creatures legend swatch.
const HERB_HUE: Color = Color::new(232.0 / 255.0, 176.0 / 255.0, 64.0 / 255.0, 1.0);
// Predator accent (a hot red-orange — the apex tier) for the predator population
// sparkline, status, and legend swatch. Deliberately hotter than the amber herb
// hue and the green plant hue so the three trophic levels read apart at a glance.
const PRED_HUE: Color = Color::new(233.0 / 255.0, 96.0 / 255.0, 58.0 / 255.0, 1.0);

const MARGIN: f32 = 14.0;
/// How many recent positions each agent's motion trail retains.
const TRAIL_LEN: usize = 18;
/// Trail length for the eco spotlight protagonist — longer than the challenge
/// crowd trails, since only the one selected grazer carries a trail here, so it
/// can afford a strong, far-reaching comet tail.
const ECO_TRAIL_LEN: usize = 44;
/// In unlimited mode, record a trail sample at most every this-many sim steps,
/// bounding the per-frame recording cost when thousands of steps run per frame
/// (visual continuity is a normal-speed concern, not an unlimited-speed one).
const UNLIMITED_TRAIL_STRIDE: u32 = 4;
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
            // Only sample trails when the world is shown, and only every kth
            // step: unlimited mode runs thousands of steps per frame, so
            // recording every one would be wasteful.
            let mut since_record: u32 = 0;
            loop {
                for _ in 0..64 {
                    simulator.step();
                    steps_this_frame += 1;
                    if display {
                        since_record += 1;
                        if since_record >= UNLIMITED_TRAIL_STRIDE {
                            since_record = 0;
                            record_trails(&mut trails, &simulator.positions());
                        }
                    }
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
                // One trail sample per executed sim step (not per frame), so
                // trails stay continuous when several steps run per frame.
                record_trails(&mut trails, &simulator.positions());
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
            // One render snapshot per frame (position + colors), in `order`
            // sequence; every selection index and per-agent draw reads from it.
            let views = simulator.agent_views();

            // (Re)build the challenge overlay only when the generation changes.
            if overlay_gen != Some(simulator.generation) {
                build_overlay(simulator, &mut overlay);
                overlay_gen = Some(simulator.generation);
                // Agents were replaced on turnover: reselect via the heuristic.
                let idx = best_agent(simulator, &views);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }
            if selected.is_none() {
                let idx = best_agent(simulator, &views);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }

            // Mouse click selects the nearest agent within a few cells.
            if is_mouse_button_pressed(MouseButton::Left) {
                let (mx, my) = mouse_position();
                if mx < world_size && my < world_size {
                    let cx = (mx / world_size * 128.0) as i32;
                    let cy = (my / world_size * 128.0) as i32;
                    if let Some(idx) = nearest_agent(&views, cx, cy, 4) {
                        set_selection(simulator, Some(idx), &mut selected, &mut selected_gen, &mut graph);
                    }
                }
            }
            // `b` reselects the heuristic "best" agent.
            if is_key_pressed(KeyCode::B) {
                let idx = best_agent(simulator, &views);
                set_selection(simulator, idx, &mut selected, &mut selected_gen, &mut graph);
            }

            // Heatmap occupancy sample (trails are recorded per sim step, in the
            // stepping loop above, so they stay continuous at multiple steps per
            // frame).
            if heatmap {
                accumulate_heat(&mut heat, &simulator.positions());
            }

            draw_world(
                &views, &overlay, &heat, heatmap, color_mode, selected, &trails, pulse.as_ref(),
                world_size, &mut frame_image, &texture,
            );
            draw_dashboard(
                simulator, &views, world_size, target_sps, unlimited, display, heatmap, color_mode,
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

// ===========================================================================
// Eco mode (rung 1): the living meadow. A layered diorama on the shared dark
// dashboard shell — nutrient as a warm soil wash, plant biomass as green — with
// biomass/coverage sparklines. No agents yet; q/e speed and v display still work.
// ===========================================================================

/// The eco-mode dashboard loop. Advances the continuous terrarium (no
/// generations — ticks run continuously) and renders the meadow each frame.
async fn run_eco(eco: &mut EcoSim, ticks: u64) {
    // Coarsen metrics recording so a long viewer session's history stays bounded
    // (the headless runner keeps the default per-tick recording for its curve).
    eco.set_metrics_interval(10);
    eco.seed_initial();

    let (gw, gh) = (eco.width(), eco.height());
    let mut target_sps: f64 = 240.0;
    let mut unlimited = false;
    let mut display = true;
    let mut step_accum: f64 = 0.0;

    // --- storytelling state (viewer-side only) ---
    // Spotlight mode: dim the crowd to a wash, elevate one selected protagonist.
    let mut spotlight = false;
    // The tracked protagonist, by stable hecs entity (its `order` index shifts as
    // neighbours die). Follows the creature across ticks; on its death we
    // auto-reselect via the `b` heuristic.
    let mut selected: Option<Entity> = None;
    // The selected grazer's pruned signal-flow brain, rebuilt only on reselection.
    let mut graph: Option<BrainGraph> = None;
    // The protagonist's motion trail (world cells, newest last), sampled per frame.
    let mut trail: Vec<(u32, u32)> = Vec::new();

    // Persistent texture sized to the grid, updated in place each frame.
    let mut frame_image = Image::gen_image_color(gw as u16, gh as u16, WHITE);
    let texture = Texture2D::from_image(&frame_image);
    texture.set_filter(FilterMode::Nearest);

    let mut last_frame = Instant::now();
    let mut last_report = Instant::now();
    let mut steps_since_report: u64 = 0;
    let mut fps_smooth = 60.0;
    let mut sps_smooth = 0.0;

    // Optional smoke-test screenshot: grab one frame once the meadow has grown in.
    // `NEURAL_SHOT_TICK` overrides when (default 2000); `NEURAL_SPOTLIGHT` starts
    // in spotlight mode so the automated grab can capture the protagonist view.
    let screenshot_path = std::env::var("NEURAL_SCREENSHOT").ok();
    let shot_tick: u64 = std::env::var("NEURAL_SHOT_TICK").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    if std::env::var("NEURAL_SPOTLIGHT").is_ok() {
        spotlight = true;
    }
    let mut shot = false;
    // For a clean screenshot, fast-forward (unrendered) to just shy of the target
    // tick, so the final stretch renders at normal speed and the trail is smooth.
    if screenshot_path.is_some() {
        while eco.tick_count() + 250 < shot_tick {
            eco.tick();
        }
    }

    while eco.tick_count() < ticks {
        // --- input (speed / display, plus the eco storytelling controls) ---
        if is_key_pressed(KeyCode::Q) {
            if unlimited {
                unlimited = false;
                target_sps = UNLIMITED_THRESHOLD;
            }
            target_sps = (target_sps * 0.5).max(MIN_SPS);
            println!("target {:.0} ticks/sec", target_sps);
        }
        if is_key_pressed(KeyCode::E) {
            target_sps *= 2.0;
            if target_sps >= UNLIMITED_THRESHOLD {
                unlimited = true;
                println!("target ticks/sec: unlimited");
            } else {
                println!("target {:.0} ticks/sec", target_sps);
            }
        }
        if is_key_pressed(KeyCode::V) {
            display = !display;
            println!("display {}", if display { "on" } else { "off" });
        }
        if display && is_key_pressed(KeyCode::F) {
            spotlight = !spotlight;
            println!("spotlight {}", if spotlight { "on" } else { "off" });
        }

        // --- advance the meadow for this frame ---
        let frame_start = Instant::now();
        let dt = frame_start.duration_since(last_frame).as_secs_f64();
        last_frame = frame_start;

        let mut steps_this_frame: u64 = 0;
        if !display || unlimited {
            let budget = if display { DISPLAY_BUDGET } else { HEADLESS_BUDGET };
            loop {
                for _ in 0..64 {
                    eco.tick();
                    steps_this_frame += 1;
                }
                if frame_start.elapsed() >= budget {
                    break;
                }
            }
        } else {
            step_accum += target_sps * dt;
            let cap = (target_sps * 0.25).ceil().max(1.0);
            let n = step_accum.floor().min(cap);
            step_accum -= n;
            for _ in 0..(n as u64) {
                eco.tick();
                steps_this_frame += 1;
            }
        }
        steps_since_report += steps_this_frame;

        if dt > 0.0 {
            fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1;
            sps_smooth = sps_smooth * 0.9 + (steps_this_frame as f64 / dt) * 0.1;
        }
        if last_report.elapsed() >= Duration::from_secs(2) {
            let secs = last_report.elapsed().as_secs_f64();
            println!("~{:.0} ticks/sec", steps_since_report as f64 / secs);
            steps_since_report = 0;
            last_report = Instant::now();
        }

        // --- draw ---
        clear_background(BG_PAGE);
        if display {
            let world_px = screen_height();
            // One creature snapshot per frame (both species); selection + drawing
            // read from it.
            let views = eco.creature_views();
            let top_lineage = eco.latest().and_then(|m| m.lineage_counts.first().map(|&(l, _)| l));

            // Resolve the selection: drop it if the protagonist died, then
            // (re)select the heuristic best so a brain is always on show.
            if let Some(e) = selected
                && !views.iter().any(|v| v.entity == e)
            {
                selected = None;
            }
            if selected.is_none() {
                set_creature_selection(eco, best_herb(&views, top_lineage), &mut selected, &mut graph, &mut trail);
            }

            // Click selects the nearest creature (either species); `b` reselects
            // the best grazer, `p` the fattest predator (to inspect a hunter).
            if is_mouse_button_pressed(MouseButton::Left) {
                let (mx, my) = mouse_position();
                let cell = world_px / eco.width().max(1) as f32;
                if mx < world_px && my < world_px && cell > 0.0 {
                    let (cx, cy) = ((mx / cell) as i32, (my / cell) as i32);
                    if let Some(e) = nearest_creature(&views, cx, cy, 5) {
                        set_creature_selection(eco, Some(e), &mut selected, &mut graph, &mut trail);
                    }
                }
            }
            if is_key_pressed(KeyCode::B) {
                set_creature_selection(eco, best_herb(&views, top_lineage), &mut selected, &mut graph, &mut trail);
            }
            if is_key_pressed(KeyCode::P) {
                set_creature_selection(eco, best_predator(&views), &mut selected, &mut graph, &mut trail);
            }

            // Track the protagonist's path (skip consecutive duplicates so a
            // stationary grazer keeps a short trail, not a stack of one point).
            let sel_view = selected.and_then(|e| views.iter().find(|v| v.entity == e));
            if let Some(v) = sel_view {
                if trail.last() != Some(&v.pos) {
                    trail.push(v.pos);
                }
                if trail.len() > ECO_TRAIL_LEN {
                    trail.remove(0);
                }
            }

            draw_eco_world(eco, &views, spotlight, sel_view, &trail, world_px, &mut frame_image, &texture);
            draw_eco_dashboard(
                eco, sel_view, graph.as_ref(), spotlight, world_px, target_sps, unlimited, display,
                fps_smooth, sps_smooth,
            );
        }

        if let Some(path) = &screenshot_path
            && !shot
            && eco.tick_count() >= shot_tick
        {
            // Grab once the meadow has passed its pioneer bloom and settled into
            // the plant↔herbivore coexistence, so the shot shows the steady state.
            get_screen_data().export_png(path);
            println!("screenshot saved to {path}");
            shot = true;
        }
        next_frame().await;
    }
}

/// Assign the eco selection and rebuild the (cached) signal-flow brain for it,
/// resetting the protagonist's trail. The wiring is copied out of the ECS; the
/// tighter [`CREATURE_MAX_EDGES`] cap keeps the circuit legible. Works for either
/// species — the brain panel picks the sensor-row labels from the selection's
/// species at draw time.
fn set_creature_selection(
    eco: &EcoSim,
    sel: Option<Entity>,
    selected: &mut Option<Entity>,
    graph: &mut Option<BrainGraph>,
    trail: &mut Vec<(u32, u32)>,
) {
    *selected = sel;
    *graph = sel
        .and_then(|e| eco.creature_connections(e))
        .map(|c| BrainGraph::build_capped(&c, CREATURE_MAX_EDGES));
    trail.clear();
}

/// The auto-selected protagonist: the highest-energy **herbivore** of the current
/// largest lineage (the reigning dynasty's fittest grazer), falling back to the
/// globally fattest grazer before the first dynasty record exists. Selecting
/// within the dominant bloodline ties the spotlight to the bloodlines strip — the
/// ringed creature's hue matches the strip's widest band — and energy picks a
/// thriving, long-lived individual, so the spotlight stays populated across ticks.
fn best_herb(views: &[CreatureView], top_lineage: Option<u32>) -> Option<Entity> {
    let fattest = |pool: &mut dyn Iterator<Item = &CreatureView>| {
        pool.max_by(|a, b| {
            a.energy_frac.partial_cmp(&b.energy_frac).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|v| v.entity)
    };
    let herbs = |v: &&CreatureView| v.species == Species::Herbivore;
    if let Some(l) = top_lineage
        && let Some(e) = fattest(&mut views.iter().filter(herbs).filter(|v| v.lineage == l))
    {
        return Some(e);
    }
    fattest(&mut views.iter().filter(herbs))
}

/// The fattest predator, for the `p` "inspect a hunter" pick. `None` if there are
/// no predators (then the selection is left as-is).
fn best_predator(views: &[CreatureView]) -> Option<Entity> {
    views
        .iter()
        .filter(|v| v.species == Species::Predator)
        .max_by(|a, b| {
            a.energy_frac.partial_cmp(&b.energy_frac).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|v| v.entity)
}

/// Nearest creature (either species) to cell (cx, cy) within `radius` cells
/// (squared-Euclidean), if any — the click-to-select pick. Returns the stable
/// entity, not an index.
fn nearest_creature(views: &[CreatureView], cx: i32, cy: i32, radius: i32) -> Option<Entity> {
    let mut best: Option<(i32, Entity)> = None;
    for v in views {
        let (ax, ay) = (v.pos.0 as i32, v.pos.1 as i32);
        let d2 = (ax - cx).pow(2) + (ay - cy).pow(2);
        if d2 <= radius * radius && best.is_none_or(|(bd, _)| d2 < bd) {
            best = Some((d2, v.entity));
        }
    }
    best.map(|(_, e)| e)
}

/// Composite the meadow into the grid-sized texture (a nutrient wash under plant
/// green, obstacles solid) and blit it into the world square, then draw the
/// grazers over it. Two moods: the **calm baseline** (`spotlight` off) draws the
/// whole crowd as a living texture — size and brightness both rising with energy,
/// so fat and starving read at a glance; **spotlight** (`spotlight` on) dims the
/// crowd to a faint wash and elevates the one selected protagonist. In either
/// mood the selected grazer gets a full-bright body, a bright selection ring, and
/// a fading comet-tail trail.
#[allow(clippy::too_many_arguments)]
fn draw_eco_world(
    eco: &EcoSim,
    views: &[CreatureView],
    spotlight: bool,
    sel_view: Option<&CreatureView>,
    trail: &[(u32, u32)],
    world_px: f32,
    image: &mut Image,
    texture: &Texture2D,
) {
    let p = eco.params();
    let (soil_cap, biomass_max) = (p.soil_cap.max(1e-6), p.biomass_max.max(1e-6));
    let pixels = image.get_image_data_mut();
    for (px, cell) in pixels.iter_mut().zip(eco.cells()) {
        *px = eco_pixel(cell, soil_cap, biomass_max);
    }
    texture.update(image);
    draw_texture_ex(
        texture,
        0.0,
        0.0,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(world_px, world_px)),
            ..Default::default()
        },
    );

    let cell = world_px / eco.width().max(1) as f32;
    let center = |p: (u32, u32)| ((p.0 as f32 + 0.5) * cell, (p.1 as f32 + 0.5) * cell);
    let sel_entity = sel_view.map(|v| v.entity);

    if spotlight {
        // Dim the whole crowd to a faint wash — a quiet field the protagonist
        // stands against. Grazers are faint dots; predators stay faint warm
        // chevrons (still legible as the rarer apex tier), skipping the selected.
        let r = (cell * 0.42).max(1.1);
        for c in views {
            if Some(c.entity) == sel_entity {
                continue;
            }
            let (cx, cy) = center(c.pos);
            match c.species {
                Species::Herbivore => {
                    let (cr, cg, cb) = hsl_to_rgb(lineage_hue(c.lineage), 0.55, 0.42);
                    draw_circle(cx, cy, r, Color::new(cr, cg, cb, 0.16));
                }
                Species::Predator => {
                    let s = (cell * 0.7).max(2.0);
                    draw_predator_mark(cx, cy, s, predator_color(c.lineage, c.energy_frac, 0.34), false);
                }
            }
        }
    } else {
        // Calm baseline. Grazers first: a living texture whose radius *and*
        // brightness rise with energy (fat vs starving at a glance), a thin dark
        // ring keeping lineage color legible over the green — a breathing meadow.
        for c in views.iter().filter(|c| c.species == Species::Herbivore) {
            let (cx, cy) = center(c.pos);
            let r = (cell * (0.24 + 0.32 * c.energy_frac)).max(1.2);
            let (cr, cg, cb) = hsl_to_rgb(lineage_hue(c.lineage), 0.70, 0.38 + 0.26 * c.energy_frac);
            draw_circle(cx, cy, r + 0.4, Color::new(0.04, 0.05, 0.03, 0.42));
            draw_circle(cx, cy, r, Color::new(cr, cg, cb, 0.82));
        }
        // Predators over the grazers: the apex tier, drawn as larger warm chevrons
        // (a fang/arrowhead vs the grazers' dots) so a hunter is unmistakable at a
        // glance. Size ∝ energy; a dark edge keeps them crisp over the meadow.
        for c in views.iter().filter(|c| c.species == Species::Predator) {
            let (cx, cy) = center(c.pos);
            let s = (cell * (0.55 + 0.4 * c.energy_frac)).max(2.2);
            draw_predator_mark(cx, cy, s, predator_color(c.lineage, c.energy_frac, 0.95), true);
        }
    }

    // The protagonist's comet tail: newer segments brighter and thicker. Warm for
    // a predator, its lineage hue for a grazer. Reads strongest against the dimmed
    // spotlight field, but drawn in both moods.
    if trail.len() >= 2 {
        let (tr, tg, tb) = match sel_view {
            Some(v) if v.species == Species::Predator => {
                let c = predator_color(v.lineage, 1.0, 1.0);
                (c.r, c.g, c.b)
            }
            Some(v) => hsl_to_rgb(lineage_hue(v.lineage), 0.85, 0.62),
            None => hsl_to_rgb(0.12, 0.85, 0.62),
        };
        let n = trail.len();
        for k in 1..n {
            let f = k as f32 / n as f32; // 0 (old) .. 1 (new)
            let (x0, y0) = center(trail[k - 1]);
            let (x1, y1) = center(trail[k]);
            draw_line(x0, y0, x1, y1, 1.0 + 2.4 * f, Color::new(tr, tg, tb, 0.12 + 0.66 * f));
        }
    }

    // The protagonist itself: full-bright body (species-shaped) + a bright
    // selection ring, so the one selected creature stands out in either mood.
    if let Some(v) = sel_view {
        let (cx, cy) = center(v.pos);
        match v.species {
            Species::Herbivore => {
                let r = (cell * 0.62).max(2.2);
                let (cr, cg, cb) = hsl_to_rgb(lineage_hue(v.lineage), 0.90, 0.42 + 0.26 * v.energy_frac);
                draw_circle(cx, cy, r + 1.2, Color::new(0.02, 0.03, 0.02, 0.9));
                draw_circle(cx, cy, r, Color::new(cr, cg, cb, 1.0));
            }
            Species::Predator => {
                let s = (cell * 0.95).max(3.2);
                draw_predator_mark(cx, cy, s, predator_color(v.lineage, v.energy_frac, 1.0), true);
            }
        }
        let ring = (cell * 2.4).max(9.0);
        draw_circle_lines(cx, cy, ring, 2.0, Color::new(1.0, 1.0, 1.0, 0.95));
        draw_circle_lines(cx, cy, ring + 2.0, 1.0, Color::new(1.0, 1.0, 1.0, 0.32));
    }
}

/// The warm apex palette for a predator: a red→orange band that reads as a
/// clearly different, hotter species than the grazers' full-spectrum lineage
/// hues, still lineage-varied (a small hue jitter) and energy-brightened.
fn predator_color(lineage: u32, energy_frac: f32, alpha: f32) -> Color {
    let hue = 0.015 + 0.055 * lineage_hue(lineage); // ~0.015..0.07 (red → orange)
    let (r, g, b) = hsl_to_rgb(hue, 0.88, 0.42 + 0.20 * energy_frac);
    Color::new(r, g, b, alpha)
}

/// Draw a predator's apex marker — an upward chevron (a fang/arrowhead) that
/// reads as a distinct, larger species over the round grazer dots. `s` is the
/// half-height; `outline` adds a dark edge so it stays legible over the meadow.
fn draw_predator_mark(cx: f32, cy: f32, s: f32, col: Color, outline: bool) {
    let tip = vec2(cx, cy - s);
    let bl = vec2(cx - s * 0.92, cy + s * 0.70);
    let br = vec2(cx + s * 0.92, cy + s * 0.70);
    draw_triangle(tip, bl, br, col);
    if outline {
        draw_triangle_lines(tip, bl, br, 1.5, Color::new(0.06, 0.02, 0.02, 0.85));
    }
}

/// One cell's diorama pixel: obstacles solid, otherwise the dark world with a
/// subtle warm soil wash (∝ nutrient) under a green plant layer (∝ √biomass, so
/// sparse seedlings still read).
fn eco_pixel(cell: &Cell, soil_cap: f32, biomass_max: f32) -> [u8; 4] {
    if cell.obstacle {
        return OBSTACLE_PX;
    }
    let mut px = WORLD_BG_PX;
    let na = (cell.nutrient / soil_cap).clamp(0.0, 1.0) * 0.30;
    px = blend_px(px, SOIL_RGB, na);
    let bt = (cell.biomass / biomass_max).clamp(0.0, 1.0).sqrt();
    blend_px(px, PLANT_RGB, 0.92 * bt)
}

/// The eco dashboard column: status, the predator-prey sparklines, the dynasty
/// **bloodlines strip**, and the selected grazer's **signal-flow brain**, with
/// the keybind strip pinned to the bottom. Reuses the shared panel helpers.
#[allow(clippy::too_many_arguments)]
fn draw_eco_dashboard(
    eco: &EcoSim,
    sel_view: Option<&CreatureView>,
    graph: Option<&BrainGraph>,
    spotlight: bool,
    world_px: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    fps: f64,
    sps: f64,
) {
    draw_line(world_px, 0.0, world_px, screen_height(), 1.0, HAIRLINE);

    let col_x = world_px + MARGIN;
    let col_w = (screen_width() - world_px - 2.0 * MARGIN).max(120.0);
    let mut y = MARGIN;

    // Corner HUD over the world.
    let hud = format!("{:.0} fps   {:.0} ticks/s", fps, sps);
    let dim = measure_text(&hud, None, 15, 1.0);
    draw_text(&hud, world_px - dim.width - 12.0, 20.0, 15.0, MUTED);

    // Keybind strip pinned to the bottom (computed first so the panels above can
    // fill the remaining height). `f` toggles spotlight, `b`/`p` reselect the best
    // grazer / predator, click picks the nearest creature of either species.
    let legend = "q/e speed · f spotlight · b grazer · p predator · click select · v display";
    let legend_h = 26.0;
    let legend_y = screen_height() - MARGIN - legend_h;
    draw_text(legend, col_x, legend_y + 17.0, 14.0, MUTED);

    let status_h = 234.0;
    draw_eco_status_panel(eco, col_x, y, col_w, status_h, target_sps, unlimited, display, spotlight);
    y += status_h + MARGIN;

    // The tri-trophic readout: plant coverage, then the two mobile levels
    // (herbivores in amber, predators in hot red) as the coupled pair whose lagged
    // oscillation is the whole point, then total plant biomass. Four stacked
    // sparklines = soil→plants→herbivores→predators at a glance.
    let spark_h = 84.0;
    let coverage: Vec<f32> = eco.metrics_history().iter().map(|m| m.coverage as f32).collect();
    draw_sparkline(col_x, y, col_w, spark_h, &coverage, Some((0.0, 1.0)), "plant coverage", PLANT_HUE, true, "{:.0}%", 100.0);
    y += spark_h + MARGIN;

    let population: Vec<f32> = eco.metrics_history().iter().map(|m| m.population as f32).collect();
    draw_sparkline(col_x, y, col_w, spark_h, &population, None, "herbivores", HERB_HUE, true, "{:.0}", 1.0);
    y += spark_h + MARGIN;

    let predators: Vec<f32> = eco.metrics_history().iter().map(|m| m.predators as f32).collect();
    draw_sparkline(col_x, y, col_w, spark_h, &predators, None, "predators", PRED_HUE, true, "{:.0}", 1.0);
    y += spark_h + MARGIN;

    let biomass: Vec<f32> = eco.metrics_history().iter().map(|m| m.total_biomass as f32).collect();
    draw_sparkline(col_x, y, col_w, spark_h, &biomass, None, "biomass", NUTRIENT_HUE, false, "{:.0}", 1.0);
    y += spark_h + MARGIN;

    // Bloodlines strip: herbivore dynasty population share over time — the "watch
    // evolution happen" panel (selective sweeps, blooms, crashes).
    let dynasty_h = 130.0;
    draw_dynasty_strip(col_x, y, col_w, dynasty_h, eco.metrics_history());
    y += dynasty_h + MARGIN;

    // The selected creature's live signal-flow brain fills the rest of the column.
    let brain_h = (legend_y - MARGIN - y).max(120.0);
    draw_eco_brain_panel(eco, col_x, y, col_w, brain_h, sel_view, graph);
}

#[allow(clippy::too_many_arguments)]
fn draw_eco_status_panel(
    eco: &EcoSim,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    target_sps: f64,
    unlimited: bool,
    display: bool,
    spotlight: bool,
) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;
    draw_header("status", x + pad, y + 22.0);

    let m = eco.latest();
    let coverage = m.map(|m| m.coverage).unwrap_or(0.0);
    let biomass = m.map(|m| m.total_biomass).unwrap_or(0.0);
    let population = m.map(|m| m.population).unwrap_or(0);
    let mean_energy = m.map(|m| m.mean_energy).unwrap_or(0.0);
    let births = m.map(|m| m.births).unwrap_or(0);
    let deaths = m.map(|m| m.deaths).unwrap_or(0);
    let predators = m.map(|m| m.predators).unwrap_or(0);
    let pred_energy = m.map(|m| m.pred_mean_energy).unwrap_or(0.0);
    let pred_births = m.map(|m| m.pred_births).unwrap_or(0);
    let pred_deaths = m.map(|m| m.pred_deaths).unwrap_or(0);
    let ctrnn_herb = m.map(|m| m.ctrnn_population).unwrap_or(0);
    let ctrnn_pred = m.map(|m| m.ctrnn_predators).unwrap_or(0);
    let speed = if unlimited { "unlimited".to_string() } else { format!("{:.0}/s", target_sps) };

    // Brain-type readout: the run's brain dynamics, and for a mixed run the live
    // feed-forward vs CTRNN herbivore population shares — the A/B competition made
    // watchable (feed-forward has no CTRNN creatures, so it just names the mode).
    let ff_herb = population.saturating_sub(ctrnn_herb);
    let ff_pred = predators.saturating_sub(ctrnn_pred);
    let brains_val = if population == 0 && predators == 0 {
        "—".to_string()
    } else if ctrnn_herb == 0 && ctrnn_pred == 0 {
        "feed-forward".to_string()
    } else if ff_herb == 0 && ff_pred == 0 {
        "ctrnn (recurrent)".to_string()
    } else {
        let ct_pct = ctrnn_herb as f64 * 100.0 / population.max(1) as f64;
        format!("mixed  ·  FF {:.0}%   CTRNN {:.0}%", 100.0 - ct_pct, ct_pct)
    };

    let label_x = x + pad;
    let value_x = x + pad + 118.0;
    let row_h = 20.0;
    let mut ly = y + 46.0;
    let row = |label: &str, value: &str, vcol: Color, ly: &mut f32| {
        draw_text(label, label_x, *ly, 17.0, MUTED);
        draw_text(value, value_x, *ly, 17.0, vcol);
        *ly += row_h;
    };
    row("mode", "eco  ·  terrarium (rung 3)", TEXT_SECONDARY, &mut ly);
    row("tick", &eco.tick_count().to_string(), TEXT_PRIMARY, &mut ly);
    row("plants", &format!("{:.1}% cover  ·  {biomass:.0} mass", coverage * 100.0), TEXT_SECONDARY, &mut ly);
    row("herbivores", &format!("{population}  ·  mean-E {mean_energy:.2}"), TEXT_PRIMARY, &mut ly);
    row("  flux", &format!("+{births} births  ·  -{deaths} deaths"), MUTED, &mut ly);
    row("predators", &format!("{predators}  ·  mean-E {pred_energy:.2}"), PRED_HUE, &mut ly);
    row("  flux", &format!("+{pred_births} births  ·  -{pred_deaths} deaths"), MUTED, &mut ly);
    row("brains", &brains_val, TEXT_SECONDARY, &mut ly);
    row(
        "view",
        &format!("{speed}   display {}   spotlight {}", if display { "on" } else { "off" }, if spotlight { "on" } else { "off" }),
        TEXT_SECONDARY,
        &mut ly,
    );
}

/// How many distinct lineages the bloodlines strip tracks as their own colored
/// band (the top-N by peak population share across the visible history); every
/// other lineage folds into the muted "other" band. Kept below the record's
/// [`TOP_LINEAGES`] cap so the bands stay hue-distinct.
const DYNASTY_BANDS: usize = 10;

/// The bloodlines strip: a stacked-area chart of **lineage population share over
/// time**, driven by the committed per-interval dynasty record
/// ([`EcoMetrics::lineage_counts`]). Each colored band is one dynasty, hued by
/// the same `lineage_hue` mapping its grazers use on the meadow (so a band and
/// its creatures match), stacked over a muted "other" for the long tail. Watch
/// dynasties bloom, dominate, and crash — a live selective sweep.
fn draw_dynasty_strip(x: f32, y: f32, w: f32, h: f32, history: &[EcoMetrics]) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;
    draw_header("bloodlines", x + pad, y + 20.0);

    let plot_x = x + pad;
    let plot_w = (w - 2.0 * pad).max(1.0);
    let plot_y = y + 30.0;
    let plot_h = (h - 40.0).max(1.0);
    let baseline = plot_y + plot_h;

    if history.len() < 2 {
        draw_line(plot_x, baseline, plot_x + plot_w, baseline, 1.0, BASELINE);
        draw_text("(gathering lineages…)", plot_x, plot_y + plot_h * 0.5, 15.0, MUTED);
        return;
    }

    // Tracked dynasties: the top-N lineages by peak share across the history.
    // A lineage that ever rose to prominence keeps its band (so crashes show),
    // and the set is stable frame to frame, so bands don't flicker.
    let mut peak: BTreeMap<u32, f32> = BTreeMap::new();
    for m in history {
        if m.population == 0 {
            continue;
        }
        let pop = m.population as f32;
        for &(l, c) in &m.lineage_counts {
            let s = c as f32 / pop;
            let e = peak.entry(l).or_insert(0.0);
            if s > *e {
                *e = s;
            }
        }
    }
    let mut ranked: Vec<(u32, f32)> = peak.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    ranked.truncate(DYNASTY_BANDS);
    let mut tracked: Vec<u32> = ranked.into_iter().map(|(l, _)| l).collect();
    tracked.sort_unstable(); // stable stacking slots, ordered by lineage id

    // Per-record share vector: one entry per tracked lineage, then "other" (the
    // remainder), so every column sums to exactly 1.0.
    let bands = tracked.len() + 1;
    let shares: Vec<Vec<f32>> = history
        .iter()
        .map(|m| {
            let mut row = vec![0.0f32; bands];
            if m.population > 0 {
                let pop = m.population as f32;
                let mut sum = 0.0;
                for (i, &l) in tracked.iter().enumerate() {
                    if let Some(&(_, c)) = m.lineage_counts.iter().find(|&&(ll, _)| ll == l) {
                        let s = c as f32 / pop;
                        row[i] = s;
                        sum += s;
                    }
                }
                row[bands - 1] = (1.0 - sum).max(0.0);
            } else {
                row[bands - 1] = 1.0;
            }
            row
        })
        .collect();

    // Downsample records to at most plot-width columns (bucket-mean per band).
    let cols = (plot_w as usize).clamp(2, shares.len());
    let col_share = |j: usize| -> Vec<f32> {
        let a = j * shares.len() / cols;
        let b = ((j + 1) * shares.len() / cols).max(a + 1).min(shares.len());
        let mut acc = vec![0.0f32; bands];
        for r in &shares[a..b] {
            for (k, v) in r.iter().enumerate() {
                acc[k] += v;
            }
        }
        let n = (b - a) as f32;
        for v in &mut acc {
            *v /= n;
        }
        acc
    };
    let band_color = |i: usize| -> Color {
        if i == bands - 1 {
            Color::new(0.24, 0.24, 0.22, 0.78) // muted "other"
        } else {
            let (r, g, b) = hsl_to_rgb(lineage_hue(tracked[i]), 0.62, 0.55);
            Color::new(r, g, b, 0.92)
        }
    };

    // Stacked areas: for each column pair, a filled trapezoid per band. Bands
    // stack bottom→top in `tracked` (id) order, then "other" on top.
    let dx = plot_w / (cols - 1) as f32;
    let mut prev = col_share(0);
    for j in 1..cols {
        let cur = col_share(j);
        let x0 = plot_x + (j - 1) as f32 * dx;
        let x1 = plot_x + j as f32 * dx;
        let mut cum0 = 0.0f32;
        let mut cum1 = 0.0f32;
        for k in 0..bands {
            let (s0, s1) = (prev[k], cur[k]);
            let yb0 = baseline - cum0 * plot_h;
            let yt0 = baseline - (cum0 + s0) * plot_h;
            let yb1 = baseline - cum1 * plot_h;
            let yt1 = baseline - (cum1 + s1) * plot_h;
            let col = band_color(k);
            draw_triangle(vec2(x0, yb0), vec2(x0, yt0), vec2(x1, yt1), col);
            draw_triangle(vec2(x0, yb0), vec2(x1, yt1), vec2(x1, yb1), col);
            cum0 += s0;
            cum1 += s1;
        }
        prev = cur;
    }
    draw_line(plot_x, baseline, plot_x + plot_w, baseline, 1.0, BASELINE);

    // Right-aligned caption: the current dominant dynasty's share.
    if let Some(m) = history.last()
        && m.population > 0
        && let Some(&(_, c)) = m.lineage_counts.first()
    {
        let lbl = format!("top {:.0}%", c as f32 / m.population as f32 * 100.0);
        let tw = measure_text(&lbl, None, 14, 1.0).width;
        draw_text(&lbl, plot_x + plot_w - tw - 2.0, y + 20.0, 14.0, MUTED);
    }
}

/// The selected grazer's brain panel: header + subtitle, then the shared
/// signal-flow node-link diagram ([`draw_brain_graph`]) over the forager
/// sensorium ([`HERB_INPUT_LABELS`]). Live activations are read every frame; the
/// pruned graph was cached at selection.
fn draw_eco_brain_panel(
    eco: &EcoSim,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    sel_view: Option<&CreatureView>,
    graph: Option<&BrainGraph>,
) {
    draw_panel(x, y, w, h);
    let pad = MARGIN;

    // The header + sensor-row labels follow the selection's species: a grazer's
    // forager sensorium or a predator's hunting sensorium.
    let species = sel_view.map(|v| v.species);
    let (base, labels): (&str, &[&str]) = match species {
        Some(Species::Predator) => ("brain - predator", &PRED_INPUT_LABELS),
        _ => ("brain - grazer", &HERB_INPUT_LABELS),
    };
    // Tag the brain's dynamics — CTRNN carries persistent recurrent state
    // (memory), feed-forward resets each tick — so the two competing types in a
    // mixed run are legible on the inspected creature.
    let kind_tag = match sel_view.map(|v| v.kind) {
        Some(BrainKind::Ctrnn) => "  ·  ctrnn",
        Some(BrainKind::Feedforward) => "  ·  feed-forward",
        None => "",
    };
    draw_header(&format!("{base}{kind_tag}"), x + pad, y + 22.0);

    let (Some(v), Some(g)) = (sel_view, graph) else {
        draw_text("(no creature selected)", x + pad, y + 50.0, 15.0, MUTED);
        return;
    };

    let subtitle = if g.active_total > g.edges.len() {
        format!(
            "lineage {}  ·  {:.0}% energy  ·  showing {}/{} edges",
            v.lineage, v.energy_frac * 100.0, g.edges.len(), g.active_total
        )
    } else {
        format!(
            "lineage {}  ·  {:.0}% energy  ·  {} edges, {} inner",
            v.lineage, v.energy_frac * 100.0, g.edges.len(), g.inner_ids.len()
        )
    };
    draw_text(&subtitle, x + pad, y + 42.0, 14.0, MUTED);

    let gx = x + pad;
    let gy = y + 58.0;
    let gw = w - 2.0 * pad;
    let gh = h - 66.0;
    if gh < 40.0 {
        return;
    }
    let Some(neurons) = eco.creature_neurons(v.entity) else {
        return;
    };
    draw_brain_graph(gx, gy, gw, gh, g, &neurons, labels);
}

/// Append the current agent positions as the newest trail sample, capped to
/// `TRAIL_LEN`. Consecutive duplicates are skipped so a stationary agent keeps a
/// short trail rather than a stack of identical points. Trails are index-keyed
/// and rebuilt whenever the population size changes (belt-and-suspenders with the
/// explicit clear at turnover).
fn record_trails(trails: &mut Vec<Vec<(u8, u8)>>, positions: &[(u32, u32)]) {
    if trails.len() != positions.len() {
        trails.clear();
        trails.resize(positions.len(), Vec::new());
    }
    for (i, &(x, y)) in positions.iter().enumerate() {
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
fn accumulate_heat(heat: &mut [u32], positions: &[(u32, u32)]) {
    for &(x, y) in positions {
        heat[(y * 128 + x) as usize] += 1;
    }
}

/// Heuristic "best" agent for auto-selection: prefer an agent currently in the
/// survival zone, breaking ties (and the no-survivor case) by highest `y`. This
/// is exactly right for the band/gauntlet challenges (whose goal is high `y`)
/// and picks a live winner for the others when one exists. Returns None only for
/// an empty population. The returned index is a position in `order` (= `views`).
fn best_agent(sim: &Simulator, views: &[AgentView]) -> Option<usize> {
    views
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| (sim.is_safe(v.pos) as u8, v.pos.1))
        .map(|(i, _)| i)
}

/// Nearest agent to cell (cx, cy) within `radius` cells (Chebyshev), if any.
/// Returns a position in `order` (= `views`).
fn nearest_agent(views: &[AgentView], cx: i32, cy: i32, radius: i32) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    for (i, v) in views.iter().enumerate() {
        let (ax, ay) = v.pos;
        let d2 = (ax as i32 - cx).pow(2) + (ay as i32 - cy).pow(2);
        if d2 <= radius * radius && best.is_none_or(|(bd, _)| d2 < bd) {
            best = Some((d2, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Assign the selection and rebuild the (cached) brain layout for it. The
/// selection index maps into `order`; the wiring is copied out of the ECS.
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
        .and_then(|i| sim.order().get(i).copied())
        .and_then(|e| sim.agent_connections(e))
        .map(|conns| BrainGraph::build(&conns));
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
    views: &[AgentView],
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
        let Some(v) = views.get(i) else { continue };
        let col = agent_color(v, color_mode);
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
    for (i, v) in views.iter().enumerate() {
        let (gx, gy) = v.pos;
        let (px, py) = (sx(gx as f32), sx(gy as f32));
        let col = agent_color(v, color_mode);
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
        && let Some(v) = views.get(i)
    {
        let (x, y) = v.pos;
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
fn agent_color(view: &AgentView, mode: ColorMode) -> Color {
    let hue = match mode {
        ColorMode::Genome => rgb_to_hue(view.rgba),
        ColorMode::Lineage => (view.lineage as f32 * 0.618_034).fract(),
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

/// The dynasty hue (0..1) for a lineage id — the golden-ratio mapping shared by
/// the grazers on the meadow, the selection body/ring, and the bloodlines strip,
/// so one dynasty is the same color everywhere it appears.
fn lineage_hue(lineage: u32) -> f32 {
    (lineage as f32 * 0.618_034).fract()
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
    views: &[AgentView],
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
    draw_status_panel(sim, views, col_x, y, col_w, status_h, target_sps, unlimited, display, heatmap, color_mode, selected);
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
    views: &[AgentView],
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

    let pop = views.len();
    let live_safe = views.iter().filter(|v| sim.is_safe(v.pos)).count();
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
    let px_at = |i: usize| plot_x + i as f32 * dx;

    if n == 1 {
        // A single sample: a flat band across the plot at its value. Kept
        // strictly inside the plot rect.
        let vy = map_y(pts[0]);
        draw_rectangle(plot_x, vy, plot_w, baseline - vy, fill);
        draw_line(plot_x, vy, plot_x + plot_w, vy, 2.0, stroke);
    } else {
        // Fill under the curve as trapezoids between consecutive samples. This
        // stays within [plot_x, plot_x + plot_w] at any point count; the old
        // per-sample vertical line overhung its x by dx/2, which bled out of
        // the panel (over the world view) when few, widely-spaced samples made
        // dx large.
        for i in 1..n {
            let (x0, x1) = (px_at(i - 1), px_at(i));
            let (y0, y1) = (map_y(pts[i - 1]), map_y(pts[i]));
            draw_triangle(vec2(x0, y0), vec2(x1, y1), vec2(x1, baseline), fill);
            draw_triangle(vec2(x0, y0), vec2(x1, baseline), vec2(x0, baseline), fill);
        }
        // Thin line on top.
        for i in 1..n {
            draw_line(px_at(i - 1), map_y(pts[i - 1]), px_at(i), map_y(pts[i]), 2.0, stroke);
        }
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

    // Min/max value labels, right-aligned inside the plot rect (a couple px off
    // the right edge) so a wide auto-scaled value can't spill over the panel
    // edge or the line.
    let hi_lbl = fmt.replace("{:.0}", &format!("{:.0}", hi * scale));
    let lo_lbl = fmt.replace("{:.0}", &format!("{:.0}", lo * scale));
    let right_label = |lbl: &str, ty: f32| {
        let tw = measure_text(lbl, None, 14, 1.0).width;
        draw_text(lbl, plot_x + plot_w - tw - 2.0, ty, 14.0, MUTED);
    };
    right_label(&hi_lbl, plot_y + 12.0);
    right_label(&lo_lbl, baseline - 2.0);
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
/// Herbivore forager sensorium labels, keyed to `eco::HERB_INPUTS` (12 inputs):
/// bias, oscillator, own energy, random, biomass-here, the two-axis biomass
/// gradient, four directional blocked-neighbour sensors, and local crowding.
const HERB_INPUT_LABELS: [&str; 12] = [
    "bias", "osc", "energy", "rand", "food", "grad-ns", "grad-ew", // 0..6
    "blk-N", "blk-S", "blk-E", "blk-W", "crowd", // 7..11
];
/// Predator hunting sensorium labels, keyed to `eco::PRED_INPUTS` (12 inputs):
/// bias, oscillator, own energy, random, prey-density-here, the two-axis prey
/// direction (over the wide sensing radius), four directional blocked sensors,
/// and own-species (pack) spacing.
const PRED_INPUT_LABELS: [&str; 12] = [
    "bias", "osc", "energy", "rand", "prey", "prey-ns", "prey-ew", // 0..6
    "blk-N", "blk-S", "blk-E", "blk-W", "pack", // 7..11
];
const OUTPUT_LABELS: [&str; 5] = ["rand", "N", "S", "E", "W"];
/// Cap on edges drawn in the challenge brain panel; above this we keep the
/// top-|weight|.
const MAX_EDGES: usize = 120;
/// Tighter edge cap for the eco creature signal-flow panel (grazer or predator) —
/// the ~30 strongest connections, so the live wiring reads as one legible circuit.
const CREATURE_MAX_EDGES: usize = 30;
/// Live-signal magnitude (|source activation × weight|) that maps to a fully
/// bright edge in the signal-flow inspector; stronger signals clamp here.
const SIGNAL_FULL: f32 = 1.2;
/// Output-neuron activation above which a movement output "fires" (the brain's
/// `move_activation` threshold is 0.0); firing outputs flash in the inspector.
const FIRE_THRESHOLD: f32 = 0.0;

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
    /// Prune to the subgraph that can influence an output, capping at the
    /// default [`MAX_EDGES`] (the challenge inspector's budget).
    fn build(conns: &[Connection]) -> BrainGraph {
        Self::build_capped(conns, MAX_EDGES)
    }

    /// Prune to the subgraph that can influence an output (backward reachability
    /// from the output layer), then cap to the top `max_edges` by |weight|. Takes
    /// the connections copied out of the ECS (the viewer holds no borrow). The
    /// eco creature panel passes a tighter [`CREATURE_MAX_EDGES`] cap.
    fn build_capped(conns: &[Connection], max_edges: usize) -> BrainGraph {
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

        if edges.len() > max_edges {
            edges.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap());
            edges.truncate(max_edges);
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
    let Some(&entity) = sim.order().get(idx) else {
        draw_text("(selection lost)", x + pad, y + 50.0, 15.0, MUTED);
        return;
    };
    let Some(lineage) = sim.agent_lineage(entity) else {
        draw_text("(selection lost)", x + pad, y + 50.0, 15.0, MUTED);
        return;
    };

    let subtitle = if g.active_total > g.edges.len() {
        format!("lineage {}   ·   showing {}/{} edges", lineage, g.edges.len(), g.active_total)
    } else {
        format!("lineage {}   ·   {} edges, {} inner", lineage, g.edges.len(), g.inner_ids.len())
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

    // Live activations, copied out of the ECS (no borrow held during drawing).
    let Some(neurons) = sim.agent_neurons(entity) else {
        return;
    };
    draw_brain_graph(gx, gy, gw, gh, g, &neurons, &INPUT_LABELS);
}

/// The shared signal-flow node-link renderer for a pruned [`BrainGraph`], drawn
/// into the rect `(gx, gy, gw, gh)`. Inputs run down the left, inner neurons the
/// middle, movement outputs the right. **Edges carry live signal**: colored by
/// weight sign (blue excitatory / red inhibitory), thickness by |weight| (the
/// stable structure) but brightness + alpha by the *live* signal magnitude
/// |source activation × weight|, so active pathways light up and a quiet brain
/// goes dark. Output nodes **flash** (a warm halo) on the frame they fire (cross
/// the [`FIRE_THRESHOLD`]). `input_labels` names the sensor rows and sets how
/// many input nodes to draw, so the same renderer serves the 15-input challenge
/// agent and the 12-input eco herbivore.
fn draw_brain_graph(
    gx: f32,
    gy: f32,
    gw: f32,
    gh: f32,
    g: &BrainGraph,
    neurons: &[Vec<f32>],
    input_labels: &[&str],
) {
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
    let n_in = input_labels.len();
    let input_y = |id: u8| pos_y(id as usize, n_in);
    let output_y = |id: u8| pos_y(id as usize, OUTPUT_LABELS.len());
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
    let act = |layer: u8, id: u8| -> f32 {
        neurons.get(layer as usize).and_then(|l| l.get(id as usize)).copied().unwrap_or(0.0)
    };

    // Edges first (under the nodes). Sign sets the hue; the live signal
    // (source activation × weight) sets brightness/alpha, so the diagram shows
    // what is *flowing* this tick, not just the static wiring.
    for e in &g.edges {
        let (x0, y0) = (node_x(e.src_layer), node_y(e.src_layer, e.src_id));
        let (x1, y1) = (node_x(e.sink_layer), node_y(e.sink_layer, e.sink_id));
        let wmag = (e.weight.abs().min(2.0) / 2.0).clamp(0.0, 1.0);
        let signal = (act(e.src_layer, e.src_id) * e.weight).abs();
        let smag = (signal / SIGNAL_FULL).clamp(0.0, 1.0);
        let thick = 0.6 + wmag * 2.2;
        let target = if e.weight >= 0.0 { SURVIVAL_HUE } else { EDGE_NEG };
        // A faint structural ghost (alpha 0.10) so quiet wiring stays readable,
        // brightening toward the sign color as live signal grows.
        let col = Color::new(
            lerp(BASELINE.r, target.r, smag),
            lerp(BASELINE.g, target.g, smag),
            lerp(BASELINE.b, target.b, smag),
            0.10 + smag * 0.80,
        );
        // A recurrent self-loop (an inner neuron feeding itself — the CTRNN memory
        // latch) is a zero-length line; draw it as a small ring beside the node so
        // the recurrence is visible.
        if e.src_layer == e.sink_layer && e.src_id == e.sink_id {
            draw_circle_lines(x0 + 6.0, y0 - 6.0, 4.5, 0.6 + wmag * 1.4, col);
            continue;
        }
        draw_line(x0, y0, x1, y1, thick, col);
    }

    // Inner nodes (brightness by current activation).
    for &id in &g.inner_ids {
        draw_circle(x_mid, inner_y(id), 3.0, activation_color(act(1, id)));
    }

    // Input nodes + labels (left).
    for (id, label) in input_labels.iter().enumerate() {
        let ny = input_y(id as u8);
        draw_circle(x_in, ny, 4.0, activation_color(act(0, id as u8)));
        draw_text(label, gx, ny + 4.0, 13.0, MUTED);
    }
    // Output nodes + labels (right). A firing output (activation over the
    // movement threshold) flashes a warm halo — the creature's live decision.
    for (id, label) in OUTPUT_LABELS.iter().enumerate() {
        let a = act(2, id as u8);
        let ny = output_y(id as u8);
        if a > FIRE_THRESHOLD {
            let fire = a.clamp(0.0, 1.0);
            draw_circle(x_out, ny, 6.0 + 6.0 * fire, Color::new(1.0, 0.82, 0.36, 0.32 * fire));
            draw_circle_lines(x_out, ny, 7.5, 1.6, Color::new(1.0, 0.88, 0.5, 0.75 * fire));
        }
        draw_circle(x_out, ny, 5.0, activation_color(a));
        draw_text(label, x_out + 10.0, ny + 4.0, 13.0, TEXT_SECONDARY);
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
        let views = sim.agent_views();
        let i = best_agent(&sim, &views).unwrap();
        let conns = sim.agent_connections(sim.order()[i]).unwrap();
        let g = BrainGraph::build(&conns);
        assert!(g.edges.len() <= MAX_EDGES, "drawn edges exceed cap");
        assert!(
            g.active_total <= conns.len(),
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
