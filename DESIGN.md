# Design & roadmap

This project began as a hand-written biosim4-style evolution sim (128×128 grid,
sparse genome-encoded neural nets, positional selection). It is being grown into
a small, watchable artificial-life system. This document records the direction
and — more importantly — the **discipline** that governs it.

## The governing principle: signal before substrate

A sibling project (`ctrnn-sim`) died at "it works but it sucks." Its failure was
not bad code — the CTRNN math was fine. It was **building substrate before
establishing signal**: it jumped to evolved active-vision MNIST (where random
networks sit at chance, so selection is drift, not learning) and stacked
GPU tensors, a bevy visualizer, a TUI, and a 7-parameter trait system *before
ever observing a single fitness curve*.

Every step in this roadmap obeys the inversion of that mistake:

1. **Measure before you build.** Each new mechanic ships with a seeded baseline
   curve from the metrics harness. If evolution can't be *seen* improving on it,
   the mechanic is wrong or the task is degenerate — fix that before adding more.
2. **Keep the genome sparse.** The connection-list genome (each gene → one
   connection) is a better search space than a dense weight matrix. Upgrades
   change what a neuron *does*, not how the genome is encoded.
3. **Clean and minimal.** No speculative abstraction, no framework. A new
   environment is an `enum` variant and a match arm, not a trait hierarchy.
   Prefer deleting to wrapping.
4. **The lib stays render-agnostic.** Simulation core has no dependency on the
   display. Visual richness lives in the viewer; the lib only exposes data.

## Where we are

- **World model** *(rung 0 of the ecosystem ladder — the foundational refactor
  that the layered-terrarium steps below build on)*: a **hybrid** substrate.
  Creatures are **hecs entities** (a `Position` component + an `Agent` bundle of
  brain/genome/lineage/…), and the spatial world is a **typed cell `Grid`**
  (`Vec<Cell>`, each `Cell` an obstacle flag + `Option<Entity>` occupant),
  replacing the old `Vec<u128>` occupancy bitmask. This lifts the hard 128×128
  cap (grid width/height are parameters, still defaulting to 128, which the
  challenge coordinates assume) and gives every cell room to grow per-cell state
  (nutrient/biomass) for the energy and pheromone rungs — without touching the
  agents. Determinism is unchanged: the Simulator keeps its own `order:
  Vec<Entity>` in stable birth order (hecs archetype iteration is *not* stable
  across despawns), and position in `order` is the per-agent RNG index exactly as
  the old `Vec<Agent>` index was.
- **Ecosystem terrarium — producer base** *(rung 1 of the ecosystem ladder;
  `--mode eco`)*: a **separate, continuous (generation-less)** sim in
  `src/eco.rs` that reuses the shared substrate — the cell `Grid`, seeded
  ChaCha8 determinism, rayon, and the JSONL-metrics/dashboard patterns — without
  touching the generational `Simulator`; the two coexist behind
  `--mode {challenge,eco}` (default `challenge`, fully unchanged and
  byte-identical). Two trophic layers are modeled as **double-buffered
  cellular-automaton fields** on the grid (no animals / hecs entities yet — plants
  are per-cell state): a **nutrient** field (soil/decomposers: von-Neumann
  diffusion with no-flux boundaries + replenishment toward a soil capacity) and a
  **plant biomass** field (producers: nutrient-fed logistic growth that *consumes
  local nutrient* — the self-limiter — plus probabilistic seeding into empty
  neighbors, slow senescence, and a gentle disturbance mortality that returns mass
  to soil and keeps the mosaic breathing). The update reads an immutable snapshot
  and writes a fresh buffer, so it is order-independent, rayon-parallel over cells,
  and reproducible; the one stochastic step (seeding/mortality) draws a ChaCha8
  seeded from `(seed, tick, cell-index)`. **Signal-first gate met**: from a sparse
  ~1% seeding (seed 42, 128²) coverage climbs (tick 25 → 4.3%, 50 → 11.0%,
  100 → 29.8%), overshoots to a ~56% pioneer bloom at tick 200, then **relaxes to
  a stable, patchy, gently-fluctuating plateau of ~32% coverage** (tick 500 →
  43.5%, 1000 → 32.4%, 5000 → 32.6%, 10000 → 32.4%; mean nutrient ~0.31) — neither
  filling to 100% nor collapsing to 0%, and seed-independent (seed 123 → 32.5%).
  Same-seed
  runs are **byte-identical** (verified by diffing two headless metrics files).
  Headless `--mode eco --ticks N` prints the coverage-over-time curve; the viewer
  renders a layered diorama (subtle soil wash ∝ nutrient, green ∝ √biomass) with
  biomass + coverage sparklines on the shared dark shell. This producer base is
  what rung 2's herbivores (hecs entities with energy) will graze.
- **Ecosystem terrarium — herbivores** *(rung 2 of the ecosystem ladder;
  `--mode eco`)*: the first **mobile trophic level** and the project's first
  **generation-less birth/death evolution** — individuals with an energy budget
  who forage, reproduce, and die with **no global fitness function**. Herbivores
  are **hecs entities** (`Position` + a `Herbivore` bundle of genome/brain/energy/
  lineage) reusing the challenge sim's sparse-genome feed-forward [`Brain`], now
  width-parametric, over a **12-input forager sensorium** (bias, oscillator, own
  energy, random, biomass-here, a 2-axis biomass gradient, 4 directional
  blocked-neighbor sensors, and local herbivore density) → the same movement
  outputs. Each tick is (1) the rung-1 parallel field CA, then (2) a **serial,
  order-stable entity phase**: per herbivore in birth `order`, sense → brain →
  move (costs energy, blocked by the grid) → **graze** (biomass→energy up to a
  cap and a satiation headroom, depleting the cell) → pay **metabolism** → then
  **die** (energy ≤ 0 ⇒ despawn, body → a nutrient pulse in the cell, closing the
  loop herbivore→soil→plant) or **reproduce** (energy ≥ threshold ⇒ a
  `mutate_genome` child on an empty neighbor, energy split in half). Determinism
  is preserved exactly as the `Simulator`: a stable `order: Vec<Entity>` (append
  on birth, `retain` on death), position-in-`order` as the per-entity RNG index,
  a per-entity `herbivore_seed(seed, tick, index)` stream (domain-tagged so it
  never aliases the plant CA's `cell_seed`), and the parallel field phase strictly
  separated from the serial entity phase — **two same-seed runs are byte-identical
  JSONL** (verified by diff). `EcoMetrics` gains population / births / deaths /
  mean-energy (the Lotka–Volterra readout); the viewer draws the grazers as
  bright lineage-colored creatures on the meadow with a **population sparkline**
  beside coverage + biomass. **Coexistence gate met** (seed 42, 128², defaults,
  no reseeding): from 150 founders the meadow blooms (coverage 0.9%→**53%** by
  tick 200), grazers boom in its wake (150→**2523** by tick 500) and crash the
  bloom (coverage **53%→19%** by tick 1000) — one clear predator-prey cycle —
  after which plants and herbivores **relax to a persistent steady state**
  (coverage ~12%, biomass ~1000, **~1600–2000 grazers**, mean energy ~3.3,
  births≈deaths) that **self-sustains to 40 000 ticks without collapsing or
  exploding**, robust across seeds (42/7/123 all settle to the same band). The
  balance: metabolism (0.02) vs graze-cap×coverage sets a per-tick energy budget
  that is *marginally* positive on the patchy meadow, so the grazer count is
  pinned by food, not runaway; large energy reserves (`energy_max` 10, satiation
  headroom on grazing, `repro_threshold` 7) keep grazers off the starvation edge
  — that reserve buffer is what makes coexistence robust rather than a knife-edge.
  Sustained *global* oscillation stays damped (the producer base's own strong
  self-regulation plus spatial averaging over 16 384 cells), so what persists is
  a stable coexistence with an opening cycle rather than a limit cycle.
- **Core sim**: parallel (rayon) step, seeded ChaCha8 determinism (reproducible
  across the parallel decision phase), geometric-skip mutation.
- **Harness**: lib + headless/viewer bins, clap CLI, per-generation JSONL
  metrics (survival, mean/max final y, genome diversity), champion save/load,
  `--bench` regression gate, unit tests incl. determinism and the grid/order
  invariants.
- **Challenges** (`--challenge`): the positional set `north-band` (default),
  `corners`, `gauntlet`, `moving-band`, `enclosure`, plus the pattern-forming set
  `flock`, `ring`, `heart`, `orbit` — each an obstacle layout + survival
  predicate, with documented baseline curves (see `CHALLENGES.md`). All climb
  under selection; the pattern set sculpts the swarm into a blob, ring, heart, or
  pinwheel.
- **Sensors**: 17 inputs — constants, oscillator, age, random, global crowd
  centroid, eight directional obstacle sensors, and (new) two self-position
  sensors (`x/127`, `y/127`) that let an agent navigate to absolute coordinates,
  which is what unlocks the pattern challenges.
- **Dashboard**: dark-themed live viewer — world with challenge-aware terrain
  tinting, survival/diversity sparklines, lineage coloring, a live brain-inspector
  node-link diagram, motion trails, oriented agents, generation-turnover pulse,
  occupancy heatmap.

## What the current substrate can already produce (just not selected for / shown)

These need no new mechanics — only selection pressure and the right readout:

- **Herding / flow.** Agents sense the NS/EW population gradient; social movement
  is latent. Bottleneck challenges (gauntlet) already induce faint columns at the
  gaps. Spotlight + trails make it legible.
- **Rhythmic gaits.** The oscillator input can evolve into periodic movement.
  Invisible while we only record final position; a path-shape readout surfaces it.
- **Selective sweeps.** Neutral diversity collapsing into a few winning lineages,
  with rare comebacks — real population genetics, made visible by a bloodlines
  strip chart (stacked lineage share over generations).

## Roadmap (dependency-ordered; each step is a measurable gate)

*Rung 0 (done): the hybrid world-model refactor above — hecs entities for
creatures + a typed cell grid for the substrate — is the foundation the layered
terrarium is built on. Steps 4 (energy/food) and 5 (pheromone field) are the ones
that cash in its per-cell state and per-entity components; it was landed first, as
a behavior-preserving swap, so those can be built cleanly.*

### 1. Viewer storytelling *(in progress)*
The sim view shows 1000 equally-weighted dots — spectacle without story; the brain
panel shows static wiring, not activity. Fixes:
- **Spotlight mode**: dim the crowd, elevate one protagonist; its brain panel
  becomes its live mind, so behavior (left) correlates with circuit (right).
- **Signal-flow brain**: edge brightness = live signal (source activation ×
  weight), output nodes flash on firing, cap to the ~30 strongest edges.
- **Bloodlines strip**: lineage population share over generations (needs a tiny
  data-only lib hook: per-generation lineage counts).
- **Calmer baseline**: smaller/dimmer crowd; full trails become part of spotlight.

*Gate: one creature's decisions are legible from its brain panel; a lineage sweep
is visible in the strip.*

### 2. Creative visual goals — position sensor + pattern challenges *(current)*
The near-term payoff: selection targets that make the swarm *draw* something.
Landed here is the self-position sensor (inputs 15/16) plus four position-based
challenges — `flock` (converge to a blob), `ring` (hold a radius), `heart` (fill
an implicit heart curve), `orbit` (circle the center). The self-position sensor is
the enabling change: without it an agent senses only the crowd and walls, so it
can follow but never navigate to a coordinate. Each challenge ships with a seeded
baseline curve (`CHALLENGES.md`) and shows a clean climb from a low gen-0 floor.
The static targets (`ring`, `heart`) already tint the world overlay for free; the
dynamic ones (`flock`, `orbit`) still want a viewer readout (crowd blob / rotation
arc) to be fully legible, which rides with the storytelling pass above.

*Gate (met for the sim half): all four evolve a visible climb (`heart` 3 → 70%,
`flock` 1 → 56%, `ring` 3 → 62%, `orbit` 2 → 24%). Remaining: a viewer treatment
for the two dynamic goals.*

### 3. Phase C — CTRNN brains
Upgrade neurons to continuous-time recurrent leaky integrators (per-neuron time
constant + bias, persistent state across steps, inner↔inner recurrence) on the
**same sparse connection genome**. Memory and timing are prerequisites for most
interesting behavior — a feed-forward net cannot "remember where food was."
- `--brain {feedforward,ctrnn}` so we A/B, not replace.
- Stay CPU + rayon; burn/GPU was accidental complexity.

*Gate: CTRNN beats feed-forward on `moving-band` and `enclosure` (the tasks that
reward memory), shown as paired baseline curves. If it doesn't, understand why
before proceeding.*

### 4. Continuous world + energy/food
The biggest unlock: make survival **behavioral** (forage, don't starve) instead
of **positional** (be north at the buzzer). Introduces exploration/exploitation
tradeoffs and, if food regrows, boom/bust population cycles — the sim becomes an
ecology rather than a fitness test. A continuous (sub-cell) world also lets agents
be smooth-moving shapes instead of grid-snapped texels — a visual upgrade in
itself.

*Gate: a foraging strategy measurably outperforms a random walker; population
shows density-dependent dynamics.*

*Landed via the ecosystem ladder (rung 2, `--mode eco`): survival is now purely
behavioral (graze or starve) with generation-less energy-driven birth/death, and
the population shows clear density-dependent dynamics — a founder bloom, a boom
that crashes its own food, then a self-sustaining plant↔herbivore coexistence
(see "Where we are"). Foraging-strategy-beats-random-walker is implicit (better
foragers leave more offspring, so lineages that ignore the food gradient are
selected out), but is not yet isolated as a paired baseline curve; the continuous
sub-cell world / smooth shapes remain the one unbuilt piece of this gate.*

### 5. Pheromone / stigmergy layer
A diffusing scalar field agents deposit into and sense (another `Vec<f32>` grid,
like the world). This is how ants form trails with no central control — and it is
the project's likely **beauty peak**: a living, glowing substrate the creatures
paint as they move. Cheap to add, high visual and behavioral reward.

*Gate: visible deposit→follow trails form between resources without being
hand-coded.*

### 6. Predation / multi-species
A second population (or intraspecific predation) drives a prey–predator arms race:
evasion, pursuit, possibly pack behavior — the richest emergence available, and
the most spectacular to watch. Depends on the continuous world (4) and benefits
from memory (3) and signaling fields (5).

*Gate: coupled population dynamics (Lotka–Volterra-like oscillation) and at least
one evolved pursuit or evasion behavior.*

*Now concretely the **next eco rung (rung 3)**: add a carnivore trophic level on
top of rung 2's herbivores — the coupled plant↔herbivore dynamics and the
byte-identical serial-entity-phase machinery are already in place, so rung 3 is a
second entity species that senses + eats herbivores (prey energy → predator
energy, closing the pyramid). The rung-2 handoff seams for it: herbivores already
carry energy + lineage and die into the nutrient field; a predator reuses the
same `Position` + brain-bundle + `order`/`herbivore_seed` determinism pattern,
adds a "nearest prey" sensor, and turns a catch into an energy transfer + a
herbivore death. CTRNN brains (roadmap 3) and the pheromone field (roadmap 5)
compose with it but are not prerequisites.*

### Deferred / lower priority
- **Communication channels** (an output other agents can sense): evolvable
  signaling, but hard to *see* without instrumentation — revisit after predation.
- **3D view**: adds camera complexity without adding information at current
  density; the render-agnostic lib keeps it a new viewer bin, not a rewrite, if
  ever wanted.

## Long-term shape

Eventually: **continuous (generation-less) evolution** in a larger world, where
birth/death are local events driven by energy rather than a global generation
clock. The steps above are chosen so that this arrives as the natural consequence
of stages 4–6, not as a rewrite.
