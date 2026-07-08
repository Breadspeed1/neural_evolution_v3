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
- **Ecosystem terrarium — predators** *(rung 3 of the ecosystem ladder;
  `--mode eco`)*: the **apex trophic level** and the project's **first
  multi-species coevolution**, completing the pyramid soil → plants → herbivores →
  predators. Herbivores and predators are both mobile hecs entities sharing one
  `Creature` bundle (`Position` + genome/brain/energy/lineage) tagged by a
  **`Species { Herbivore, Predator }`** enum; a **single stable `order`** covers
  both species (position-in-`order` = the per-entity RNG index, one
  `creature_seed(seed, tick, index)` stream), and the serial entity phase iterates
  it **twice** — grazers first (rung-2 logic verbatim, so a herbivore-only run is
  byte-identical), then hunters — branching on species. Predators reuse the
  feed-forward [`Brain`] over a **13→12-input hunting sensorium** (`PRED_INPUTS`:
  bias/oscillator/own-energy/random, a wide-radius **prey-density** scalar + a
  distance-weighted **prey-direction** NS/EW gradient, four directional
  blocked-neighbour sensors, and own-species **pack** spacing). Each tick a
  predator senses → brain → moves (up to `pred_speed` cells, cell-by-cell) →
  **catches** the first adjacent herbivore (prey dies; predator assimilates
  `catch_efficiency` of its energy, the remainder → soil), pays metabolism, then
  starves (corpse → nutrient) or reproduces (mutated child, energy split). Caught
  prey are removed from `order` in the same order-stable, index-preserving way as
  starvation (deaths collected in a `dead_set`, applied once after both
  sub-phases; a prey born *and* eaten in one tick counts as a birth and a death
  and never joins `order`). **Scale separation** (the "birds" flavour): predators
  sense prey over radius **4** (vs the herbivore's single-cell biomass gradient)
  and move **2** cells/tick (vs 1) — a distinct, faster hunting scale.
  **Determinism preserved**: two same-seed eco runs (predators on) are
  byte-identical JSONL (verified by diff); the challenge sim is untouched and
  stays byte-identical. `EcoMetrics` gains predator count / births / deaths /
  mean-energy; the viewer draws predators as **warm-red apex chevrons** over the
  grazer dots, adds a **predator population sparkline** (the full soil → plants →
  herbivores → predators readout), and extends selection + the signal-flow brain
  inspector to predators (`p` picks the fattest hunter; the panel switches to the
  predator sensor labels). **Tri-trophic coexistence gate met** (seeds 42/7/123,
  128², committed defaults, **no reseeding**): from 150 herbivore + 26 predator
  founders the meadow blooms (coverage 0.9 % → 55 % by tick 200), herbivores boom
  in its wake (→ ~520 by tick 500), predators **lag-peak** trailing them (→ ~160
  by tick 1000, ~500 ticks behind the prey peak) and crash the herbivore boom,
  after which the three levels settle into a **persistent, self-sustaining, lagged
  oscillation that runs to 50 000 ticks on every seed tested** (seed 42: herbivores
  swing ~15–730, predators ~26–274, plant coverage ~33–39 %, biomass ~4400–4700,
  all bounded away from zero; seeds 7 and 123 the same band). The balance is a
  **top-down trophic cascade** (the "green world"): predators hold herbivores far
  below their food carrying capacity (~2000 without predators → ~100s with them),
  so the meadow stays **lush** at ~33 % coverage rather than being grazed to the
  rung-2 ~13 %. Three things make the coexistence robust rather than a
  Lotka–Volterra death spiral — the genuinely hard part of the 3-level balance:
  **(1) a *moderate* sensing radius (4).** Wider radii (tested 6–10) make the hunt
  *mean-field* — every predator sees the whole prey field, so local prey
  depletions synchronize into one global predator crash and the system spirals to
  extinction (radius-6/9 configs collapse on 2 of 3 seeds). Radius 4 keeps the
  hunt **spatially structured**: many asynchronous local predator–prey cycles
  average into a stable global coexistence, and sparse low-density prey patches are
  a refuge the short-sighted predators can't find. **(2) Deep predator energy
  reserves** (`pred_energy_max` 80 vs metabolism 0.045 ≈ **1800 ticks** of famine
  buffer) so predators *coast through* a prey trough instead of starving out at it
  — the same reserve-buffer principle that made the rung-2 herbivore layer robust,
  applied to the apex tier (shallow reserves ⇒ extinction in the first deep
  trough). **(3) Modest per-catch value + slow reproduction** (efficiency 0.6, a
  high repro threshold of 42, and a hard cap of **one catch per predator per
  tick**) keep predators from over-multiplying and cropping the prey to zero. See
  the balance discussion under roadmap step 6.
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
  tinting, survival/diversity sparklines, lineage coloring, a live **signal-flow**
  brain-inspector node-link diagram (edge brightness = live signal, outputs flash
  on firing), motion trails, oriented agents, generation-turnover pulse, occupancy
  heatmap. The eco terrarium adds a **spotlight** protagonist mode (`f`), a
  **bloodlines** stacked-area strip of dynasty share over time, entity-tracked
  click/`b` selection, and an energy-scaled calmer crowd (see roadmap 1).

## What the current substrate can already produce (just not selected for / shown)

These need no new mechanics — only selection pressure and the right readout:

- **Herding / flow.** Agents sense the NS/EW population gradient; social movement
  is latent. Bottleneck challenges (gauntlet) already induce faint columns at the
  gaps. Spotlight + trails make it legible.
- **Rhythmic gaits.** The oscillator input can evolve into periodic movement.
  Invisible while we only record final position; a path-shape readout surfaces it.
- **Selective sweeps.** Neutral diversity collapsing into a few winning lineages,
  with rare comebacks — real population genetics. Now made visible in the eco
  terrarium by the **bloodlines strip** (stacked lineage share over time); seed 42
  resolves from a diverse rainbow into one dynasty holding ~64% by tick ~9000.

## Roadmap (dependency-ordered; each step is a measurable gate)

*Rung 0 (done): the hybrid world-model refactor above — hecs entities for
creatures + a typed cell grid for the substrate — is the foundation the layered
terrarium is built on. Steps 4 (energy/food) and 5 (pheromone field) are the ones
that cash in its per-cell state and per-entity components; it was landed first, as
a behavior-preserving swap, so those can be built cleanly.*

### 1. Viewer storytelling *(done)*
The sim view showed thousands of equally-weighted dots — spectacle without story
("like watching sprinkles"); the brain panel showed static wiring, not activity.
Shipped, in the eco terrarium and (where it unified cleanly) the challenge sim:
- **Spotlight mode** (`f`): dims the whole crowd to a faint wash and elevates the
  one selected protagonist — full-bright body, a white selection ring, a fading
  comet-tail trail. The primary confetti fix: one protagonist against a quiet
  field. Off by default (the calm baseline is the resting view).
- **Signal-flow brain**: the shared node-link inspector now carries *live* signal
  — edge brightness/alpha = |source activation × weight| (quiet wiring goes dark,
  active pathways light up), output nodes flash a warm halo on the tick they fire
  (cross the movement threshold), capped to the ~30 strongest edges for the eco
  herbivore (input rows labelled from the 12-input forager sensorium). The
  challenge inspector inherits the same live edges + output flash.
- **Eco selection**: click picks the nearest grazer, `b` the heuristic best —
  the *highest-energy member of the current largest lineage* (the reigning
  dynasty's fittest grazer), so the ringed protagonist's hue matches the widest
  bloodlines band. Selection follows the creature by stable entity id across
  ticks and auto-reselects (via `b`) on its death.
- **Bloodlines strip**: a stacked-area chart of lineage population share over
  time, driven by the per-interval dynasty record (top-N lineages by population,
  the rest folded to "other"). Bands are hued by the same `lineage_hue` mapping
  the grazers use, so a band matches its creatures on the field — watch dynasties
  bloom, dominate, and crash (a real selective sweep: seed 42 resolves from a
  diverse rainbow into one dynasty holding **~64%** by tick ~9000).
- **Calmer baseline** (spotlight off): the crowd drawn smaller/dimmer and
  desaturated, with radius *and* brightness ∝ energy (fat vs starving at a
  glance), so density reads as a breathing meadow; per-agent trails are now
  spotlight-only (the one protagonist), not everyone-always.

The lib stayed render-agnostic — the only new data surface was the committed
read-only hooks (herbivore brain/lineage/energy accessors + the dynasty record);
all rendering lives in `main.rs`. Determinism unchanged (same-seed eco runs stay
byte-identical). Keys: `q`/`e` speed, `v` display, `f` spotlight, `b` best,
click select (challenge keeps `c` colors, `h` heat).

*Gate met: one grazer's decisions are legible from its live brain panel (outputs
flash as it moves), and a lineage sweep is visible in the bloodlines strip.*

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

*Landed as the eco terrarium's **rung 3** (`--mode eco`, predators; see "Where we
are").* The coupled population dynamics gate is **met**: predators and prey trace a
persistent **lagged oscillation** (predator peaks trailing prey peaks by ~500
ticks) that self-sustains to 50 000 ticks across seeds 42/7/123 with no reseeding,
on a lush plant base — a top-down trophic cascade, the full soil → plants →
herbivores → predators pyramid. Predators reuse rung-2's seams exactly: the
`Position` + brain-bundle + single-`order`/`creature_seed` determinism pattern
(now shared by both species behind a `Species` tag), a wide-radius prey sensor,
and a catch = herbivore death + energy transfer. **The 3-level balance was the
hard part** — the naive tuning collapses (predators too effective → prey wiped →
predators starve; or too weak → predators die out → back to rung 2). What makes it
coexist: a *moderate* (spatially-structured, not mean-field) predator sensing
radius so local cycles don't synchronize into a global crash, **deep predator
energy reserves** that coast through prey troughs, and modest per-catch value +
slow reproduction (+ one catch/tick) so predators can't crop the prey to zero (the
full numbers + reasoning are in the rung-3 bullet above). *Evolved pursuit/evasion
is latent but not yet isolated as a paired baseline curve — the prey-direction
sensor makes pursuit selectable, and herbivores sensing predator-occupied
neighbours as "blocked" makes evasion selectable — a natural next readout.*
**Open seams from here:** (a) **CTRNN brains** (roadmap 3) give memory-based
hunting/foraging — a predator that remembers where prey was last seen, or a prey
that flees a *remembered* threat direction — which the current feed-forward brain
cannot; the sparse connection genome + `--brain` A/B seam compose directly with
both species. (b) **Niche enrichment**: a second prey or predator species, size/
speed trait axes, evolvable pack-hunting off the `pack` sensor, or the pheromone
field (roadmap 5) as a shared alarm/trail channel — each is an additive
`Species`/sensor variant on the same order/determinism backbone, not a rewrite.

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
