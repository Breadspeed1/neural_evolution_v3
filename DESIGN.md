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

- **Core sim**: parallel (rayon) step, seeded ChaCha8 determinism (reproducible
  across the parallel decision phase), geometric-skip mutation.
- **Harness**: lib + headless/viewer bins, clap CLI, per-generation JSONL
  metrics (survival, mean/max final y, genome diversity), champion save/load,
  `--bench` regression gate, unit tests incl. determinism.
- **Challenges** (`--challenge`): `north-band` (default), `corners`, `gauntlet`,
  `moving-band`, `enclosure` — each an obstacle layout + survival predicate, with
  documented baseline curves (see `CHALLENGES.md`). `moving-band` and `enclosure`
  are the two that stay unsaturated and reward real behavior.
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

### 2. Phase C — CTRNN brains
Upgrade neurons to continuous-time recurrent leaky integrators (per-neuron time
constant + bias, persistent state across steps, inner↔inner recurrence) on the
**same sparse connection genome**. Memory and timing are prerequisites for most
interesting behavior — a feed-forward net cannot "remember where food was."
- `--brain {feedforward,ctrnn}` so we A/B, not replace.
- Stay CPU + rayon; burn/GPU was accidental complexity.

*Gate: CTRNN beats feed-forward on `moving-band` and `enclosure` (the tasks that
reward memory), shown as paired baseline curves. If it doesn't, understand why
before proceeding.*

### 3. Continuous world + energy/food
The biggest unlock: make survival **behavioral** (forage, don't starve) instead
of **positional** (be north at the buzzer). Introduces exploration/exploitation
tradeoffs and, if food regrows, boom/bust population cycles — the sim becomes an
ecology rather than a fitness test. A continuous (sub-cell) world also lets agents
be smooth-moving shapes instead of grid-snapped texels — a visual upgrade in
itself.

*Gate: a foraging strategy measurably outperforms a random walker; population
shows density-dependent dynamics.*

### 4. Pheromone / stigmergy layer
A diffusing scalar field agents deposit into and sense (another `Vec<f32>` grid,
like the world). This is how ants form trails with no central control — and it is
the project's likely **beauty peak**: a living, glowing substrate the creatures
paint as they move. Cheap to add, high visual and behavioral reward.

*Gate: visible deposit→follow trails form between resources without being
hand-coded.*

### 5. Predation / multi-species
A second population (or intraspecific predation) drives a prey–predator arms race:
evasion, pursuit, possibly pack behavior — the richest emergence available, and
the most spectacular to watch. Depends on the continuous world (3) and benefits
from memory (2) and signaling fields (4).

*Gate: coupled population dynamics (Lotka–Volterra-like oscillation) and at least
one evolved pursuit or evasion behavior.*

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
of stages 3–5, not as a rewrite.
