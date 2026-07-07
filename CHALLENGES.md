# Challenges

Each `--challenge` selects a *selection environment*: an obstacle layout stamped
into the 128×128 world at the start of every generation, plus a survival
predicate applied to each agent's final position. Both are pure functions of the
generation number (or constant), so runs stay bit-for-bit reproducible. Agents
that satisfy the predicate reproduce; the rest are culled. There is no energy,
food, or per-step scoring — only where you end up.

Names are kebab-case: `north-band` (default), `corners`, `gauntlet`,
`moving-band`, `enclosure`, and the pattern-forming set `flock`, `ring`,
`heart`, `orbit`.

## Spawn-zone exclusion

Spawn positions are drawn from the master seeded RNG and are now required to be
both unoccupied **and outside the current challenge's survival zone** (evaluated
with the same survival predicate the agent will face). Without this, agents could
spawn already inside the safe zone and "survive" with zero behavior, inflating
early survival. The exclusion makes gen-0 numbers reflect movement, not luck. It
slightly lowers `north-band`'s legacy baseline — an accepted change.

## Baselines

Seeded headless run, fixed seed 42, default population (1000), 200 steps/gen,
150 generations (`--seed 42 --generations 151 --metrics ...`). Survival rate at
selected generations, and the max reached over the run:

| challenge    | gen 0 | gen 25 | gen 50 | gen 100 | gen 150 | max   |
|--------------|-------|--------|--------|---------|---------|-------|
| north-band   | 16.4% | 94.5%  | 93.3%  | 93.8%   | 94.9%   | 97.0% |
| corners      | 14.0% | 32.9%  | 34.4%  | 35.0%   | 35.5%   | 36.0% |
| gauntlet     | 14.6% | 92.6%  | 94.3%  | 94.2%   | 92.8%   | 96.7% |
| moving-band  |  8.1% | 31.3%  | 34.9%  | 36.0%   | 51.2%   | 67.2% |
| enclosure    |  4.4% | 11.5%  | 16.0%  | 19.4%   | 25.1%   | 27.6% |
| flock        |  1.3% | 19.6%  | 32.8%  | 47.6%   | 56.4%   | 56.4% |
| ring         |  3.1% | 42.4%  | 51.6%  | 60.1%   | 61.8%   | 61.8% |
| heart        |  3.4% | 49.0%  | 57.2%  | 65.5%   | 70.2%   | 70.2% |
| orbit        |  2.1% | 15.2%  | 19.1%  | 22.3%   | 23.5%   | 24.5% |

Every challenge starts low and improves under selection — clear headroom, none
saturates by gen 10 or is flat-impossible. `north-band` and `gauntlet` sit near
15% at gen 0 because both are pure "reach the north edge" tasks whose gen-0 floor
is set by vertical diffusion, not by the obstacle; they then climb to ~95%. The
four pattern challenges start lower (1–4%): they select for convergence onto a
point, curve, or path, so at gen 0 — the population still scattered by random
genomes — only a few percent land on target. None is extinction-fragile (13–34
survivors at gen 0) and each climbs cleanly (e.g. `heart` 3 → 70%).

These numbers moved slightly from the earlier baselines because the input layer
widened 15 → 17 (the new self-position sensor, below). Widening changes gene
decoding (`source_id % 17`), so the same seed grows different brains: some curves
rose (`moving-band` 41 → 67% max, `enclosure` 17 → 28%), others shifted a point
or two at gen 0. The shape — start low, climb — is unchanged, and `north-band`
remains the default.

## The challenges

**north-band** (default) — Survive iff final `y > 108`. A single solid barrier
spans `x∈[10,118]` on row `y=108`, leaving both flanks open. Trivial: evolution
just learns "go north," so it's kept as the default and reaches ~95% quickly. The
legacy behavior, modulo the spawn-exclusion baseline shift.

**corners** — Survive iff the final position lies within radius 10 of any of the
four world corners. A 33×33 obstacle block fills the center (`x,y∈[48,80]`) to
break up clumping and give the directional sensors something to read. Selects for
dispersal and a committed directional preference; the four small targets keep the
ceiling modest (~35%).

**gauntlet** (flagship) — Survive iff final `y > 108`, but the barrier on row
`y=108` is full width (`x∈[0,127]`) with only two width-3 gaps, centered near
`x=40` and `x=88`. Unlike north-band there are no open flanks: the only way north
is through a gap, so an agent must locate and thread one using the directional
obstacle sensors. Climbs from ~17% to ~95% as the population learns to funnel
through the openings.

**moving-band** — Survive iff final `y` is within 10 of a band center that
oscillates with the generation number: `center = round(64 + 24·sin(0.25·gen))`,
a height-21 band whose midline sweeps `y∈[40,88]`. No obstacles. Because the
target moves between generations and agents have no absolute-position sense, a
hardcoded "go north" fails; it selects for robust centering and still climbs
across the run (6.8% → 33%).

**enclosure** — Survive iff the final position is strictly inside a central
walled box: thickness-1 walls form the square `x,y∈[44,84]` with a width-15
entrance in the bottom wall (`x∈[57,71]`, `y=44`); safe interior is `x,y∈[45,83]`.
Selects for seeking the box and threading the single entrance while following
walls off the directional sensors. Steady improvement (4.4% → 28%).

## Pattern-forming challenges

These four have no obstacles — they sculpt the swarm with the survival predicate
alone. All are enabled by the self-position sensor (inputs 15/16; see below): an
agent can only navigate to a coordinate, radius, or shape if it knows where it
is. Grid center is `(64, 64)`. Two are *static* zones (`ring`, `heart`) that the
viewer tints automatically; two are *dynamic* (`flock`, `orbit`) and resolved
once per generation in `Simulator::eval_survival`, so they carry no static tint.

**flock** — Survive iff the final position lies within `FLOCK_RADIUS = 20` of the
population's own **final** centroid (mean of all agents' end positions, computed
once at generation end). There is no fixed target — it is wherever the crowd
gathers — so it selects purely for convergence into one tight blob. The radius is
kept small (a tight blob) but ≥ ~18, the radius a disk needs to actually hold all
~1000 collision-separated agents, so full convergence stays feasible. Gen-0 is
low (1.3%) because a scattered random population is nowhere near its own centroid;
it climbs to ~56% as lineages learn to home on the crowd. Dynamic (no tint yet).

**ring** — Survive iff the distance from the grid center falls in the annulus
`[22, 42]`. A hollow ring: the empty hole at `r < 22` is what makes it a ring
rather than a disk, forcing agents to hold a target radius — neither collapsing
inward nor fleeing to the edge. Static zone (tinted). Climbs 3.1% → 62%.

**heart** — Survive iff the final position lies inside an embedded heart, defined
analytically (no bitmap asset) by the implicit curve
`(nx² + ny² − 1)³ − nx²·ny³ ≤ 0` over grid coordinates normalized about the center
with `y` flipped so the heart sits upright (`nx = (x−64)/34`, `ny = (64−y)/34`).
`HEART_SCALE = 34` sets the size; the inequality gives a *filled* heart, so the
swarm is sculpted into a solid heart shape. The showpiece. Static zone (tinted).
Climbs 3.4% → 70%.

**orbit** — Survive iff the agent has swept at least `ORBIT_THETA = 2.0` rad
(~115°) of **signed** angle around the center over the generation **and** ends in
the radius band `[14, 50]`. The angle is accumulated per agent in the serial
application phase (the signed angle between old and new position about the center,
each time it moves), reset to 0 at every spawn. Requiring net swept angle rewards
sustained rotation rather than a spiral in/out, selecting for a pinwheel. The
half-turn (π) of the original spec left gen-0 at ~0.2% (extinction-fragile), so Θ
was tuned to 2.0 rad — still clearly rotational — lifting gen-0 to 2.1% for a
robust climb to ~24%. The hardest of the four; dynamic (no tint yet).

## Sensor inputs

The input layer is 17 wide: `0` const-0, `1` const-1, `2` oscillator, `3` age,
`4` random, `5`/`6` the global crowd-centroid y/x, `7–14` the eight directional
obstacle sensors, `15`/`16` the self-position sensors.

**Directional obstacle sensors (7–14).** Eight "can I move here" sensors: each
reads `1.0` when the neighboring cell in that direction is open and `0.0` when a
wall (or another agent) blocks it. They fire against real challenge geometry —
e.g. under `gauntlet`, an agent just below the barrier reads its north sensor as
`0.0` over a solid segment and `1.0` under a gap (covered by unit tests).
`gauntlet` and `enclosure` depend on these for navigation. Computed lazily — only
for the ids a given brain actually wires up.

**Self-position sensors (15/16).** `input[15] = x / 127`, `input[16] = y / 127` —
the agent's own normalized coordinates, filled unconditionally every step. This
is the enabling change for the pattern challenges: previously an agent could
sense the crowd's absolute centroid but **not its own position**, so it could
only follow the crowd or walls, never navigate to a coordinate. With self-position
it can hold a radius (`ring`), sit inside a shape (`heart`), converge on the crowd
(`flock`), or circle the center (`orbit`). Widening the layer from 15 to 17 shifts
gene decoding (`source_id % 17`), which is why the existing baselines above moved
slightly.
