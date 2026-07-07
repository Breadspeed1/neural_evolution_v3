# Challenges

Each `--challenge` selects a *selection environment*: an obstacle layout stamped
into the 128×128 world at the start of every generation, plus a survival
predicate applied to each agent's final position. Both are pure functions of the
generation number (or constant), so runs stay bit-for-bit reproducible. Agents
that satisfy the predicate reproduce; the rest are culled. There is no energy,
food, or per-step scoring — only where you end up.

Names are kebab-case: `north-band` (default), `corners`, `gauntlet`,
`moving-band`, `enclosure`.

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
| north-band   | 18.7% | 93.9%  | 94.9%  | 93.7%   | 94.9%   | 96.8% |
| corners      | 14.5% | 31.6%  | 33.1%  | 34.3%   | 34.3%   | 35.7% |
| gauntlet     | 17.2% | 93.9%  | 93.4%  | 94.6%   | 94.4%   | 97.0% |
| moving-band  |  6.8% | 13.2%  | 14.3%  | 21.4%   | 33.0%   | 41.5% |
| enclosure    |  6.2% |  8.8%  | 10.1%  | 12.5%   | 16.1%   | 17.3% |

Every challenge starts low and improves under selection — clear headroom, none
saturates by gen 10 or is flat-impossible. (`north-band` and `gauntlet` sit near
18% at gen 0 because both are pure "reach the north edge" tasks whose gen-0 floor
is set by vertical diffusion, not by the obstacle; they then climb to ~95%.)

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
walls off the directional sensors. The hardest to learn — steady but slow
improvement (6.2% → 16%).

## Directional obstacle sensors

Inputs 7–14 are the eight directional "can I move here" sensors: each reads `1.0`
when the neighboring cell in that direction is open and `0.0` when a wall (or
another agent) blocks it. They fire against real challenge geometry — e.g. under
`gauntlet`, an agent just below the barrier reads its north sensor as `0.0` over a
solid segment and `1.0` under a gap (covered by unit tests). `gauntlet` and
`enclosure` depend on these for navigation; the sensors were confirmed working
and required no changes.
