use rand::RngExt;
use rand_chacha::ChaCha8Rng;

pub mod binary_util;

/// Input-vector width of a challenge [`Agent`]'s brain (see the `INPUTS` comment
/// in `lib.rs`). The [`Brain`] is width-parametric so other trophic layers (the
/// eco herbivores) can reuse it with their own sensor count; this const pins the
/// challenge agent's layout so its behavior is byte-identical to before.
pub(crate) const AGENT_INPUTS: usize = 17;

/// Which neuron dynamics a [`Brain`] runs. Both variants decode the **same**
/// sparse connection genome — they differ only in how neurons update each step:
///
/// - `Feedforward`: neurons are reset to 0 every step, then one forward pass with
///   `tanh` activation. Recurrent (inner→inner) genes still decode, but the
///   per-step reset neutralizes them, so the brain is **memoryless**. This is the
///   original behavior; a `Feedforward` brain is byte-identical to before.
/// - `Ctrnn`: neurons are continuous-time leaky integrators whose state
///   **persists across steps** (reset only on birth). The same recurrent genes
///   now become functional memory — a neuron can hold a past input after it is
///   gone. Per-neuron time constants; see [`Brain::step_ctrnn`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrainKind {
    #[default]
    Feedforward,
    Ctrnn,
}

/// CTRNN Euler timestep. One sim tick = one integration step (`dt = 1`), so a
/// neuron's time constant `tau` is measured directly in ticks.
const CTRNN_DT: f32 = 1.0;
/// Inner-neuron time constants are spread **geometrically** across this range, so
/// the brain owns both fast (near-reactive) and slow (long-memory) integrators. A
/// neuron with `tau = t` retains ~`(1 - dt/t)` of its state each tick — an
/// exponential memory of ~`t` ticks. This spread is the substrate the recurrent
/// genes exploit; evolution picks which timescales to wire into loops.
const CTRNN_TAU_INNER_MIN: f32 = 2.0;
const CTRNN_TAU_INNER_MAX: f32 = 20.0;
/// Output neurons integrate fast (`tau = 1` ⇒ an instantaneous read-out of their
/// input current), so movement tracks the recurrent inner state without extra
/// lag — the memory lives in the inner layer, the outputs just read it out.
const CTRNN_TAU_OUTPUT: f32 = 1.0;
/// Per-neuron bias (the operating point of the `tanh`). Zero keeps the resting
/// state at 0 and the genome unchanged; evolution shapes the dynamics through the
/// connection weights (the sparse-genome search space), not per-neuron biases.
const CTRNN_BIAS: f32 = 0.0;

/// The per-creature component bundle: everything about an agent *except* its
/// spatial position, which lives in a separate `Position` component (position is
/// queried far more often than the brain, so keeping it a small standalone
/// component keeps the hot spatial queries cache-friendly). Held as one hecs
/// component so the brain/genome move together.
pub struct Agent {
    pub genome: Vec<u32>,
    /// Lineage id: the index of this agent's founding ancestor at the initial
    /// generation (or extinction reseed). Inherited verbatim by children, so all
    /// descendants of one founder share an id. Used only for viewer coloring.
    pub lineage: u32,
    /// Signed angle (radians) accumulated around the grid center over this
    /// generation's steps — the running sum of the angle swept between old and
    /// new position each time the agent moves. Starts at 0 for every freshly
    /// constructed agent (so it resets at spawn / generation start) and is only
    /// advanced by the `orbit` challenge's serial application phase. Not
    /// serialized; the genome fully determines the brain, this is transient path
    /// state.
    accumulated_angle: f32,
    brain: Brain,
    rgba: [u8; 4],
    amt_inners: u8,
    /// Feed-forward or CTRNN dynamics — inherited verbatim by children. Held so a
    /// child rebuilds its brain with the same dynamics as its parent.
    kind: BrainKind,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            genome: self.genome.clone(),
            lineage: self.lineage,
            accumulated_angle: self.accumulated_angle,
            brain: self.brain.clone(),
            rgba: self.get_rgba(),
            amt_inners: self.amt_inners,
            kind: self.kind,
        }
    }
}

impl Agent {
    pub fn new(genome: &[u32], amt_inners: u8, lineage: u32, kind: BrainKind) -> Agent {
        Agent {
            genome: genome.to_vec(),
            lineage,
            accumulated_angle: 0.0,
            brain: Brain::from(genome.to_vec(), AGENT_INPUTS, amt_inners, kind),
            rgba: Agent::calc_rgba(genome),
            amt_inners,
            kind,
        }
    }

    /// Which brain dynamics this agent runs (feed-forward or CTRNN).
    pub fn brain_kind(&self) -> BrainKind {
        self.kind
    }

    /// Number of inner neurons the brain was built with (needed to build a
    /// child's brain at the same width).
    pub fn amt_inners(&self) -> u8 {
        self.amt_inners
    }

    /// Read-only view of the decoded brain connections (for the viewer's brain
    /// inspector). No serialization; purely for live introspection.
    pub fn brain_connections(&self) -> &[Connection] {
        &self.brain.connections
    }

    /// Read-only view of the brain's last neuron activations, indexed
    /// `[layer][id]` with layer 0 = inputs (17), 1 = inner, 2 = outputs (5).
    pub fn brain_neurons(&self) -> &[Vec<f32>] {
        &self.brain.neurons
    }

    /// Signed angle (radians) accumulated around the grid center this
    /// generation. Read by the `orbit` challenge's survival predicate.
    pub fn accumulated_angle(&self) -> f32 {
        self.accumulated_angle
    }

    /// Add `delta` radians to the accumulated orbit angle. Called only from the
    /// serial application phase (so it stays deterministic) when the agent moves.
    pub fn add_angle(&mut self, delta: f32) {
        self.accumulated_angle += delta;
    }

    pub fn get_used_inputs(&mut self) -> Vec<usize> {
        self.brain.get_used_inputs()
    }

    pub fn step(&mut self, input: Vec<f32>, rng: &mut ChaCha8Rng) -> (i32, i32) {
        self.brain.step(input, rng)
    }

    pub fn produce_child(&self, mutation_rate: f32, rng: &mut ChaCha8Rng) -> Agent {
        let genome = mutate_genome(&self.genome, mutation_rate, rng);
        Agent::new(&genome, self.amt_inners, self.lineage, self.kind)
    }

    pub fn get_rgba(&self) -> [u8; 4] {
        self.rgba
    }

    pub fn calc_rgba(genome: &[u32]) -> [u8; 4] {
        let mut av: f32 = 0.0;
        genome.iter().for_each(|x| av += *x as f32);
        let bits = av.to_bits();

        [
            binary_util::get_segment(&bits, 0..=7) as u8,
            binary_util::get_segment(&bits, 8..=15) as u8,
            binary_util::get_segment(&bits, 16..=23) as u8,
            255
        ]
    }

}

/// Flip each bit of the genome independently with probability
/// `mutation_rate` (matching the original per-bit trial), but instead of
/// rolling the RNG once per bit we sample the gaps between flipped bits
/// from a geometric distribution. For the default rate (0.001) this turns
/// ~8192 RNG calls per agent into ~8, with the identical flip distribution.
pub fn mutate_genome(genome: &[u32], mutation_rate: f32, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let mut out: Vec<u32> = genome.to_vec();
    let p = mutation_rate as f64;
    if p <= 0.0 {
        return out;
    }

    let total_bits = out.len() * 32;
    let ln_1mp = (1.0 - p).ln();
    let mut pos: usize = 0;

    loop {
        // Number of bits that are *not* flipped before the next flip:
        // geometric with success probability p.
        let u: f64 = rng.random::<f64>();
        let skip_f = u.ln() / ln_1mp;
        if !skip_f.is_finite() {
            break;
        }
        pos = pos.saturating_add(skip_f.floor() as usize);
        if pos >= total_bits {
            break;
        }
        out[pos / 32] = binary_util::flip(&out[pos / 32], pos % 32);
        pos += 1;
    }

    out
}

/// A feed-forward brain decoded from the sparse connection genome. Width of the
/// input layer is a construction parameter (`num_inputs`), so the challenge
/// [`Agent`] (17 inputs) and the eco herbivores (their own forager sensorium)
/// share the exact same decode/step machinery over different sensor counts. The
/// genome decodes source/sink ids *modulo* each layer's length, so any width is
/// valid. Kept `pub(crate)` so `crate::eco` can build one directly.
pub(crate) struct Brain {
    genome: Vec<u32>,
    move_activation: f32,
    used_input_ids: Vec<usize>,
    connections: Vec<Connection>,
    neurons: Vec<Vec<f32>>,
    move_vec: Vec<(i32, i32)>,
    /// Neuron dynamics: feed-forward (reset each step) or CTRNN (persistent
    /// state). Selects the `step` path; the decoded genome is identical either way.
    kind: BrainKind,
    /// CTRNN persistent state `y_i` per neuron, indexed `[layer][id]` exactly like
    /// `neurons` (layer 0 = inputs, unused; 1 = inner; 2 = output). Carried across
    /// ticks — this vector *is* the memory — and reset to 0 only on birth (a fresh
    /// `Brain`). Empty for a feed-forward brain (which allocates none of this).
    state: Vec<Vec<f32>>,
    /// CTRNN per-neuron time constants, same `[layer][id]` shape as `state`. Empty
    /// for a feed-forward brain.
    tau: Vec<Vec<f32>>,
    /// CTRNN per-neuron bias, same shape as `state`. Empty for feed-forward.
    bias: Vec<Vec<f32>>,
    /// CTRNN per-step scratch for the summed input current `I_i`, same shape as
    /// `state`. Held on the struct to avoid a per-step allocation. Empty for FF.
    currents: Vec<Vec<f32>>,
}

impl Clone for Brain {
    fn clone(&self) -> Self {
        Brain {
            genome: self.genome.clone(),
            move_activation: self.move_activation,
            used_input_ids: self.used_input_ids.clone(),
            connections: self.connections.clone(),
            neurons: self.neurons.clone(),
            move_vec: self.move_vec.clone(),
            kind: self.kind,
            state: self.state.clone(),
            tau: self.tau.clone(),
            bias: self.bias.clone(),
            currents: self.currents.clone(),
        }
    }
}

impl Brain {
    pub(crate) fn from(genome: Vec<u32>, num_inputs: usize, amt_inners: u8, kind: BrainKind) -> Brain {
        let mut out: Brain = Brain {
            genome,
            move_activation: 0.0,
            used_input_ids: Vec::new(),
            connections: Vec::new(),
            neurons: vec![
                vec![0.0; num_inputs],
                vec![0.0; amt_inners as usize],
                vec![0.0; 5]
            ],
            move_vec: vec![
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
            ],
            kind,
            state: Vec::new(),
            tau: Vec::new(),
            bias: Vec::new(),
            currents: Vec::new(),
        };

        out.generate_connections();

        if kind == BrainKind::Ctrnn {
            out.init_ctrnn();
        }

        out
    }

    /// Allocate + initialize the CTRNN state buffers (once, at construction, for a
    /// `Ctrnn` brain). State starts at 0 — a fresh brain has no memory. Per-neuron
    /// `tau` is a geometric spread over the inner layer (fast→slow integrators)
    /// and `CTRNN_TAU_OUTPUT` for the outputs; `bias` is `CTRNN_BIAS`. Every buffer
    /// mirrors `neurons`' `[layer][id]` shape (layer 0 present but unused), so a
    /// connection's sink/source layer tags index them directly.
    fn init_ctrnn(&mut self) {
        self.state = self.neurons.iter().map(|l| vec![0.0; l.len()]).collect();
        self.currents = self.neurons.iter().map(|l| vec![0.0; l.len()]).collect();
        self.bias = self.neurons.iter().map(|l| vec![CTRNN_BIAS; l.len()]).collect();
        // tau defaults to 1.0; the inner layer gets the geometric spread and the
        // output layer the fast read-out constant.
        self.tau = self.neurons.iter().map(|l| vec![1.0; l.len()]).collect();
        let inner_n = self.neurons[1].len();
        for (i, t) in self.tau[1].iter_mut().enumerate() {
            *t = if inner_n <= 1 {
                CTRNN_TAU_INNER_MIN
            } else {
                let f = i as f32 / (inner_n - 1) as f32;
                CTRNN_TAU_INNER_MIN * (CTRNN_TAU_INNER_MAX / CTRNN_TAU_INNER_MIN).powf(f)
            };
        }
        for t in &mut self.tau[2] {
            *t = CTRNN_TAU_OUTPUT;
        }
    }

    /// Read-only view of the decoded connections (for the viewer's brain
    /// inspector on any trophic layer that reuses this brain, e.g. the eco
    /// herbivores). Crate-internal because `Connection` layout is an
    /// implementation detail exposed only through the sim's data accessors.
    pub(crate) fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Read-only view of the last neuron activations, indexed `[layer][id]`
    /// (layer 0 = inputs, 1 = inner, 2 = outputs). Valid after a `step`; read
    /// live by the signal-flow inspector.
    pub(crate) fn neurons(&self) -> &[Vec<f32>] {
        &self.neurons
    }

    pub(crate) fn step(&mut self, input: Vec<f32>, rng: &mut ChaCha8Rng) -> (i32, i32) {
        if self.kind == BrainKind::Ctrnn {
            return self.step_ctrnn(input, rng);
        }
        // Feed-forward: reset, one forward pass, read the outputs. Unchanged from
        // the original (byte-identical), only the motor decode is now shared.
        self.reset_all();
        self.neurons[0] = input;
        self.calculate_all();
        self.decode_movement(rng)
    }

    /// One continuous-time recurrent update (Euler, `dt = CTRNN_DT`). For each
    /// non-input neuron `i`:
    ///
    /// > `y_i += (dt / tau_i) * ( -y_i + Σ_j w_ij · a_j )`
    ///
    /// where `a_j` is the *activation* of source `j` — the raw input for an input
    /// neuron, or `tanh(y_j + bias_j)` for an inner neuron read at its **previous**
    /// step's value (the recurrent signal). All currents are summed from the
    /// connections *before* any state is updated (a synchronous update), so an
    /// inner→inner loop feeds back last tick's activation — the persistent memory
    /// a feed-forward brain lacks. Output activations are then read by the shared
    /// motor decoder. State persists across calls; only birth (a fresh `Brain`)
    /// clears it.
    fn step_ctrnn(&mut self, input: Vec<f32>, rng: &mut ChaCha8Rng) -> (i32, i32) {
        self.neurons[0] = input;

        // Sum the input current into each inner/output neuron from the *current*
        // activations: inputs were just set; inner/output hold last step's tanh.
        for layer in 1..self.currents.len() {
            for v in &mut self.currents[layer] {
                *v = 0.0;
            }
        }
        for c in &self.connections {
            let a_src = self.neurons[c.source_type as usize][c.source_id as usize];
            self.currents[c.sink_type as usize][c.sink_id as usize] += c.weight * a_src;
        }

        // Euler-integrate the leaky state, then refresh activations
        // `a_i = tanh(y_i + bias_i)` for the inner + output layers.
        for layer in 1..self.state.len() {
            for i in 0..self.state[layer].len() {
                let y = self.state[layer][i];
                let dy = (CTRNN_DT / self.tau[layer][i]) * (-y + self.currents[layer][i]);
                let y_new = y + dy;
                self.state[layer][i] = y_new;
                self.neurons[layer][i] = ((y_new + self.bias[layer][i]) as f64).tanh() as f32;
            }
        }

        self.decode_movement(rng)
    }

    /// Read the 5 movement outputs (`neurons[2]`) into a translation, shared by
    /// both brain kinds so their motor decoding is identical: output 0 (over the
    /// `move_activation` threshold) adds a random ±1 jitter, outputs 1..5 add their
    /// cardinal direction, the sum clamped to a unit step. The single conditional
    /// RNG draw (only when output 0 fires) matches the original feed-forward decode
    /// exactly, so a feed-forward brain's behavior — and its RNG-stream
    /// consumption — is byte-identical to before.
    fn decode_movement(&self, rng: &mut ChaCha8Rng) -> (i32, i32) {
        let mut request = (0, 0);

        if self.neurons[2][0] > self.move_activation {
            request = (request.0 + rng.random_range(-1..=1), request.1 + rng.random_range(-1..=1));
        }

        let move_vec: Vec<(i32, i32)> = vec![
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ];

        for i in 1..self.neurons[2].len() {
            if self.neurons[2][i] > self.move_activation {
                request = (request.0 + move_vec[i - 1].0, request.1 + move_vec[i - 1].1);
            }
        }

        request.clamp((-1, -1), (1, 1))
    }

    fn get_used_inputs(&mut self) -> Vec<usize> {
        self.used_input_ids.clone()
    }

    fn reset_all(&mut self) {
        for i in 1..self.neurons.len() {
            for j in 0..self.neurons[i].len() {
                self.neurons[i][j] = 0.0;
            }
        }
    }

    fn calculate_all(&mut self) {
        for i in 0..self.connections.len() {
            self.calculate(i);
        }

        for i in 0..self.neurons[2].len() {
            self.neurons[2][i] = (self.neurons[2][i] as f64).tanh() as f32
        }
    }

    fn calculate(&mut self, index: usize) {
        let connection: &Connection = &self.connections[index];

        if connection.source_type != 0 {
            self.neurons[connection.source_type as usize][connection.source_id as usize] = (self.neurons[connection.source_type as usize][connection.source_id as usize] as f64).tanh() as f32;
        }

        self.neurons[connection.sink_type as usize][connection.sink_id as usize] += connection.weight * self.neurons[connection.source_type as usize][connection.source_id as usize];
    }

    fn generate_connections(&mut self) {
        for i in 0..self.genome.len() {
            self.generate_connection_from_genome_segment(i);
        }

        self.connections.sort_by_key(|a| a.sink_id);
    }

    fn generate_connection_from_genome_segment(&mut self, index: usize) {
        let dec: u32 = self.genome[index];

        let source_type: u8 = binary_util::get_segment(&dec, /*&(0b10000000000000000000000000000000 as u32)*/ 0..=0) as u8;
        let source_id: u8 = binary_util::get_segment(&dec, /*&(0b01111111000000000000000000000000 as u32)*/ 1..=6) as u8 % self.neurons[source_type as usize].len() as u8;
        let sink_type: u8 = binary_util::get_segment(&dec, /*&(0b10000000100000000000000000000000 as u32)*/ 7..=7) as u8 + 1;
        let sink_id: u8 = binary_util::get_segment(&dec, /*&(0b10000000011111110000000000000000 as u32)*/8..=15) as u8 % self.neurons[sink_type as usize].len() as u8;
        //println!("{}", self.neurons[sink_type as usize].len());
        let magnitude = binary_util::get_segment(&dec, 17..=31) as f32 / 16000.0;
        let weight = if binary_util::get_segment(&dec, 16..=16) == 1 {
            magnitude
        } else {
            -magnitude
        };

        if source_type == 0 && source_id > 6
            && !self.used_input_ids.contains(&(source_id as usize)) {
                self.used_input_ids.push(source_id as usize);
            }


        //println!("{}-{} {}-{} {}", source_type, source_id, sink_type, sink_id, weight);

        self.connections.push(Connection{
            source_type,
            source_id,
            sink_type,
            sink_id,
            weight
        })
    }
}

/// A decoded synaptic connection. `source_type`/`sink_type` are layer tags:
/// source 0 = input, 1 = inner; sink 1 = inner, 2 = output. Fields are public so
/// the viewer's brain inspector can read the wiring; the struct is never
/// serialized (the brain is a pure function of the genome).
pub struct Connection {
    pub source_type: u8,
    pub source_id: u8,
    pub sink_type: u8,
    pub sink_id: u8,
    pub weight: f32
}

impl Clone for Connection {
    fn clone(&self) -> Self {
        Connection {
            source_type: self.source_type,
            source_id: self.source_id,
            sink_type: self.sink_type,
            sink_id: self.sink_id,
            weight: self.weight
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn get_segment_extracts_bit_range() {
        // 0b1011_0100 = 180; bits [2..=5] read low-first = 0b1101 = 13.
        assert_eq!(binary_util::get_segment(&0b1011_0100u32, 2..=5), 13);
        assert_eq!(binary_util::get_segment(&0xFFFF_FFFFu32, 0..=0), 1);
        assert_eq!(binary_util::get_segment(&0u32, 0..=31), 0);
    }

    #[test]
    fn flip_toggles_single_bit() {
        assert_eq!(binary_util::flip(&0b0000u32, 2), 0b0100);
        assert_eq!(binary_util::flip(&0b0100u32, 2), 0b0000);
        assert_eq!(binary_util::flip(&0u32, 31), 1 << 31);
    }

    #[test]
    fn gene_decodes_to_expected_connection() {
        // Packing (bit 0 = LSB): [source_type:1][source_id:6][sink_type:1]
        // [sink_id:8][sign:1][weight:15].
        let dec: u32 = (5 << 1)                  // source_id = 5  (5 % 17 = 5)
            | (1 << 7)               // sink_type raw = 1 -> decoded 2 (output, len 5)
            | (3 << 8)               // sink_id = 3   (3 % 5 = 3)
            | (1 << 16)              // sign = 1 -> positive
            | (16000u32 << 17);      // weight raw = 16000 -> 16000/16000 = 1.0

        let brain = Brain::from(vec![dec], 17, 10, BrainKind::Feedforward);
        assert_eq!(brain.connections.len(), 1);
        let c = &brain.connections[0];
        assert_eq!(c.source_type, 0);
        assert_eq!(c.source_id, 5);
        assert_eq!(c.sink_type, 2);
        assert_eq!(c.sink_id, 3);
        assert_eq!(c.weight, 1.0);
    }

    /// Two hand-built genes: `input0 → inner0` (+1) and `inner0 → output0` (+1).
    /// Packing per `gene_decodes_to_expected_connection`.
    fn mem_genome() -> Vec<u32> {
        // input0 -> inner0 : source_type 0, source_id 0, sink_type raw 0 (inner),
        // sink_id 0, sign 1, magnitude 16000 (weight +1.0).
        let in_inner = (1u32 << 16) | (16000u32 << 17);
        // inner0 -> output0 : source_type 1, source_id 0, sink_type raw 1 (output),
        // sink_id 0, sign 1, magnitude 16000.
        let inner_out = 1u32 | (1u32 << 7) | (1u32 << 16) | (16000u32 << 17);
        vec![in_inner, inner_out]
    }

    #[test]
    fn ctrnn_remembers_a_past_input_but_feedforward_does_not() {
        // The headline CTRNN property: an inner neuron holds a memory of a past
        // input after that input is removed. The feed-forward brain resets each
        // step, so the same neuron is back to 0 once the drive is gone.
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut ctrnn = Brain::from(mem_genome(), 4, 4, BrainKind::Ctrnn);
        ctrnn.step(vec![1.0, 0.0, 0.0, 0.0], &mut rng); // drive the input high
        let charged = ctrnn.neurons()[1][0];
        assert!(charged.abs() > 0.1, "CTRNN inner should charge from the input: {charged}");
        ctrnn.step(vec![0.0, 0.0, 0.0, 0.0], &mut rng); // remove the input
        let remembered = ctrnn.neurons()[1][0];
        assert!(
            remembered.abs() > 0.05,
            "CTRNN inner must retain state after the input is removed: {remembered}"
        );

        let mut ff = Brain::from(mem_genome(), 4, 4, BrainKind::Feedforward);
        ff.step(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
        ff.step(vec![0.0, 0.0, 0.0, 0.0], &mut rng);
        assert_eq!(
            ff.neurons()[1][0],
            0.0,
            "feed-forward inner resets to 0 once the input is gone (no memory)"
        );
    }

    #[test]
    fn ctrnn_integrates_repeated_input_feedforward_is_static() {
        // A leaky integrator builds up over repeated identical input (integration
        // across steps); the memoryless feed-forward brain produces the same inner
        // value every identical step.
        let genome = vec![(1u32 << 16) | (16000u32 << 17)]; // input0 -> inner0, +1
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut ctrnn = Brain::from(genome.clone(), 4, 4, BrainKind::Ctrnn);
        let mut prev = 0.0f32;
        let mut rose = 0;
        for _ in 0..5 {
            ctrnn.step(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
            let a = ctrnn.neurons()[1][0];
            if a > prev + 1e-6 {
                rose += 1;
            }
            prev = a;
        }
        assert!(rose >= 3, "CTRNN inner should build up over repeated input (integration)");

        let mut ff = Brain::from(genome, 4, 4, BrainKind::Feedforward);
        ff.step(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
        let a1 = ff.neurons()[1][0];
        ff.step(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
        let a2 = ff.neurons()[1][0];
        assert_eq!(a1, a2, "feed-forward inner is identical every identical step (no integration)");
    }

    #[test]
    fn ctrnn_tau_spread_is_monotone_fast_to_slow() {
        // The inner layer's time constants are a geometric spread from the fast
        // (reactive) end to the slow (long-memory) end — the multi-timescale
        // substrate. Build a wide brain and check tau rises across the inner layer.
        let brain = Brain::from(vec![0u32; 4], 8, 16, BrainKind::Ctrnn);
        let tau = &brain.tau[1];
        assert!((tau[0] - CTRNN_TAU_INNER_MIN).abs() < 1e-4, "first inner is the fastest");
        assert!(
            (tau[tau.len() - 1] - CTRNN_TAU_INNER_MAX).abs() < 1e-3,
            "last inner is the slowest"
        );
        assert!(tau.windows(2).all(|w| w[1] >= w[0]), "tau must be non-decreasing across inners");
    }

    #[test]
    fn mutation_frequency_matches_rate() {
        // Geometric-skip mutation must reproduce the nominal per-bit flip rate.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let rate = 0.01f32;
        let genome = vec![0u32; 256]; // 8192 bits
        let genomes = 200;
        let mut flips: u64 = 0;
        for _ in 0..genomes {
            let mutated = mutate_genome(&genome, rate, &mut rng);
            flips += mutated.iter().map(|g| g.count_ones() as u64).sum::<u64>();
        }
        let total_bits = (genome.len() * 32 * genomes) as f64;
        let frac = flips as f64 / total_bits;
        assert!((frac - rate as f64).abs() < 0.0005, "flip fraction {frac} off nominal {rate}");
    }
}
