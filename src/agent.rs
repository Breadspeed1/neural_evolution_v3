use rand::RngExt;
use rand_chacha::ChaCha8Rng;

pub mod binary_util;

/// Input-vector width of a challenge [`Agent`]'s brain (see the `INPUTS` comment
/// in `lib.rs`). The [`Brain`] is width-parametric so other trophic layers (the
/// eco herbivores) can reuse it with their own sensor count; this const pins the
/// challenge agent's layout so its behavior is byte-identical to before.
pub(crate) const AGENT_INPUTS: usize = 17;

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
        }
    }
}

impl Agent {
    pub fn new(genome: &[u32], amt_inners: u8, lineage: u32) -> Agent {
        Agent {
            genome: genome.to_vec(),
            lineage,
            accumulated_angle: 0.0,
            brain: Brain::from(genome.to_vec(), AGENT_INPUTS, amt_inners),
            rgba: Agent::calc_rgba(genome),
            amt_inners,
        }
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
        Agent::new(&genome, self.amt_inners, self.lineage)
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
    move_vec: Vec<(i32, i32)>
}

impl Clone for Brain {
    fn clone(&self) -> Self {
        Brain {
            genome: self.genome.clone(),
            move_activation: self.move_activation,
            used_input_ids: self.used_input_ids.clone(),
            connections: self.connections.clone(),
            neurons: self.neurons.clone(),
            move_vec: self.move_vec.clone()
        }
    }
}

impl Brain {
    pub(crate) fn from(genome: Vec<u32>, num_inputs: usize, amt_inners: u8) -> Brain {
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
            ]
        };

        out.generate_connections();

        out
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
        self.reset_all();
        self.neurons[0] = input;
        self.calculate_all();

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

        let brain = Brain::from(vec![dec], 17, 10);
        assert_eq!(brain.connections.len(), 1);
        let c = &brain.connections[0];
        assert_eq!(c.source_type, 0);
        assert_eq!(c.source_id, 5);
        assert_eq!(c.sink_type, 2);
        assert_eq!(c.sink_id, 3);
        assert_eq!(c.weight, 1.0);
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
