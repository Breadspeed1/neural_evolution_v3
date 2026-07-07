use rand::RngExt;
use serde::{Serialize};

mod binary_util;

#[derive(Serialize)]
pub struct Agent {
    pub genome: Vec<u32>,
    pub pos: (u32, u32),
    brain: Brain,
    rgba: [u8; 4],
    amt_inners: u8
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            pos: self.pos,
            genome: self.genome.clone(),
            brain: self.brain.clone(),
            rgba: self.get_rgba(),
            amt_inners: self.amt_inners
        }
    }
}

impl Agent {
    pub fn new(genome: &Vec<u32>, amt_inners: u8, pos: (u32, u32)) -> Agent {
        Agent {
            pos,
            genome: genome.clone(),
            brain: Brain::from(genome.clone(), amt_inners),
            rgba: Agent::calc_rgba(genome),
            amt_inners
        }
    }

    pub fn set_pos(&mut self, pos: (u32, u32)) {
        self.pos = pos;
    }

    pub fn get_pos(&self) -> (u32, u32) {
        self.pos
    }

    pub fn get_used_inputs(&mut self) -> Vec<usize> {
        self.brain.get_used_inputs()
    }

    pub fn step(&mut self, input: Vec<f32>) -> (i32, i32) {
        self.brain.step(input)
    }

    pub fn produce_child(&self, mutation_rate: f32, pos: (u32, u32)) -> Agent {
        let genome = self.mutate_genome(mutation_rate);
        Agent::new(
            &genome,
            self.amt_inners,
            pos
        )
    }

    pub fn get_rgba(&self) -> [u8; 4] {
        self.rgba
    }

    pub fn calc_rgba(genome: &Vec<u32>) -> [u8; 4] {
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

    /// Flip each bit of the genome independently with probability
    /// `mutation_rate` (matching the original per-bit trial), but instead of
    /// rolling the RNG once per bit we sample the gaps between flipped bits
    /// from a geometric distribution. For the default rate (0.001) this turns
    /// ~8192 RNG calls per agent into ~8, with the identical flip distribution.
    fn mutate_genome(&self, mutation_rate: f32) -> Vec<u32> {
        let mut out: Vec<u32> = self.genome.clone();
        let p = mutation_rate as f64;
        if p <= 0.0 {
            return out;
        }

        let mut rng = rand::rng();
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
}

#[derive(Serialize)]
struct Brain {
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
    pub fn from(genome: Vec<u32>, amt_inners: u8) -> Brain {
        let mut out: Brain = Brain {
            genome,
            move_activation: 0.0,
            used_input_ids: Vec::new(),
            connections: Vec::new(),
            neurons: vec![
                vec![0.0; 15],
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

    fn step(&mut self, input: Vec<f32>) -> (i32, i32) {
        self.reset_all();
        self.neurons[0] = input;
        self.calculate_all();

        let mut request = (0, 0);

        if self.neurons[2][0] > self.move_activation {
            let mut rand = rand::rng();
            request = (request.0 + rand.random_range(-1..=1), request.1 + rand.random_range(-1..=1));
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

        self.connections.sort_by(|a, b| a.sink_id.cmp(&b.sink_id));
    }

    fn generate_connection_from_genome_segment(&mut self, index: usize) {
        let dec: u32 = self.genome[index];

        let source_type: u8 = binary_util::get_segment(&dec, /*&(0b10000000000000000000000000000000 as u32)*/ 0..=0) as u8;
        let source_id: u8 = binary_util::get_segment(&dec, /*&(0b01111111000000000000000000000000 as u32)*/ 1..=6) as u8 % self.neurons[source_type as usize].len() as u8;
        let sink_type: u8 = binary_util::get_segment(&dec, /*&(0b10000000100000000000000000000000 as u32)*/ 7..=7) as u8 + 1;
        let sink_id: u8 = binary_util::get_segment(&dec, /*&(0b10000000011111110000000000000000 as u32)*/8..=15) as u8 % self.neurons[sink_type as usize].len() as u8;
        //println!("{}", self.neurons[sink_type as usize].len());
        let weight: f32;

        if binary_util::get_segment(&dec, /*&(0b00000000000000001000000000000000 as u32)*/ 16..=16) == 1 {
            weight = binary_util::get_segment(&dec, /*&(0b00000000000000000111111111111111 as u32)*/ 17..=31) as f32 / 16000.0;
        }
        else {
            weight = binary_util::get_segment(&dec, /*&(0b00000000000000000111111111111111 as u32)*/ 17..=31) as f32 / -16000.0;
        }

        if source_type == 0 && source_id > 6 {
            if !self.used_input_ids.contains(&(source_id as usize)) {
                self.used_input_ids.push(source_id as usize);
            }
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

#[derive(Serialize)]
struct Connection {
    source_type: u8,
    source_id: u8,
    sink_type: u8,
    sink_id: u8,
    weight: f32
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
