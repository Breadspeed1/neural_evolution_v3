use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use libm::tanh;
use rand::Rng;
use serde::{Serialize};

mod binary_util;

pub fn test() {
    let mut rand = rand::thread_rng();
    println!("{}", binary_util::get_segment(&rand.gen::<u32>(), 16..31))
}

#[derive(Serialize)]
pub struct Agent {
    pub id: u64,
    pub genome: Vec<u32>,
    brain: Brain
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            id: self.id,
            genome: self.genome.clone(),
            brain: self.brain.clone()
        }
    }
}

impl Agent {
    pub fn new(genome: &Vec<u32>, amt_inners: u8) -> Agent {
        let mut rng = rand::thread_rng();
        Agent {
            id: rng.gen::<u64>().clamp(1, u64::MAX),
            genome: genome.clone(),
            brain: Brain::from(genome.clone(), amt_inners)
        }
    }

    pub fn step(&mut self, input: Vec<f32>) -> (i32, i32) {
        self.brain.step(input)
    }

    pub fn produce_child(&mut self, mutation_rate: f32) -> Agent {
        let mut rng = rand::thread_rng();
        let genome = self.mutate_genome(mutation_rate);

        Agent {
            id: rng.gen::<u64>(),
            genome: genome.clone(),
            brain: Brain::from(genome, self.brain.neurons[1].len() as u8)
        }
    }

    fn mutate_genome(&mut self, mutation_rate: f32) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        let mut out: Vec<u32> = self.genome.clone();

        for i in 0..out.len() {
            for j in 0..31 {
                if rng.gen_range(0..(1.0/mutation_rate) as i32) == 0 {
                    out[i] = binary_util::flip(&out[i], j)
                }
            }
        }

        out
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Borrow<u64> for Agent {
    fn borrow(&self) -> &u64 {
        &self.id
    }
}

impl Eq for Agent {}

impl Hash for Agent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Serialize)]
struct Brain {
    genome: Vec<u32>,
    move_activation: f32,
    used_input_ids: Vec<u8>,
    connections: Vec<Connection>,
    neurons: Vec<Vec<f32>>
}

impl Clone for Brain {
    fn clone(&self) -> Self {
        Brain {
            genome: self.genome.clone(),
            move_activation: self.move_activation,
            used_input_ids: self.used_input_ids.clone(),
            connections: self.connections.clone(),
            neurons: self.neurons.clone()
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
                vec![0.0; 13],
                vec![0.0; amt_inners as usize],
                vec![0.0; 5]
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
            let mut rand = rand::thread_rng();
            request = (request.0 + rand.gen_range(-1..1), request.1 + rand.gen_range(-1..1));
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
            self.neurons[2][i] = tanh(self.neurons[2][i] as f64) as f32
        }
    }

    fn calculate(&mut self, index: usize) {
        let connection: &Connection = &self.connections[index];

        if connection.source_type != 0 {
            self.neurons[connection.source_type as usize][connection.source_id as usize] = libm::tanh(self.neurons[connection.source_type as usize][connection.source_id as usize] as f64) as f32;
        }

        self.neurons[connection.sink_type as usize][connection.sink_id as usize] += connection.weight * self.neurons[connection.source_type as usize][connection.source_id as usize];
    }

    fn generate_connections(&mut self) {
        for i in 0..self.genome.len() {
            self.generate_connection_from_genome_segment(i);
        }
    }

    fn generate_connection_from_genome_segment(&mut self, index: usize) {
        let dec: u32 = self.genome[index];

        let source_type: u8 = binary_util::get_segment(&dec, (0..0)) as u8;
        let source_id: u8 = binary_util::get_segment(&dec, (1..7)) as u8 % self.neurons[source_type as usize].len() as u8;
        let sink_type: u8 = binary_util::get_segment(&dec, (8..8)) as u8 + 1;
        let sink_id: u8 = binary_util::get_segment(&dec, (9..15)) as u8 % self.neurons[sink_type as usize].len() as u8;
        let mut weight: f32 = 0.0;

        if binary_util::get_segment(&dec, 16..16) == 1 {
            weight = binary_util::get_segment(&dec, (17..31)) as f32 / 8000.0;
        }
        weight = binary_util::get_segment(&dec, (17..31)) as f32 / -8000.0;

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
