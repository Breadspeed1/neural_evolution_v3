use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use rand::Rng;

mod binary_util;

pub fn test() {

}

pub struct Agent {
    id: u64,
    age: i32,
    position: (u32, u32),
    genome: Vec<u32>,
    brain: Brain
}

impl Agent {
    pub fn new(genome: &Vec<u32>, amt_inners: u8, position: (u32, u32)) -> Agent {
        let mut rng = rand::thread_rng();
        Agent {
            id: rng.gen::<u64>(),
            age: 0,
            position,
            genome: genome.clone(),
            brain: Brain::from(genome.clone(), amt_inners)
        }
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

struct Brain {
    genome: Vec<u32>,
    move_activation: f32,
    used_input_ids: Vec<u8>,
    connections: Vec<Connection>,
    neurons: Vec<Vec<f32>>
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
        self.neurons[0] = input;
        self.reset_all();
        self.calculate_all();
        (0, 0)
    }

    fn reset_all(&mut self) {
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons[i].len() {
                self.neurons[i][j] = 0.0;
            }
        }
    }

    fn calculate_all(&mut self) {
        for i in 0..self.connections.len() {
            self.calculate(i);
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
        let weight: f32 = binary_util::get_segment(&dec, (16..31)) as f32 / 8000.0;

        self.connections.push(Connection{
            source_type,
            source_id,
            sink_type,
            sink_id,
            weight
        })
    }
}

struct Connection {
    source_type: u8,
    source_id: u8,
    sink_type: u8,
    sink_id: u8,
    weight: f32
}
