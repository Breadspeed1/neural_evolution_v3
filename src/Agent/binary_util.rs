use std::ops::Range;
use libm::pow;

pub fn get_segment(dec: &u32, range: Range<i32>) -> u32 {
    ((dec >> 31 - range.end) << 31 - (range.end - range.start)) >> 31 - (range.end - range.start)

}

pub fn flip(dec: &u32, index: usize) -> u32 {
    let mask: u32 = (2 as u32).pow(index as u32);
    dec ^ mask
}