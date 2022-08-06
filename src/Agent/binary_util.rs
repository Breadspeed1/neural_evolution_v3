use std::ops::{Range, RangeBounds, RangeInclusive};
use libm::pow;

pub fn get_segment(dec: &u32, /*mask: &u32*/ range: RangeInclusive<i32>) -> u32 {
    //((dec >> 31 - range.end()) << 31 - (range.end() - range.start()) >> 31 - (range.end()) - range.start())
    let mut mask: u32 = 0;
    let start = *range.start();

    range.for_each(|x| { mask += (2 as u32).pow(x as u32) });

    (dec & mask)/(2 as u32).pow(start as u32)
}

pub fn flip(dec: &u32, index: usize) -> u32 {
    let mask: u32 = (2 as u32).pow(index as u32);
    dec ^ mask
}