use std::ops::Range;

pub fn get_segment(dec: &u32, range: Range<i32>) -> u32 {
    ((dec >> 31 - range.end) << range.start) >> range.start
}