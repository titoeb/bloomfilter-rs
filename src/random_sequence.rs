use rand::Rng;
use std::ops::Range;
// Get a random alement from a range.
///
/// # Arguments
///
/// * `range` - The range that should be sampled.
///
pub fn random_elem_from_range<T>(range: Range<T>) -> T
where
    T: std::cmp::PartialOrd + rand::distributions::uniform::SampleUniform + Copy,
{
    let first_element_in_range = range.start;
    if !range.is_empty() {
        rand::thread_rng().gen_range::<T, Range<T>>(range)
    } else {
        first_element_in_range
    }
}

// Get a random alement from a range.
///
/// # Arguments
///
/// * `range` - The range that should be sampled.
///
pub fn random_sequence_from_range<T>(range: Range<T>, sequence_length: usize) -> Vec<T>
where
    T: std::cmp::PartialOrd + rand::distributions::uniform::SampleUniform + Copy,
{
    (0..sequence_length)
        .map(|_| random_elem_from_range(range.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    mod get_elem_from_range {
        use super::super::random_elem_from_range;
        #[test]
        fn sample_int_range() {
            random_elem_from_range(0..10);
        }
        #[test]
        fn sample_float_range() {
            random_elem_from_range(0.0..1.0);
        }
        #[test]
        fn sample_empty_range() {
            assert_eq!(random_elem_from_range(0..0), 0);
        }
    }
    mod random_sequence_from_range {
        use super::super::random_sequence_from_range;
        #[test]
        fn get_range_with_ten_elements() {
            let sampled_range: Vec<i32> = random_sequence_from_range(0..0, 10);
            assert_eq!(sampled_range.len(), 10);
        }
    }
}
