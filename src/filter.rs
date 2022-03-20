use crate::random_sequence;
use core::iter::zip;
use fasthash_fork;
use rand::Rng;
use std::collections::HashSet;
use std::time;
/// Bloom Filter
pub struct Bloomfilter {
    /// Storing the 1s if activated by a hash funtion, zero otherwise.
    bitmap: Vec<u8>,
    /// The family of hash function thats will be used.
    seeds_for_hash_function: Vec<u64>,
    /// The hash function that will be used
    hash_function: fn(&[u8], u64) -> u64,
}

impl Bloomfilter {
    /// Get a new bloom filter with a bitmap of length `n_bits` and `n_hash_functions`
    /// hash functions.
    ///
    /// # Arguments
    ///
    /// * `n_bits`: How many bits should the bitmap of the bloomfilter have? More bits make
    ///     collision (False Positives)  more unlikely.
    ///
    /// * `n_hash_functions`: How many hash functions should be used?
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20
    /// );
    /// ```
    pub fn new(n_bits: usize, n_hash_functions: usize) -> Self {
        Bloomfilter {
            bitmap: vec![0; n_bits],
            seeds_for_hash_function: get_random_seeds(n_hash_functions),
            hash_function: |element, seed| fasthash_fork::xx::hash64_with_seed(element, seed),
        }
    }
    /// Generate a bloomfilter with non-random seeds.
    ///
    /// # Arguments
    ///
    /// * `n_bits`: How many bits should the bitmap of the bloomfilter have? More bits make
    ///     collision (False Positives)  more unlikely.
    ///
    /// * `seeds`: Provide custom seeds to parameterize the hash functions. Given a different seed, the hash function
    /// would hash the same objects to different hashes. You should not provide the same seeds twice.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let bloom_filter = filter::Bloomfilter::new_with_seeds(
    ///     10,
    ///     vec![1, 10, 100],
    /// );
    /// ```
    pub fn new_with_seeds(n_bits: usize, seeds: Vec<u64>) -> Self {
        Bloomfilter {
            bitmap: vec![0; n_bits],
            seeds_for_hash_function: seeds,
            hash_function: |element, seed| fasthash_fork::xx::hash64_with_seed(element, seed),
        }
    }
    /// Generate a bloomfilter with non-random seeds and a custom hash function.
    ///
    /// # Arguments
    ///
    /// * `n_bits`: How many bits should the bitmap of the bloomfilter have? More bits make
    ///     collision (False Positives)  more unlikely.
    ///
    /// * `seeds`: Provide custom seeds to parameterize the hash functions. Given a different seed, the hash function
    /// would hash the same objects to different hashes. You should not provide the same seeds twice.
    ///
    /// * `hash_function`: Specify a custom hash function. Hash functions should take in the bytes of the data and a seed to
    /// parmetrize the behaviour of the hash function. Given different hash values the function should show different behaviour.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let bloom_filter = filter::Bloomfilter::new_with_seeds_and_custom_hash_function(
    ///     10,
    ///     vec![1, 10, 100],
    ///     |bytes, seed| -> u64
    ///         {(bytes.iter().map(|elem| *elem as u64).sum::<u64>() + seed).into()}
    /// );
    /// ```
    pub fn new_with_seeds_and_custom_hash_function(
        n_bits: usize,
        seeds: Vec<u64>,
        hash_function: fn(&[u8], u64) -> u64,
    ) -> Self {
        Bloomfilter {
            bitmap: vec![0; n_bits],
            seeds_for_hash_function: seeds,
            hash_function,
        }
    }
    /// Create a random bloomfilter given its supposed characteristics.
    ///
    /// Give the number of expected entries and the acceptable false positive rates, we can create a
    /// hashfilter that will have the expected characteristic. If we use more entries then expected, the
    /// false positive rate might decrease.
    ///
    /// # Arguments:
    /// * expected_nubmber_of_entries: How many entries do you expect in the hash filter?
    /// * acceptable_false_positive_rate: What false positive rate is acceptabel for your use-case?
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let my_hash_filter = filter::Bloomfilter::from_specification(10, 0.5);
    /// ```
    pub fn from_specification(
        expected_number_of_entries: usize,
        acceptable_false_positive_rate: f64,
    ) -> Self {
        // Formulas taken from https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions.
        let optimal_number_of_bits = (-(expected_number_of_entries as f64
            * f64::ln(acceptable_false_positive_rate)))
            / (f64::powi(f64::ln(2.0), 2));
        let optimal_number_hash_functions = -f64::ln(acceptable_false_positive_rate) / f64::ln(2.0);
        Bloomfilter::new(
            optimal_number_of_bits.ceil() as usize,
            optimal_number_hash_functions.floor() as usize,
        )
    }
    /// How many bits does the bitmap of the filter have?
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20
    /// );
    /// println!("{}", bloom_filter.number_of_bits())
    /// ```

    pub fn number_of_bits(&self) -> usize {
        self.bitmap.len()
    }
    /// Insert an object into the bloomfilter. The object should be a string.
    ///
    /// If you map higher-level objects (struct ect.) to string, make sure that the representation does exactly
    /// identify each object (e.g. two different objects cannot map to the same string representation).
    ///
    /// # Arguments
    ///
    /// * `object`: The object to insert.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let mut bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20,
    /// );
    /// bloom_filter.insert("To be inserted");
    /// ```

    pub fn insert(&mut self, object_as_string: &str) {
        let n_bits = self.number_of_bits();
        for seed in &self.seeds_for_hash_function {
            self.bitmap[bytes_to_position_in_bitmap(
                *seed,
                object_as_string.as_bytes(),
                n_bits,
                self.hash_function,
            )] = 1;
        }
    }
    /// Insert an object as bytes into the bloomfilter.
    ///
    /// # Arguments
    ///
    /// * `object_as_bytes`: The object to insert.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let mut bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20,
    /// );
    /// bloom_filter.insert_object_as_bytes("To be inserted".as_bytes());
    /// ```
    pub fn insert_object_as_bytes(&mut self, object_as_bytes: &[u8]) {
        let n_bits = self.number_of_bits();
        for seed in &self.seeds_for_hash_function {
            self.bitmap
                [bytes_to_position_in_bitmap(*seed, object_as_bytes, n_bits, self.hash_function)] =
                1;
        }
    }
    /// Test that an object is in the hash filter. The objects needs to be string.
    ///
    /// The bloomfilter is probabilistic in nature. False negatives cannot occur (if `bloomfilter.contains`
    /// returns false, that will always be true), but false positives can occur.
    ///
    /// # Arguments
    ///
    /// * `object_as_string`: The object to check if it is contained in the bloomfilter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let mut bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20,
    /// );
    /// bloom_filter.contains("To be inserted");
    /// ```
    pub fn contains(&self, object_as_string: &str) -> bool {
        let mut bit_map_for_elem: Vec<u8> = vec![0; self.number_of_bits()];
        for seed in &self.seeds_for_hash_function {
            bit_map_for_elem[bytes_to_position_in_bitmap(
                *seed,
                object_as_string.as_bytes(),
                self.number_of_bits(),
                self.hash_function,
            )] = 1
        }

        all_elements_bitmap_a_also_in_b(&self.bitmap, &bit_map_for_elem).unwrap()
    }
    /// Test that an object as byte-array is in the hash filter.
    ///
    /// # Arguments
    ///
    /// * `object_as_bytes`: The object to check if it is contained in the bloomfilter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomfilter_rs::filter;
    ///
    /// let mut bloom_filter = filter::Bloomfilter::new(
    ///     10,
    ///     20,
    /// );
    /// bloom_filter.contains_object_as_bytes("To be inserted".as_bytes());
    /// ```
    pub fn contains_object_as_bytes(&self, object_as_bytes: &[u8]) -> bool {
        let mut bit_map_for_elem: Vec<u8> = vec![0; self.number_of_bits()];
        for seed in &self.seeds_for_hash_function {
            bit_map_for_elem[bytes_to_position_in_bitmap(
                *seed,
                object_as_bytes,
                self.number_of_bits(),
                self.hash_function,
            )] = 1
        }

        all_elements_bitmap_a_also_in_b(&self.bitmap, &bit_map_for_elem).unwrap()
    }
}

// Utilities used for the bloomfilter-implementation
/// Generate seeds that are used within the bloom filter
/// to parameterize hash-functions be have different behaviour.
/// # Arguments
///
/// * `numbers of seeds` - How many seeds should be generated? Should be the same number of needed hash
/// functions in the bloom filter.
///
fn get_random_seeds(number_of_seeds: usize) -> Vec<u64> {
    (0..number_of_seeds)
        .map(|_| rand::thread_rng().gen())
        .collect()
}
/// The if bitmap `contains_all` contains all entries that are in `contains_some`
/// # Arguments
///
/// * `contains_all` - Bitmap that should contain all entries.
/// * `contains_some` - Bitmap that should contain some entries, but only entries from `contains_all`.
///
fn all_elements_bitmap_a_also_in_b(
    contains_all: &Vec<u8>,
    contains_some: &Vec<u8>,
) -> Option<bool> {
    if contains_all.len() != contains_some.len() {
        return None;
    }
    for (elem_in_contains_all, elem_in_contains_some) in zip(contains_all, contains_some) {
        if *elem_in_contains_all == 0 && *elem_in_contains_some == 1 {
            return Some(false);
        }
    }
    Some(true)
}
/// Given an object of generic type (needs to implement `as_bytes`) hash the elem and map it to a bitmap
/// of length `number_of_bits`.
///
/// # Arguments:
/// * `seed`: Seed that should be used for the hash function.
/// * `bytes`: Bytes that should be added to the bloomfilter.
/// * `number_of_bits`: The number of bits in the bitmap.
/// * `hash_function`: The hash function that should be used for the mapping. It is assumed to take a reference to a bytes
/// array (the data that should be hashed) and a seed.
fn bytes_to_position_in_bitmap(
    seed: u64,
    bytes: &[u8],
    number_of_bits: usize,
    hash_function: fn(&[u8], u64) -> u64,
) -> usize {
    let elem_as_hash = hash_function(bytes, seed);
    (elem_as_hash % number_of_bits as u64) as usize
}
/// This function computes the actual false_positive rate for a bloomfilter.
/// The formula is taken from `https://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives`
pub fn compute_false_positive_rate_of_bloomfilter(
    n_hash_functions: usize,
    n_bits_in_bitmap: usize,
    number_of_expected_entries: usize,
) -> f64 {
    let inner_most_bracket = 1.0 as f64 - ((1.0 as f64) / (n_bits_in_bitmap as f64));
    let outer_bracket = 1.0 as f64
        - f64::powf(
            inner_most_bracket,
            (n_hash_functions * number_of_expected_entries) as f64,
        );
    f64::powf(outer_bracket, n_hash_functions as f64)
}
/// Create a hash-filter given the specification of the function and test
/// its false positives, false negatives rate and time to insert object /
/// test that objects are in there.
///
/// # Arguments:
/// * `n_samples_in_bloom_filter`: The number of samples that will be added to the bloomfilter.
/// * `n_samples_not_in_bloom_filter`: The number of samples that are definitly not in the bloomfilter but
/// that are used for testing if they the bloomfilter reports them as being in the filter.
/// * `n_bits_of_bloomfilter`: The number of bits that the bloomfilter has.
/// * `n_hash_functions_in_bloomfilter`: The number of hash functions that are used in the bloomfilter.
///
/// Returns:
/// * (false_positive_rate, false_negative_rate, insertion_time_per_1000_entries, contains_time_per_1000_entry, expected_false_positive_rate)
pub fn compute_mistakes_and_execution_time_bloomfilter(
    n_samples_in_bloom_filter: usize,
    n_samples_not_in_bloom_filter: usize,
    n_bits_of_bloomfilter: usize,
    n_hash_functions_in_bloomfilter: usize,
) -> (f64, f64, u64, u64, f64) {
    // Sampling 1000 sequence with 20 elements that will be added to bloom filer
    let mut sequences_in_bloom_filter: HashSet<Vec<i32>> =
        HashSet::with_capacity(n_samples_in_bloom_filter);
    while sequences_in_bloom_filter.len() < n_samples_in_bloom_filter {
        sequences_in_bloom_filter.insert(random_sequence::random_sequence_from_range(0..100, 20));
    }

    let mut bloom_filter = Bloomfilter::new(n_bits_of_bloomfilter, n_hash_functions_in_bloomfilter);

    let before = time::Instant::now();
    for sequence in &sequences_in_bloom_filter {
        bloom_filter.insert(format!("{:?}", sequence).as_ref())
    }
    let total_insertion_time = duration_to_micro_seconds(before.elapsed());

    // Test false negatives rate
    let mut false_negatives = 0.0;
    for sequence in &sequences_in_bloom_filter {
        if !bloom_filter.contains(format!("{:?}", sequence).as_ref()) {
            false_negatives += 1.0;
        }
    }

    // Test false positives rate.
    let mut false_positives = 0.0;
    let mut total_contains_time = 0;
    for _ in 0..n_samples_not_in_bloom_filter {
        let new_sequence = random_sequence::random_sequence_from_range(0..100, 20);
        if !sequences_in_bloom_filter.contains(&new_sequence) {
            let before = time::Instant::now();
            if bloom_filter.contains(format!("{:?}", new_sequence).as_ref()) {
                false_positives += 1.0;
            }
            total_contains_time += duration_to_micro_seconds(before.elapsed())
        }
    }
    let total_contains_time = total_contains_time;

    (
        false_positives / n_samples_not_in_bloom_filter as f64,
        false_negatives / n_samples_in_bloom_filter as f64,
        total_insertion_time / 1000,
        total_contains_time / 1000,
        compute_false_positive_rate_of_bloomfilter(
            n_hash_functions_in_bloomfilter,
            n_bits_of_bloomfilter,
            n_samples_in_bloom_filter,
        ),
    )
}
/// Repeate the `compute_mistakes_and_execution_time_bloomfilter` and average the
/// individual result to get more robust estimates.
pub fn repeat_compute_mistakes_and_execution_time_bloomfilter(
    n_samples_in_bloom_filter: usize,
    n_samples_not_in_bloom_filter: usize,
    n_bits_of_bloomfilter: usize,
    n_hash_functions_in_bloomfilter: usize,
    n_repititions: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut total_false_positives_rate = 0.0;
    let mut total_false_negatives_rate = 0.0;
    let mut total_insertion_time = 0.0;
    let mut total_contains_time = 0.0;
    let mut expected_false_positive_rate = 0.0;

    for _ in 0..n_repititions {
        let (
            false_positive_rate,
            false_negative_rate,
            _total_insertion_time,
            _total_contains_time,
            _expected_false_positive_rate,
        ) = compute_mistakes_and_execution_time_bloomfilter(
            n_samples_in_bloom_filter,
            n_samples_not_in_bloom_filter,
            n_bits_of_bloomfilter,
            n_hash_functions_in_bloomfilter,
        );
        total_false_positives_rate += false_positive_rate;
        total_false_negatives_rate += false_negative_rate;
        total_insertion_time += _total_insertion_time as f64;
        total_contains_time += _total_contains_time as f64;
        expected_false_positive_rate = _expected_false_positive_rate;
    }
    let n_repititions = n_repititions as f64;
    (
        total_false_positives_rate / n_repititions,
        total_false_negatives_rate / n_repititions,
        total_insertion_time / n_repititions,
        total_contains_time / n_repititions,
        expected_false_positive_rate,
    )
}

/// Compute the number of micro seconds in a `Duration`-object.
fn duration_to_micro_seconds(duration: time::Duration) -> u64 {
    let nanos = duration.subsec_nanos() as u64;
    (1000 * 1000 * 1000 * duration.as_secs() + nanos) / (1000)
}

#[cfg(test)]
pub mod tests {
    mod test_utilities {

        #[test]
        fn test_duration_to_micro_seconds() {
            use super::super::duration_to_micro_seconds;
            use std::time;
            assert_eq!(
                duration_to_micro_seconds(time::Duration::from_micros(100.5 as u64)),
                100.5 as u64
            );
        }

        #[test]
        /// Test `compute_false_positive_rate_of_bloomfilter` runs through with
        /// reasonable input values.
        fn test_compute_mistakes_and_execution_time_bloomfilter() {
            use super::super::compute_false_positive_rate_of_bloomfilter;

            _ = compute_false_positive_rate_of_bloomfilter(10, 10, 5);
        }
        #[test]
        /// Test `repeate_compute_false_positive_rate_of_bloomfilter` runs through with
        /// reasonable input values.
        fn test_repeat_compute_mistakes_and_execution_time_bloomfilter() {
            use super::super::repeat_compute_mistakes_and_execution_time_bloomfilter;

            _ = repeat_compute_mistakes_and_execution_time_bloomfilter(10, 10, 5, 10, 5);
        }
        #[test]
        fn test_get_random_seed() {
            // Simply test the function runs through, not much more that can be test
            // given it stochasticity.
            use super::super::get_random_seeds;
            let number_seeds_to_generate = 10;
            let generated_random_seeds = get_random_seeds(number_seeds_to_generate);
            assert_eq!(generated_random_seeds.len(), number_seeds_to_generate);
        }
        mod test_all_elements_bitmap_a_also_in_b {
            use super::super::super::all_elements_bitmap_a_also_in_b;
            #[test]
            fn different_length() {
                // Both bitmaps have different length, hence `None` is returned.
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![0, 0, 0, 0], &vec![0, 0, 0]),
                    None
                )
            }

            #[test]
            fn a_equals_b_both_empty() {
                // Both bitmaps are entry, therefore all elements in b are also in a
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![0, 0, 0, 0], &vec![0, 0, 0, 0]),
                    Some(true)
                )
            }

            #[test]
            fn a_equals_b_both_full() {
                // Both bitmaps are completely full, therefore all elements in b are also in a
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![1, 1, 1, 1], &vec![1, 1, 1, 1]),
                    Some(true)
                )
            }

            #[test]
            fn a_equals_b_mixed_entries() {
                // Some elements in bitmap, but the same for a and b therefore all elements in b
                // are also in a.
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![1, 0, 0, 1], &vec![1, 0, 0, 1]),
                    Some(true)
                )
            }

            #[test]
            fn a_more_entries_than_b() {
                // A has more entries in its bitmap, therefore all elements in b are also in a.
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![1, 1, 0, 1], &vec![1, 0, 0, 1]),
                    Some(true)
                )
            }

            #[test]
            fn b_more_entries_than_a() {
                // B has one element that, a does not have, therefore return Value is `Some(false)`
                assert_eq!(
                    all_elements_bitmap_a_also_in_b(&vec![1, 1, 0, 1], &vec![1, 0, 1, 1]),
                    Some(false)
                )
            }
        }
        mod test_elem_to_position_in_bitmap {
            use super::super::super::bytes_to_position_in_bitmap;
            /// Small hash function that sums the integer in the bytes array and
            /// adds the seed. This does not make sense from a hashing perspective,
            /// and is only used in testing to make the results predictable.
            fn hash_function_for_testing(bytes: &[u8], seed: u64) -> u64 {
                (bytes.iter().map(|elem| *elem as u64).sum::<u64>() + seed).into()
            }

            #[test]
            /// Test that the hash function just sums all entries and adds
            /// the seed.
            fn test_hash_function_for_testing() {
                assert_eq!(hash_function_for_testing(&vec![1, 2, 3], 5), 11);
            }
            #[test]
            /// Take adding seed to the entries in the vec gives you 3, which given a bit map of size three
            /// should result in entry 3.
            fn array_2_entries_sum_3() {
                assert_eq!(
                    bytes_to_position_in_bitmap(1, &vec![1, 1], 4, hash_function_for_testing),
                    3
                );
            }
            #[test]
            /// Take adding seed to the entries in the vec gives you 3, which given a bit map of size three
            fn array_3_entries_sum_3() {
                assert_eq!(
                    bytes_to_position_in_bitmap(3, &vec![1, 4, 9], 4, hash_function_for_testing),
                    1
                );
            }
            #[test]
            /// Add a string as bytes.
            fn test_with_string_to_bytes() {
                bytes_to_position_in_bitmap(3, "testing".as_bytes(), 4, hash_function_for_testing);
            }
            #[test]
            /// Add an integer as bytes.
            fn test_with_integer_as_bytes() {
                bytes_to_position_in_bitmap(3, &1_u32.to_ne_bytes(), 4, hash_function_for_testing);
            }
        }
        mod test_constructors {
            use super::super::super::Bloomfilter;

            #[test]
            /// Test `Bloomfilter::random`-constructor.
            fn test_random() {
                let bloom_filter = Bloomfilter::new(10, 20);
                assert_eq!(bloom_filter.bitmap.len(), 10);
                assert_eq!(bloom_filter.seeds_for_hash_function.len(), 20);
            }
            #[test]
            /// Test `Bloomfilter::new_with_seeds`-constructor.
            fn test_new_with_seeds() {
                let seeds: Vec<u64> = vec![1, 2, 3];
                let bloom_filter = Bloomfilter::new_with_seeds(10, seeds.clone());
                assert_eq!(bloom_filter.bitmap.len(), 10);
                assert_eq!(bloom_filter.seeds_for_hash_function, seeds);
            }
            #[test]
            /// Test `Bloomfilter::new_with_seeds_and_custom_hash_function`-constructor.
            fn test_new_with_seeds_and_custom_hash_function() {
                let seeds: Vec<u64> = vec![1, 2, 3];
                let hash_function = |_object_as_bytes: &[u8], _seed: u64| -> u64 { 2 };
                let bloom_filter = Bloomfilter::new_with_seeds_and_custom_hash_function(
                    10,
                    seeds.clone(),
                    hash_function,
                );
                assert_eq!(bloom_filter.bitmap.len(), 10);
                assert_eq!(bloom_filter.seeds_for_hash_function, seeds);
                // Cannot compare closures, therefore call it and check it returns 2.
                assert_eq!((bloom_filter.hash_function)(&[1, 0, 1,], 4), 2);
            }
        }
        mod test_getter_functions {
            use super::super::super::Bloomfilter;
            #[test]
            fn bloom_filter_with_5_bits() {
                let bloom_filter = Bloomfilter::new(5, 10);
                assert_eq!(bloom_filter.number_of_bits(), 5);
            }
        }

        mod test_insertion_and_retrivial {
            use super::super::super::Bloomfilter;

            fn sum_up_bitmap(bitmap: &[u8]) -> u64 {
                bitmap.iter().map(|indicator| *indicator as u64).sum()
            }
            #[test]
            fn test_insert_string() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                bloomfilter.insert("This is in there");

                // After the `insert`-function was called with 10 hash functions, the sum of the bitmap
                // has to be between 1..=10. (1 = 10 collisions, e.g. all hash functions mapped to the
                // same entry, 10 meaning all hash_function mapped this string to a different index
                // in the bitmap.)
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 10));

                bloomfilter.insert("This is also in here!");

                // After the second insertion there may be between 1 and 20 entries.
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 20));
            }
            #[test]
            fn test_insert_bytes() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                bloomfilter.insert_object_as_bytes(&1_u32.to_ne_bytes());

                // After the `insert_object_as_bytes`-function was called with 10 hash functions, the sum of the bitmap
                // has to be between 1..=10. (1 = 10 collisions, e.g. all hash functions mapped to the
                // same entry, 10 meaning all hash_function mapped this string to a different index
                // in the bitmap.)
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 10));

                bloomfilter.insert_object_as_bytes("This is also in here!".as_bytes());

                // After the second insertion there may be between 1 and 20 entries.
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 20));
            }
            #[test]
            /// Mix the two interfaces `insert`  and `insert_object_as_bytes` and test that the same
            /// properties still hold for the populated bitmap.
            fn test_insert_bytes_and_string() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                bloomfilter.insert_object_as_bytes(&1_u32.to_ne_bytes());

                // After the `insert_object_as_bytes`-function was called with 10 hash functions, the sum of the bitmap
                // has to be between 1..=10. (1 = 10 collisions, e.g. all hash functions mapped to the
                // same entry, 10 meaning all hash_function mapped this string to a different index
                // in the bitmap.)
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 10));

                bloomfilter.insert("This is also in here!");

                // After the second insertion there may be between 1 and 20 entries.
                let sum_of_entries: u64 = sum_up_bitmap(&bloomfilter.bitmap);
                assert!((sum_of_entries > 0) & (sum_of_entries <= 20));
            }
            #[test]
            fn test_contains() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                bloomfilter.insert("1");

                // Only with the single entry we can guarantee that the bloomfilter does not have
                // collision, e.g. two different objects get the same representation in the bitmap.
                assert!(bloomfilter.contains("1"));
                // Also test that a different object is not in here.
                assert!(!bloomfilter.contains("2"));
            }
            #[test]
            fn does_not_contain() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                for number_as_string in vec!["1", "2", "3", "4", "5", "6", "7", "8", "9"] {
                    bloomfilter.insert(number_as_string);
                }
                assert!(!bloomfilter.contains("10"));
            }
            #[test]
            fn test_contains_bytes() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                bloomfilter.insert_object_as_bytes("1".as_bytes());

                // Only with the single entry we can guarantee that the bloomfilter does not have
                // collision, e.g. two different objects get the same representation in the bitmap.
                assert!(bloomfilter.contains_object_as_bytes("1".as_bytes()));
                // Also test that a different object is not in here.
                assert!(!bloomfilter.contains_object_as_bytes("2".as_bytes()));
            }
            #[test]
            fn does_not_contain_bytes() {
                let mut bloomfilter = Bloomfilter::new(100, 10);
                for number_as_string in vec!["1", "2", "3", "4", "5", "6", "7", "8", "9"] {
                    bloomfilter.insert_object_as_bytes(number_as_string.as_bytes());
                }
                assert!(!bloomfilter.contains_object_as_bytes("10".as_bytes()));
            }
        }
        mod test_bloomfilter_from_specification {
            use super::super::super::super::random_sequence;
            use super::super::super::Bloomfilter;
            use std::collections;

            #[test]
            /// Test that `from_specification` with reasonable values
            /// achieves roughly the expected quality.
            fn test_from_specification() {
                let expected_false_positive_rate = 0.1;
                let number_of_entries_in_bloom_filter = 1000;
                let number_of_entries_not_in_bloom_filter = 2000;
                let margin_of_error = 0.2;

                let mut bloom_filter_to_test = Bloomfilter::from_specification(
                    number_of_entries_in_bloom_filter,
                    expected_false_positive_rate,
                );

                // Compute and insert 1000 sequences.
                let mut sequences_in_bloom_filter: collections::HashSet<Vec<i32>> =
                    collections::HashSet::with_capacity(number_of_entries_in_bloom_filter);
                while sequences_in_bloom_filter.len() < number_of_entries_in_bloom_filter {
                    sequences_in_bloom_filter
                        .insert(random_sequence::random_sequence_from_range(0..100, 20));
                }
                for sequence in &sequences_in_bloom_filter {
                    bloom_filter_to_test.insert(format!("{:?}", sequence).as_ref())
                }

                // Test false positives rate.
                let mut false_positives = 0.0;
                for _ in 0..number_of_entries_not_in_bloom_filter {
                    let new_sequence = random_sequence::random_sequence_from_range(0..100, 20);
                    if !sequences_in_bloom_filter.contains(&new_sequence) {
                        if bloom_filter_to_test.contains(format!("{:?}", new_sequence).as_ref()) {
                            false_positives += 1.0;
                        }
                    }
                }
                let actual_false_positive_rate =
                    false_positives / number_of_entries_not_in_bloom_filter as f64;
                println!("{}", actual_false_positive_rate);
                assert!(
                    (actual_false_positive_rate * (1.0 + margin_of_error)
                        > expected_false_positive_rate)
                        & (actual_false_positive_rate * (1.0 - margin_of_error)
                            < expected_false_positive_rate)
                )
            }
        }
    }
}
