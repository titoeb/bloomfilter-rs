use bloomfilter_rs::filter;
use cli_table::{self, WithTitle};

#[derive(cli_table::Table, Debug)]
struct ExperimentResults {
    #[table(title = "In Filter")]
    n_samples_in_bloomerfilter: usize,
    #[table(title = "Not in Filter")]
    n_samples_not_in_bloomfilter: usize,
    #[table(title = "FP rate.")]
    false_positive_rate: f64,
    #[table(title = "Expected FP rate.")]
    expected_false_positive_rate: f64,
    #[table(title = "FN rate.")]
    false_negative_rate: f64,
    #[table(title = "insert time")]
    total_insertion_time: f64,
    #[table(title = "contains time")]
    total_contains_time: f64,
}

fn main() {
    // Parameters used in the experiment.
    let n_samples_not_in_bloomfilter = 500;
    let n_bits_of_bloomfiler = 7500;
    let n_hash_functions_in_bloomfiler = 15;
    println!(
        "Using a bloom-filter with {} bits and {} hash-functions.",
        n_bits_of_bloomfiler, n_hash_functions_in_bloomfiler
    );

    let mut table = Vec::new();
    for n_elements_in_bloomfiler in (500..5000).step_by(500) {
        let (
            false_positive_rate,
            false_negative_rate,
            total_insertion_time,
            total_contains_time,
            expected_false_positive_rate,
        ) = filter::repeat_compute_mistakes_and_execution_time_bloomfilter(
            n_elements_in_bloomfiler,
            n_samples_not_in_bloomfilter,
            n_bits_of_bloomfiler,
            n_hash_functions_in_bloomfiler,
            50,
        );
        table.push(ExperimentResults {
            n_samples_in_bloomerfilter: n_elements_in_bloomfiler,
            n_samples_not_in_bloomfilter,
            false_positive_rate,
            false_negative_rate,
            expected_false_positive_rate,
            total_contains_time: total_contains_time
                / (n_samples_not_in_bloomfilter as f64 / 1000.0),
            total_insertion_time: total_insertion_time / (n_elements_in_bloomfiler as f64 / 1000.0),
        })
    }
    let _ = cli_table::print_stdout(table.with_title());
}
