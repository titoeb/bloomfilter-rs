use bloomfilter_rs::filter;
#[test]
fn test_false_positive_rate() {
    let (false_positive_rate, false_negative_rate, _, _, expected_false_positive_rate) =
        filter::repeat_compute_mistakes_and_execution_time_bloomfilter(1000, 1000, 10000, 20, 20);

    assert_eq!(false_negative_rate, 0.0);

    // How many percentage can the `false_positive_rate` deviate from the
    // `expected_false_positive_rate`.
    let margin_for_actual_false_negative_rate = 0.2;
    assert!(
        (false_positive_rate
            > expected_false_positive_rate * margin_for_actual_false_negative_rate)
            & (false_positive_rate
                < expected_false_positive_rate * (1.0 + margin_for_actual_false_negative_rate))
    );
}
