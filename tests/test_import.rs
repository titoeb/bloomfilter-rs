// I allow unused imports here, because I just want to test in this file, that these
// two traits can be imported from the trait.
#[allow(unused_imports)]
use bloomfilter_rs::filter;

#[test]
fn test_imports_have_run() {
    // If we get here, we have successfully imported
    // `bloomfilter::filter` and are ready
    // to use it now.
    assert!(true);
}
