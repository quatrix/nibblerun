#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::decode;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to decode() - should never panic
    // May return empty vec for malformed input, but should not crash
    let _ = decode(data);
});
