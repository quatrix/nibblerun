# nibblerun

A high-performance time series compression library for Rust, optimized for temperature sensor data.

## Features

- Bit-packed delta encoding with variable-length codes
- Zero-run length encoding for repeated values
- Configurable interval-based timestamp quantization
- Automatic averaging of multiple readings within the same interval
- ~250M readings/second encoding throughput

## Usage

```rust
use nibblerun::{Encoder, decode};

// Create encoder with 300-second intervals
let mut enc = Encoder::with_interval(300);

// Append readings (timestamp, temperature)
enc.append(1700000000, 23);
enc.append(1700000300, 24);
enc.append(1700000600, 22);

// Serialize to bytes
let bytes = enc.to_bytes();

// Decode back
let readings = decode(&bytes);
for r in readings {
    println!("ts: {}, temp: {}", r.ts, r.temperature);
}
```

## Testing

Run unit tests:
```bash
cargo test
```

Run property-based tests (included in unit tests via proptest):
```bash
cargo test proptests
```

## Fuzzing

The library includes fuzz targets using cargo-fuzz. Install cargo-fuzz first:
```bash
cargo install cargo-fuzz
```

Run fuzz targets:
```bash
# Roundtrip fuzzing (tests encode/decode invariants)
cargo fuzz run fuzz_roundtrip

# Decode-only fuzzing (tests decode never panics on arbitrary input)
cargo fuzz run fuzz_decode

# Run with time limit (e.g., 60 seconds)
cargo fuzz run fuzz_roundtrip -- -max_total_time=60
```

## License

MIT
