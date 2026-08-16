fn main() {
    let version = "checking via rustc --version separately";
    let arr = [1, 2, 3, 4, 5];
    let sum: i32 = arr.iter().sum();
    let mean = sum as f64 / arr.len() as f64;

    println!("Array: {:?}", arr);
    println!("Mean: {}", mean);
    println!("✅ Rust is working ({})", version);
}