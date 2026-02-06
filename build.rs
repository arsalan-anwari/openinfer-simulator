use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = match env::var("CARGO_MANIFEST_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(err) => {
            eprintln!("build.rs: missing CARGO_MANIFEST_DIR: {err}");
            return;
        }
    };
    let op_defs = manifest_dir.join("src/op_defs.rs");
    let op_types = manifest_dir.join("src/graph/types.rs");
    let ops_json = manifest_dir.join("ops.json");
    let settings_json = manifest_dir.join("settings.json");
    println!("cargo:rerun-if-changed={}", op_defs.display());
    println!("cargo:rerun-if-changed={}", op_types.display());
    println!("cargo:rerun-if-changed={}", ops_json.display());
    println!("cargo:rerun-if-changed={}", settings_json.display());
    if let Err(err) = openinfer_generator::settings::apply_settings(&manifest_dir) {
        eprintln!("build.rs: failed to apply settings: {err}");
    }
    if let Err(err) = openinfer_generator::vulkan_spv::generate_spv_map(&manifest_dir) {
        eprintln!("build.rs: failed to generate embedded spv map: {err}");
        if let Err(write_err) = openinfer_generator::vulkan_spv::write_empty_map() {
            eprintln!("build.rs: failed to write empty spv map: {write_err}");
        }
    }
}
