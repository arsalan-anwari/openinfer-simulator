//! Clean SPIR-V binaries using hardcoded path convention.
//!
//! Scans `src/ops/vulkan/{category}/{name}/bin/*.spv` and deletes all .spv files.
//! No ops.json devices.vulkan needed.
use std::fs;
use std::path::Path;

const VULKAN_OPS_BASE: &str = "src/ops/vulkan";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let vulkan_base = workspace_root.join(VULKAN_OPS_BASE);

    let mut deleted = 0usize;

    if vulkan_base.exists() {
        for category_entry in fs::read_dir(&vulkan_base)? {
            let category_entry = category_entry?;
            let category_path = category_entry.path();
            if !category_path.is_dir() {
                continue;
            }
            for op_entry in fs::read_dir(&category_path)? {
                let op_entry = op_entry?;
                let op_path = op_entry.path();
                if !op_path.is_dir() {
                    continue;
                }
                let bin_dir = op_path.join("bin");
                if !bin_dir.exists() || !bin_dir.is_dir() {
                    continue;
                }
                for spv_entry in fs::read_dir(&bin_dir)? {
                    let spv_entry = spv_entry?;
                    let spv_path = spv_entry.path();
                    if spv_path.extension().and_then(|e| e.to_str()) == Some("spv") {
                        fs::remove_file(&spv_path)?;
                        deleted += 1;
                    }
                }
            }
        }
    }

    println!("Deleted {} SPIR-V files.", deleted);
    Ok(())
}
