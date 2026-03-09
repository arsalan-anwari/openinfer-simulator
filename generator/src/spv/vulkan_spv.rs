//! Generate embedded SPIR-V shader maps using hardcoded path convention.
//!
//! Scans `src/ops/vulkan/{category}/{name}/bin/*.spv` for SPV binaries
//! and builds `embedded_spv()` match arms. Also emits `spv_dir_for_op(name)`.
//! No ops.json devices.vulkan needed.
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Vulkan SPV path convention: src/ops/vulkan/{category}/{name}/bin/{entrypoint}.spv
const VULKAN_OPS_BASE: &str = "src/ops/vulkan";

/// Generate a Rust map that embeds available SPIR-V binaries.
/// Scans the filesystem for .spv files under the hardcoded path convention.
/// Also emits spv_dir_for_op(op_name) for use by Vulkan kernels.
pub fn generate_spv_map(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let vulkan_base = manifest_dir.join(VULKAN_OPS_BASE);
    println!("cargo:rerun-if-changed={}", vulkan_base.display());

    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    let mut spv_dirs: Vec<(String, String)> = Vec::new();

    if vulkan_base.exists() {
        for category_entry in fs::read_dir(&vulkan_base)? {
            let category_entry = category_entry?;
            let category_path = category_entry.path();
            if !category_path.is_dir() {
                continue;
            }
            let category = category_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            for op_entry in fs::read_dir(&category_path)? {
                let op_entry = op_entry?;
                let op_path = op_entry.path();
                if !op_path.is_dir() {
                    continue;
                }
                let op_name = op_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                let bin_dir = op_path.join("bin");
                if !bin_dir.exists() || !bin_dir.is_dir() {
                    continue;
                }
                let spv_dir_str = format!("{}/{}/{}/bin", VULKAN_OPS_BASE, category, op_name);
                spv_dirs.push((op_name, spv_dir_str));
                for spv_entry in fs::read_dir(&bin_dir)? {
                    let spv_entry = spv_entry?;
                    let spv_path = spv_entry.path();
                    if spv_path.extension().and_then(|e| e.to_str()) == Some("spv") {
                        let entrypoint = spv_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                            .ok_or_else(|| format!("invalid spv filename {:?}", spv_path))?;
                        entries.push((entrypoint, spv_path));
                    }
                }
            }
        }
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("spv_embedded.rs");
    let mut output = String::new();
    output.push_str("pub fn embedded_spv(name: &str) -> Option<&'static [u8]> {\n");
    output.push_str("    match name {\n");
    for (name, path) in &entries {
        let path = path.to_string_lossy().replace('\\', "/");
        output.push_str(&format!(
            "        \"{name}\" => Some(include_bytes!(r#\"{path}\"#)),\n"
        ));
    }
    output.push_str("        _ => None,\n");
    output.push_str("    }\n");
    output.push_str("}\n\n");
    output.push_str("/// Returns spv_dir for Vulkan kernels. Use spv_dir_for_op(OpKind::X.as_str()).\n");
    output.push_str("pub fn spv_dir_for_op(op_name: &str) -> &'static str {\n");
    output.push_str("    match op_name {\n");
    for (op_name, spv_dir) in &spv_dirs {
        output.push_str(&format!("        \"{op_name}\" => \"{spv_dir}\",\n"));
    }
    output.push_str("        _ => \"src/ops/vulkan/unknown/bin\",\n");
    output.push_str("    }\n");
    output.push_str("}\n");
    fs::write(out_file, output)?;
    Ok(())
}

/// Write an empty embedded SPIR-V map (for non-Vulkan builds).
pub fn write_empty_map() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("spv_embedded.rs");
    let output = "pub fn embedded_spv(_: &str) -> Option<&'static [u8]> { None }\n\n\
        pub fn spv_dir_for_op(_: &str) -> &'static str { \"src/ops/vulkan/unknown/bin\" }\n";
    fs::write(out_file, output)?;
    Ok(())
}
