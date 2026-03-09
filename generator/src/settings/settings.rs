//! Generate build-time settings from `settings.json`.
use std::error::Error;
use std::fs;
use std::path::Path;

/// Dtype codes matching descriptor::dtype_code in the simulator.
fn dtype_code(s: &str) -> u32 {
    match s {
        "i8" => 1,
        "i16" => 2,
        "i32" => 3,
        "i64" => 4,
        "u8" => 5,
        "u16" => 6,
        "u32" => 7,
        "u64" => 8,
        "f16" => 9,
        "bf16" => 10,
        "f32" => 11,
        "f64" => 12,
        "f8" => 13,
        "bool" => 14,
        "i4" => 16,
        "u4" => 19,
        _ => 0,
    }
}

fn encode_rule(rule: &[String]) -> u32 {
    rule.iter()
        .enumerate()
        .take(4)
        .map(|(i, s)| dtype_code(s).min(0xFF) << (i * 8))
        .fold(0u32, |a, b| a | b)
}

/// Apply settings and emit build artifacts used by Vulkan shaders.
pub fn apply_settings(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let settings_path = manifest_dir.join("settings.json");
    println!("cargo:rerun-if-changed={}", settings_path.display());
    let ops_path = manifest_dir.join("ops.json");
    println!("cargo:rerun-if-changed={}", ops_path.display());
    let max_dims = read_settings_max_dims(&settings_path).unwrap_or(8);
    println!("cargo:rustc-env=OPENINFER_VK_MAX_DIMS={}", max_dims);
    write_shader_config(manifest_dir, max_dims)?;
    write_rust_config(max_dims)?;
    write_accum_utils(manifest_dir)?;
    Ok(())
}

fn read_settings_max_dims(path: &Path) -> Option<usize> {
    let contents = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    value
        .get("vulkan")
        .and_then(|v| v.get("max_tensor_rank"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

fn write_shader_config(manifest_dir: &Path, max_dims: usize) -> Result<(), Box<dyn Error>> {
    let shader_config = manifest_dir.join("src/ops/vulkan/shaders/generated_config.slang");
    let contents = format!(
        "#ifndef OPENINFER_GENERATED_CONFIG\n#define OPENINFER_GENERATED_CONFIG 1\n#define OPENINFER_VK_MAX_DIMS {}\n#endif\n",
        max_dims
    );
    fs::write(shader_config, contents)?;
    Ok(())
}

fn write_rust_config(max_dims: usize) -> Result<(), Box<dyn Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("vulkan_config.rs");
    let contents = format!("pub const MAX_DIMS: usize = {max_dims};\n");
    fs::write(out_file, contents)?;
    Ok(())
}

fn write_accum_utils(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let ops_path = manifest_dir.join("ops.json");
    let contents = fs::read_to_string(&ops_path).unwrap_or_else(|_| "{\"ops\":[]}".to_string());
    let json: serde_json::Value = serde_json::from_str(&contents)?;
    let empty: Vec<serde_json::Value> = vec![];
    let ops = json.get("ops").and_then(|o| o.as_array()).unwrap_or(&empty);

    let mut out = String::from(
        "// Auto-generated from ops.json accumulation_rules. Do not edit.\n\
         #ifndef OPENINFER_ACCUM_UTILS\n\
         #define OPENINFER_ACCUM_UTILS 1\n\n",
    );

    let empty_rules: Vec<serde_json::Value> = vec![];
    for op in ops {
        let name = op.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let rules = op
            .get("accumulation_rules")
            .and_then(|r| r.as_array())
            .unwrap_or(&empty_rules);
        if rules.is_empty() {
            continue;
        }
        let op_upper = name.to_uppercase().replace('-', "_");
        let empty_arr: Vec<serde_json::Value> = vec![];
        for rule in rules.iter() {
            let arr = rule.as_array().unwrap_or(&empty_arr);
            let dtypes: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_uppercase()))
                .collect();
            let def_name = format!("ACC_{}_{}", op_upper, dtypes.join("_"));
            let code = encode_rule(
                &arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>(),
            );
            out.push_str(&format!("#define {} 0x{:04X}u\n", def_name, code));
        }
        out.push('\n');
    }

    out.push_str("#endif // OPENINFER_ACCUM_UTILS\n");
    let out_path = manifest_dir.join("src/ops/vulkan/shaders/accum_utils.slang");
    fs::write(out_path, out)?;
    Ok(())
}
