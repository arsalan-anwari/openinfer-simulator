use anyhow::{anyhow, Result};
use openinfer::Device;
use std::env;
use std::path::PathBuf;

fn default_device() -> Device {
    Device::Cpu
}

fn parse_example_target() -> Result<Option<String>> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--target=") {
            return Ok(Some(value.to_string()));
        }
        if arg == "--target" {
            return match args.next() {
                Some(value) => Ok(Some(value)),
                None => Err(anyhow!(
                    "--target requires a value: cpu|vulkan"
                )),
            };
        }
    }
    Ok(None)
}

pub fn select_device() -> Result<Device> {
    let target = match parse_example_target()? {
        Some(target) => target,
        None => return Ok(default_device()),
    };

    match target.as_str() {
        "cpu" => Ok(Device::Cpu),
        "vulkan" => Ok(Device::Vulkan),
        _ => Err(anyhow!(
            "unknown --target value '{}'; expected cpu|vulkan",
            target
        )),
    }
}

pub fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if manifest_dir.join("res").join("models").exists() {
        return manifest_dir;
    }
    if let Some(parent) = manifest_dir.parent() {
        if parent.join("res").join("models").exists() {
            return parent.to_path_buf();
        }
    }
    manifest_dir
}
