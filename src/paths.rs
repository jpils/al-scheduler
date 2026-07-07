use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

pub fn scheduler_home() -> Result<PathBuf, Box<dyn Error>> {
    let mut dir = std::env::current_exe()?;

    dir.pop();

    loop {
        if dir.join("Cargo.toml").exists()
            && dir.join("pixi.toml").exists()
            && dir.join("python").is_dir()
        {
            return Ok(dir);
        }

        if !dir.pop() {
            break;
        }
    }

    if let Ok(home) = std::env::var("AI_SCHEDULER_HOME") {
        return Ok(PathBuf::from(home));
    }

    Err("Could not locate the ai-scheduler installation.".into())
}

/// Construct a command that executes Python inside the scheduler's
/// Pixi environment.
pub fn pixi_python(environment: &str) -> Result<Command, Box<dyn Error>>{
    let scheduler = scheduler_home()?;

    let mut cmd = Command::new("pixi");

    cmd.current_dir(&scheduler);

    cmd.arg("run")
        .arg("-e")
        .arg(environment)
        .arg("--manifest-path")
        .arg(scheduler.join("pixi.toml"))
        .arg("python");

    Ok(cmd)
}