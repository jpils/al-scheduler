use std::fs;
use std::path::{Path, PathBuf};

use crate::job_template::{render_job_template, render_template};

pub struct TrainingWorkspace;

impl TrainingWorkspace {
    pub fn create_upet_workspace(
        project_dir: &Path,
        setup_dir: &Path,
        generation: u32,
        committee_members: usize,
        checkpoint: &Path,
        energy_key: &str,
    ) -> Result<PathBuf, String> {
        if committee_members == 0 {
            return Err("Cannot prepare UPET training for zero committee members.".to_string());
        }

        if !checkpoint.is_file() {
            return Err(format!(
                "UPET checkpoint does not exist: {}",
                checkpoint.display()
            ));
        }

        let generation_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"));

        let dataset_dir = generation_dir.join("dataset");

        let models_dir = generation_dir.join("models");

        let train_set = dataset_dir.join("train.extxyz");

        let validation_set = dataset_dir.join("validation.extxyz");

        let test_set = dataset_dir.join("test.extxyz");

        for dataset_path in [&train_set, &validation_set, &test_set] {
            if !dataset_path.is_file() {
                return Err(format!(
                    "Required UPET dataset file is missing: {}",
                    dataset_path.display()
                ));
            }
        }

        let yaml_template = setup_dir.join("training").join("upet.yaml.template");

        if !yaml_template.is_file() {
            return Err(format!(
                "Missing UPET training template: {}",
                yaml_template.display()
            ));
        }

        let job_template = setup_dir
            .join("jobscripts")
            .join("upet_training_array.sh.template");

        if !job_template.is_file() {
            return Err(format!(
                "Missing UPET training job template: {}",
                job_template.display()
            ));
        }

        fs::create_dir_all(&models_dir).map_err(|error| {
            format!(
                "Failed to create UPET model directory {}: {}",
                models_dir.display(),
                error
            )
        })?;

        let checkpoint_string = absolute_path_string(checkpoint)?;

        let train_string = absolute_path_string(&train_set)?;

        let validation_string = absolute_path_string(&validation_set)?;

        let test_string = absolute_path_string(&test_set)?;

        for member_index in 0..committee_members {
            let member_dir = models_dir.join(format!("member_{member_index:03}"));

            fs::create_dir_all(&member_dir).map_err(|error| {
                format!(
                    "Failed to create committee member directory {}: {}",
                    member_dir.display(),
                    error
                )
            })?;

            let member_yaml = member_dir.join("train.yaml");

            render_template(
                &yaml_template,
                &member_yaml,
                &[
                    ("checkpoint", checkpoint_string.clone()),
                    ("train_set", train_string.clone()),
                    ("validation_set", validation_string.clone()),
                    ("test_set", test_string.clone()),
                    ("energy_key", energy_key.to_string()),
                ],
            )
            .map_err(|error| {
                format!(
                    "Failed to render UPET configuration for member_{member_index:03}: {}",
                    error
                )
            })?;
        }

        let max_index = committee_members - 1;

        let job_script = generation_dir.join("submit_training_array.sh");

        render_job_template(&job_template, &job_script, generation, max_index)
            .map_err(|error| format!("Failed to render UPET training job script: {}", error))?;

        Ok(job_script)
    }

    pub fn create_mock_upet_models(
        project_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(), String> {
        if committee_members == 0 {
            return Err("Cannot create mock UPET models for zero committee members.".to_string());
        }

        let models_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"))
            .join("models");

        for member_index in 0..committee_members {
            let member_name = format!("member_{member_index:03}");
            let member_dir = models_dir.join(&member_name);

            if !member_dir.is_dir() {
                return Err(format!(
                    "UPET member directory does not exist: {}",
                    member_dir.display()
                ));
            }

            let mock_model = member_dir.join("mock_trained_model.pt");

            fs::write(
                &mock_model,
                format!("mock UPET model for generation {generation}, {member_name}\n"),
            )
            .map_err(|error| {
                format!(
                    "Failed to create mock UPET model {}: {}",
                    mock_model.display(),
                    error
                )
            })?;
        }

        Ok(())
    }

    pub fn create_n2p2_workspace(
        project_dir: &Path,
        setup_dir: &Path,
        generation: u32,
        committee_members: usize,
    ) -> Result<(PathBuf, PathBuf), String> {
        if committee_members == 0 {
            return Err("Cannot prepare n2p2 training for zero committee members.".to_string());
        }

        let generation_dir = project_dir
            .join("training")
            .join(format!("generation_{generation}"));
        let dataset = generation_dir.join("dataset").join("input.data");
        let models_dir = generation_dir.join("models");
        let input_nn = setup_dir.join("training").join("input.nn");

        if !dataset.is_file() {
            return Err(format!(
                "Required n2p2 dataset file is missing: {}",
                dataset.display()
            ));
        }

        if !input_nn.is_file() {
            return Err(format!(
                "Missing n2p2 settings file: {}",
                input_nn.display()
            ));
        }

        fs::create_dir_all(&models_dir).map_err(|error| {
            format!(
                "Failed to create n2p2 model directory {}: {}",
                models_dir.display(),
                error
            )
        })?;

        for member_index in 0..committee_members {
            let member_dir = models_dir.join(format!("member_{member_index:03}"));

            fs::create_dir_all(&member_dir).map_err(|error| {
                format!(
                    "Failed to create n2p2 committee member directory {}: {}",
                    member_dir.display(),
                    error
                )
            })?;

            fs::copy(&dataset, member_dir.join("input.data")).map_err(|error| {
                format!(
                    "Failed to copy {} into {}: {}",
                    dataset.display(),
                    member_dir.display(),
                    error
                )
            })?;

            fs::copy(&input_nn, member_dir.join("input.nn")).map_err(|error| {
                format!(
                    "Failed to copy {} into {}: {}",
                    input_nn.display(),
                    member_dir.display(),
                    error
                )
            })?;
        }

        let scaling_template = setup_dir
            .join("jobscripts")
            .join("n2p2_scaling_array.sh.template");
        let training_template = setup_dir
            .join("jobscripts")
            .join("n2p2_training_array.sh.template");

        for template in [&scaling_template, &training_template] {
            if !template.is_file() {
                return Err(format!("Missing n2p2 job template: {}", template.display()));
            }
        }

        let max_index = committee_members - 1;
        let scaling_script = generation_dir.join("submit_scaling_array.sh");
        let training_script = generation_dir.join("submit_training_array.sh");

        render_job_template(&scaling_template, &scaling_script, generation, max_index)
            .map_err(|error| format!("Failed to render n2p2 scaling job script: {}", error))?;

        render_job_template(&training_template, &training_script, generation, max_index)
            .map_err(|error| format!("Failed to render n2p2 training job script: {}", error))?;

        Ok((scaling_script, training_script))
    }
}

fn absolute_path_string(path: &Path) -> Result<String, String> {
    let absolute = path.canonicalize().map_err(|error| {
        format!(
            "Failed to resolve absolute path {}: {}",
            path.display(),
            error
        )
    })?;

    Ok(absolute.display().to_string())
}
