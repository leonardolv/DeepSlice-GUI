import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.training.train_runner import main, prepare_training_run


def _create_mock_images(root: pathlib.Path):
    brain_a = root / "brainA"
    brain_b = root / "brainB"
    brain_c = root / "brainC"
    brain_a.mkdir(parents=True, exist_ok=True)
    brain_b.mkdir(parents=True, exist_ok=True)
    brain_c.mkdir(parents=True, exist_ok=True)

    for idx, folder in enumerate([brain_a, brain_a, brain_b, brain_b, brain_c], start=1):
        (folder / f"{folder.name}_s{idx:03d}.png").write_text("x", encoding="utf-8")


def test_prepare_training_run_writes_split_and_metadata(tmp_path):
    images_root = tmp_path / "images"
    output_root = tmp_path / "out"
    _create_mock_images(images_root)

    result = prepare_training_run(
        image_root=str(images_root),
        output_dir=str(output_root),
        species="mouse",
        group_mode="parent-folder",
        seed=11,
        train_fraction=0.60,
        val_fraction=0.20,
        recursive=True,
        preprocessing_options={"clahe": True},
        training_options={"epochs": 0},
    )

    split_path = pathlib.Path(result["split_manifest_path"])
    metadata_path = pathlib.Path(result["metadata_path"])
    assert split_path.is_file()
    assert metadata_path.is_file()

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert split_payload["counts"]["total"] == 5
    assert split_payload["counts"]["train"] + split_payload["counts"]["val"] + split_payload["counts"]["test"] == 5
    assert split_payload["leakage_free"] is True

    assert metadata_payload["species"] == "mouse"
    assert metadata_payload["split_counts"]["total"] == 5
    assert metadata_payload["split_leakage_free"] is True


def test_train_runner_main_dry_run_succeeds(tmp_path):
    images_root = tmp_path / "images"
    output_root = tmp_path / "run"
    _create_mock_images(images_root)

    exit_code = main(
        [
            "--images",
            str(images_root),
            "--output-dir",
            str(output_root),
            "--species",
            "rat",
            "--group-mode",
            "filename-prefix",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert (output_root / "training_split_manifest.json").is_file()
    assert (output_root / "training_run_metadata.json").is_file()


def test_train_runner_main_dry_run_with_xception_trainer_persists_options(tmp_path):
    images_root = tmp_path / "images"
    output_root = tmp_path / "run_xception"
    _create_mock_images(images_root)

    exit_code = main(
        [
            "--images",
            str(images_root),
            "--output-dir",
            str(output_root),
            "--trainer",
            "xception-finetune",
            "--xception-init",
            "imagenet",
            "--freeze-base-epochs",
            "2",
            "--finetune-learning-rate",
            "0.0002",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    metadata_payload = json.loads(
        (output_root / "training_run_metadata.json").read_text(encoding="utf-8")
    )
    options = metadata_payload["training_options"]

    assert options["trainer"] == "xception-finetune"
    assert options["xception_init"] == "imagenet"
    assert options["freeze_base_epochs"] == 2
    assert options["finetune_learning_rate"] == 0.0002


def test_train_runner_main_requires_labels_for_epochs(tmp_path):
    images_root = tmp_path / "images"
    output_root = tmp_path / "run"
    _create_mock_images(images_root)

    exit_code = main(
        [
            "--images",
            str(images_root),
            "--output-dir",
            str(output_root),
            "--epochs",
            "1",
        ]
    )

    assert exit_code == 1
