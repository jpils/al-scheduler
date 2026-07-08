import os
import ase.io


def export_extxyz(
    project_dir,
    generation_num,
    train_set,
    validation_set,
    test_set,
):
    """
    Write UPET extxyz datasets.
    """

    target_dir = os.path.join(
        project_dir,
        "training",
        "training_set",
        f"generation_{generation_num}",
    )

    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    ase.io.write(
        os.path.join(target_dir, "train.xyz"),
        train_set,
        format="extxyz",
    )

    if validation_set:
        ase.io.write(
            os.path.join(target_dir, "validation.xyz"),
            validation_set,
            format="extxyz",
        )

    if test_set:
        ase.io.write(
            os.path.join(target_dir, "test.xyz"),
            test_set,
            format="extxyz",
        )

    print(
        f"[+] Dataset exported successfully: {target_dir}/"
    )

    print(
        f"    - train.xyz : {len(train_set)} frames"
    )

    print(
        f"    - validation.xyz   : {len(validation_set)} frames"
    )

    print(
        f"    - test.xyz  : {len(test_set)} frames"
    )