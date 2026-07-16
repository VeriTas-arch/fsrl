"""Command-line entry point for the readable transitive-inference trainer."""

from fsrl.cli import main


if __name__ == "__main__":
    for seed in range(1, 21):
        main([
            "--output-dir", f"output/seed_{seed}",
            "--nbiter", "5000",
            "--seed", str(seed),
        ])