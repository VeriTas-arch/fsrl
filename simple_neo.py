"""Command-line entry point for the readable transitive-inference trainer."""

from fsrl.cli import main


if __name__ == "__main__":
    main(["--output-dir", "output", "--nbiter", "30000"])
