"""Command-line entry point for teaching figure generation."""

from fsrl.eval_figures import main


if __name__ == "__main__":
    main(["--model-path", "output"])
