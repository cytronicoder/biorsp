import argparse

from biorsp.simulations.generator import generate_grid


def main():
    parser = argparse.ArgumentParser(description="Generate BioRSP simulation datasets.")
    parser.add_argument(
        "--output_dir", type=str, default="sim_results/inputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--n_points", type=int, default=2000, help="Number of points per simulation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Starting simulation generation in {args.output_dir}...")
    generate_grid(n_points=args.n_points, output_dir=args.output_dir, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
