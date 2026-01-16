"""Manifest of scripts available for smoke testing.

This manifest lists all scripts in dev/ that should be tested in CI.
Each script must support --smoke or --mode smoke for quick testing.
"""

SCRIPTS = [
    {
        "id": "plot_simulation_csv",
        "relpath": "dev/make_figures/plot_simulation_csv.py",
        "type": "make_figures",
        "args_smoke": ["--help"],  # Just test help for now
        "timeout_seconds": 30,
    },
    {
        "id": "make_end_to_end_workflow",
        "relpath": "dev/make_figures/make_end_to_end_workflow.py",
        "type": "make_figures",
        "args_smoke": [
            "--smoke",
            "--feature",
            "TestGene",
            "--seed",
            "42",
            "--B",
            "12",
            "--delta-deg",
            "45",
        ],
        "expected_outputs": ["*.png"],
        "timeout_seconds": 120,
    },
    {
        "id": "make_polar_embedding_figure",
        "relpath": "dev/make_figures/make_polar_embedding_figure.py",
        "type": "make_figures",
        "args_smoke": [
            "--smoke",
            "--feature",
            "TestGene",
            "--seed",
            "42",
        ],
        "expected_outputs": ["*.png"],
        "timeout_seconds": 90,
    },
    {
        "id": "make_schematic_diagram",
        "relpath": "dev/make_figures/make_schematic_diagram.py",
        "type": "make_figures",
        "args_smoke": ["--smoke"],
        "expected_outputs": ["fig_schematic_diagram*.png"],
        "timeout_seconds": 90,
    },
    {
        "id": "debug_end_to_end",
        "relpath": "dev/debug/debug_end_to_end.py",
        "type": "debug",
        "args_smoke": ["--smoke"],
        "expected_outputs": ["debug_figure_*.png"],
        "timeout_seconds": 120,
    },
    {
        "id": "debug_selection_bias",
        "relpath": "dev/debug/debug_selection_bias.py",
        "type": "debug",
        "args_smoke": ["--smoke"],
        "expected_outputs": ["debug_selection_bias_*.png"],
        "timeout_seconds": 120,
    },
]


def get_script_by_id(script_id: str) -> dict:
    """Get script manifest entry by ID."""
    for script in SCRIPTS:
        if script["id"] == script_id:
            return script
    raise KeyError(f"Script {script_id} not found in manifest")


def get_all_scripts():
    """Get all script manifest entries."""
    return SCRIPTS
