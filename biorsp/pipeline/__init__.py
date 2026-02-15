"""Heart case-study pipeline entrypoints."""


def run_case_study(*args, **kwargs):
    from biorsp.pipeline.hierarchy import run_case_study as _run_case_study

    return _run_case_study(*args, **kwargs)


__all__ = ["run_case_study"]
