"""basler/ filming-session metadata catalog.

Scans /bucket/ReiterU/Ants/basler and emits a human-manageable catalog
(catalog.csv + videos.csv + trials.csv + catalog_run.json).

See detection_pipeline/catalog.py for the CLI entry point.
"""

__all__ = ["const", "model", "naming", "classify", "discover",
           "probe", "footprint", "qc", "sess_parse", "cache", "build"]
