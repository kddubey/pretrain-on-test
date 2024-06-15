#!/bin/bash
# This script additionally sets up the Python environment required to run the analysis.
# It assumes _setup_python_env.sh was already ran

set -uox pipefail
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
# No e b/c want to shut down regardless of success or failure


python -m pip install ".[stat]"

cd analysis
