#!/bin/bash
# Find where PlotlyPlotter is actually used to create these files

find src -name "*.py" -type f -exec grep -l "PlotlyPlotter\|save_figure\|write_html" {} \; | while read f; do
    echo "=== $f ==="
    grep -n "PlotlyPlotter\|save_figure\|write_html" "$f" | head -5
done
