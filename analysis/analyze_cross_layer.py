#!/usr/bin/env python3
import argparse
from .proxy_relationships import analyze

parser = argparse.ArgumentParser()
parser.add_argument("trace_dir")
parser.add_argument("--output-dir", default="artifacts/proxy_analysis")
args = parser.parse_args()
analyze(args.trace_dir, "previous_layer", args.output_dir)
