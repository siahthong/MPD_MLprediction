#!/usr/bin/env python3
"""Optional 273 K test analysis; manuscript test evaluation is at 298 K."""
from pathlib import Path
import argparse
from evaluate_manuscript_298K import evaluate
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    evaluate(temperature=273.0,output=args.output,verify=False)
