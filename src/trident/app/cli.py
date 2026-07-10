from __future__ import annotations
import argparse, subprocess, sys

def main():
    p=argparse.ArgumentParser('trident')
    p.add_argument('command', nargs='?', default='app', choices=['app'])
    args=p.parse_args()
    if args.command=='app':
        subprocess.check_call([sys.executable,'-m','streamlit','run','src/trident/app/workbench.py'])
