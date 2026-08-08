"""
Phase: Configuration and Setup
Expects: None
Outputs: Configuration constants (file paths, split ratios, random seed)
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "CICIDS2017", "Monday-WorkingHours.pcap_ISCX.csv"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
