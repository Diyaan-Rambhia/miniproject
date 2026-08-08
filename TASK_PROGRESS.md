# Task Progress

- [x] 1. CICIDS2017 — combined 8 CSVs (2,830,743 rows) and stratified split 70/10/20 into train/val/test with seed 42
- [x] 2. DGA + Tranco domain datasets — combined 4,347,492 unique domains and stratified split 70/10/20 into train/val/test with seed 123
- [x] 3. Update all model folders' code to use these new files — updated transformer/, VAE/, DGA_detector/, fusion_head/, Adverserial_robustness/, and pipeline scripts to load pre-split CSVs
- [x] 4. Save all trained model weights centrally for reuse — updated save_artifacts.py in all model folders to copy final weights to models/saved_weights/

## Phase 2: One-Command End-to-End Training Script

- [x] 1. Verify/create working main.py for all 5 model folders (transformer/, VAE/, DGA_detector/, fusion_head/, Adverserial_robustness/) — created models/VAE/main.py and verified all 5 main.py files
- [x] 2. Confirm dependency ordering & signals CSV generator — confirmed order and generate_signals.py compatibility
- [x] 3. Create models/train_all.py with checkpoint skip, timing, and evaluation summary table — created models/train_all.py with dependency order execution and skip logic
- [x] 4. Update requirements.txt with all required project dependencies — populated requirements.txt at project root
