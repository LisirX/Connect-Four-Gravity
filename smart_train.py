import os
import re
import shutil
import subprocess
import time
from datetime import datetime

# Import all constants from the centralized config file
from config import (
    PYTHON_EXE, MODEL_DIR, ARCHIVE_DIR, MODEL_BASENAME, VERSION_FILE,
    ARENA_WIN_RATE_THRESHOLD, CHALLENGER_MODEL_PATH
)

def get_current_version():
    """Reads the current model version from the version file."""
    if not os.path.exists(VERSION_FILE):
        print(f"Version file not found. Initializing '{VERSION_FILE}' with version 1.")
        with open(VERSION_FILE, 'w') as f:
            f.write('1')
        return 1
    with open(VERSION_FILE, 'r') as f:
        return int(f.read().strip())

def set_current_version(version):
    """Writes the new model version to the version file."""
    with open(VERSION_FILE, 'w') as f:
        f.write(str(version))

def find_or_create_initial_champion(version):
    """
    Finds the champion model for the current version.
    If it doesn't exist, it tries to rename the default model.
    """
    champion_path = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}_v{version}.pth")
    if os.path.exists(champion_path):
        return champion_path

    print(f"\n!!! Warning: Champion model not found: {champion_path}")
    # Fallback: Check if the default challenger model path exists from a previous run
    if os.path.exists(CHALLENGER_MODEL_PATH):
        print(f"Found '{CHALLENGER_MODEL_PATH}'. Promoting it to be champion v{version}.")
        shutil.move(CHALLENGER_MODEL_PATH, champion_path)
        return champion_path
    else:
        print("\n!!! ERROR: Cannot find any initial model to start the training loop.")
        print(f"!!! Please run 'train.py' once manually to create '{CHALLENGER_MODEL_PATH}'.")
        exit(1)


def run_training_loop():
    """Main function to orchestrate the AI training and evaluation loop."""
    
    # Ensure necessary directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    current_version = get_current_version()

    while True:
        try:
            print("\n" + "="*75)
            print(f"== Starting AI Training & Evaluation Loop: Version {current_version} -> {current_version + 1}")
            print("="*75 + "\n")

            # --- [Step 0] Define paths for the current loop ---
            champion_model = find_or_create_initial_champion(current_version)
            champion_backup = os.path.join(MODEL_DIR, f"champion_backup_v{current_version}.pth")

            print(f"Current Champion: v{current_version} ('{os.path.basename(champion_model)}')")

            # --- [Step 1] Backup the current champion model ---
            print("\n--- [Step 1/4] Backing up the current champion model... ---")
            shutil.copy(champion_model, champion_backup)
            print(f"Champion model backed up to '{os.path.basename(champion_backup)}'")

            # --- [Step 2] Run self-play to generate training data ---
            print("\n--- [Step 2/4] Running self_play.py to generate new data... ---")
            subprocess.run([PYTHON_EXE, "self_play.py"], check=True)

            # --- [Step 3] Train a new challenger model ---
            print("\n--- [Step 3/4] Running train.py to create a new challenger... ---")
            subprocess.run([PYTHON_EXE, "train.py"], check=True)
            print(f"Challenger model created at '{CHALLENGER_MODEL_PATH}'")

            # --- [Step 4] Run arena to evaluate the challenger ---
            print("\n--- [Step 4/4] Running arena.py to evaluate Challenger vs. Champion... ---")
            # We pass the backup as the 'old_model' for the arena
            process = subprocess.run(
                [PYTHON_EXE, "arena.py", "--old_model", champion_backup],
                capture_output=False, # We want to see arena's output in real-time
                text=True
            )
            win_rate = process.returncode # arena.py returns win_rate * 100 as exit code

            # --- [Step 5] Analyze results and promote if necessary ---
            print("\n" + "="*75)
            print("== Evaluation Result Analysis")
            print("="*75 + "\n")
            print(f"    Challenger Win Rate (vs. lose): {win_rate}%")
            print(f"    Promotion Threshold:            {ARENA_WIN_RATE_THRESHOLD}%")
            print("")

            if win_rate >= ARENA_WIN_RATE_THRESHOLD:
                new_version = current_version + 1
                print(f"*** CONCLUSION: Challenge SUCCEEDED! New model is now champion v{new_version}! ***\n")

                # Archive the old champion (which is the backup)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"model_v{current_version}_lost_to_v{new_version}_{timestamp}.pth"
                archive_path = os.path.join(ARCHIVE_DIR, archive_name)
                shutil.move(champion_backup, archive_path)
                print(f"Old champion (v{current_version}) archived to: {archive_name}")

                # Promote the challenger to the new champion
                new_champion_model = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}_v{new_version}.pth")
                shutil.move(CHALLENGER_MODEL_PATH, new_champion_model)
                print(f"New champion named: '{os.path.basename(new_champion_model)}'")

                # Update the version for the next loop
                set_current_version(new_version)
                current_version = new_version
            else:
                print("*** CONCLUSION: Challenge FAILED. Original champion remains. ***\n")
                print("Deleting failed challenger model...")
                os.remove(CHALLENGER_MODEL_PATH)
                print("Cleaning up champion backup...")
                os.remove(champion_backup)
                print(f"Champion v{current_version} remains unbeaten.")

            print("\n" + "="*75)
            print(f"== Loop complete. Starting next loop for v{current_version} in 10 seconds... ==")
            print("="*75 + "\n")
            time.sleep(10)

        except subprocess.CalledProcessError as e:
            print(f"\n!!! A critical error occurred in a subprocess: {e}")
            print("!!! The training loop has been terminated.")
            break
        except FileNotFoundError as e:
            print(f"\n!!! File not found error: {e}")
            print("!!! The training loop has been terminated.")
            break
        except KeyboardInterrupt:
            print("\n\nTraining loop interrupted by user. Exiting gracefully.")
            # Clean up backup file if it exists
            if 'champion_backup' in locals() and os.path.exists(champion_backup):
                os.remove(champion_backup)
                print("Cleaned up temporary backup file.")
            break

if __name__ == "__main__":
    run_training_loop()