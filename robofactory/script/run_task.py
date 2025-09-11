import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner with task config input.")
    parser.add_argument('config', type=str, help="Task config file to use")
    args = parser.parse_args()

    command = (
        f"python -m robofactory.planner.run "
        f"-c \"{args.config}\" " 
        f"--render-mode=\"rgb_array\" "
        f"-b=\"cpu\" "
        f"-n 1 "
    )

    os.system(command)

if __name__ == "__main__":
    main()
