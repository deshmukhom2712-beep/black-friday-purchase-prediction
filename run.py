"""One-command runner. Usage:  python run.py            (real data in data/train.csv)
                              python run.py --demo     (synthetic data, to test setup)
                              python run.py --fast     (quick run, skips tuning)"""
import sys, os, subprocess
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
args = sys.argv[1:]
if "--demo" in args:
    args.remove("--demo")
    subprocess.check_call([sys.executable, "src/make_synthetic_data.py", "data/demo.csv"])
    args += ["--data", "data/demo.csv"]
elif not os.path.exists("data/train.csv"):
    sys.exit("data/train.csv not found. Download it from Kaggle (see README.md) or run: python run.py --demo")
subprocess.check_call([sys.executable, "src/train_compare.py"] + args)
