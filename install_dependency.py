import argparse
import sys

def is_venv():
  return (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def pip_install(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

def pip_uninstall(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "uninstall", *cli_args], check=True)

def install_dep():
  print("Installing dependencies...")
  print(f"Python version: {sys.version}")

  try:
    pip_install(
      "fastapi",
      "uvicorn[standard]",
      "python-dotenv",
      "ragas",
      "openai",
      "instructor",
      "litellm",
      "python-multipart",
      "pandas",
    )
  except Exception as e:
    print(f"\nInstallation failed: {e}")
    print("\nTroubleshooting:")
    print("1. Use Python 3.11 or 3.12 instead of Python 3.14")
    print("2. Upgrade pip: python -m pip install --upgrade pip")
    print("3. Try installing packages individually to identify which one fails")
    sys.exit(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  if is_venv():
    install_dep()
  else:
    print("Not running inside a virtual environment")
