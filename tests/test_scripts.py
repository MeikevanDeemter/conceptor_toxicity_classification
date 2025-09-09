import os
import subprocess
import pytest
from pathlib import Path


def get_script_files():
    """Get all Python script files from the scripts directory."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    return [str(p) for p in scripts_dir.glob("*.py")]


def get_test_args(script_path):
    """Get appropriate test arguments for each script to ensure it can run.
    
    For scripts that require specific arguments, we provide "safe" test values.
    Otherwise, we use --help as a default that should work for most scripts.
    """
    script_name = os.path.basename(script_path)
    
    # Define specific test arguments for scripts that need them
    script_args = {
        # Test arguments for download_models.py
        "download_models.py": ["--help"],
        
        # Test arguments for generate_steering.py
        "generate_steering.py": [
            "--model_name", "gpt2", 
            "--layers", "1", 
            "--dataset", "hallucination",
            "--max_examples", "1",  # Just use 1 example for testing
            "--beta", "0.5",
            "--aperture", "0.01",
            "--help"  # Adding help to avoid actually running the full process
        ],
        
        # Test arguments for ab_test_steering.py
        "ab_test_steering.py": ["--help"],
        
        # Test arguments for open_ended_test_steering.py
        "open_ended_test_steering.py": ["--help"],
        
        # Test arguments for evaluate_open_ended.py
        "evaluate_open_ended.py": ["--help"]
    }
    
    # Default to --help for scripts without specific test arguments
    return script_args.get(script_name, ["--help"])


@pytest.mark.parametrize("script_path", get_script_files())
def test_script_runs_without_error(script_path, monkeypatch, tmp_path):
    """Test that scripts can be run without errors.
    
    This test:
    1. Uses --help for most scripts to check basic structure without causing side effects
    2. For scripts requiring specific inputs, provides minimal arguments
    3. Mocks input() calls to prevent the script from waiting for user input
    """
    # Setup a temporary directory for test outputs
    test_dir = tmp_path / "script_tests"
    test_dir.mkdir(exist_ok=True)
    
    # Mock input function to automatically return 'n' to any prompts (to cancel actions)
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    
    # Get appropriate arguments for this script
    args = get_test_args(script_path)
    
    try:
        # Run the script with appropriate arguments
        result = subprocess.run(
            ["python", script_path] + args,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,  # Timeout after 15 seconds
            cwd=test_dir,  # Run in the temporary directory
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}  # Ensure script can import project modules
        )
        
        print(f"Script: {script_path}")
        print(f"Args: {args}")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
            
        # Check that the script didn't exit with an error code
        # But accept both 0 (success) and 130 (script canceled by user input)
        assert result.returncode in [0, 130], f"Script {script_path} failed with error code {result.returncode}"
    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {script_path} timed out after 15 seconds") 