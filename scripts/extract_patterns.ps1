# scripts/extract_patterns.ps1

# Load config.env if it exists
if (Test-Path "config.env") {
    Get-Content "config.env" | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim())
    }
}

# Set PYTHONPATH to include the src directory
$env:PYTHONPATH = "src;" + $env:PYTHONPATH

# Detect python executable (either local .venv or system python)
$python_cmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $python_cmd = ".venv\Scripts\python.exe"
}

& $python_cmd src/lerobot/scripts/lerobot_extract_patterns.py
