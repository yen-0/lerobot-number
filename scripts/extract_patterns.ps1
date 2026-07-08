# scripts/extract_patterns.ps1

# Load optional local secrets first, then tracked non-secret config.
foreach ($configFile in @("config.env", "config.shared.env")) {
    if (-not (Test-Path $configFile)) {
        continue
    }
    Get-Content $configFile | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
        $line = $_ -replace "^\s*export\s+", ""
        $name, $value = $line.Split('=', 2)
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
