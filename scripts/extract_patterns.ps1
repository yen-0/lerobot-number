# scripts/extract_patterns.ps1
param (
    [string]$Mode = "process",
    [string]$Crop = "386 60 642 238",
    [string]$NewRepoId = "yen-0/so101-writei-patterns"
)

# Load config.env if it exists
if (Test-Path "config.env") {
    Get-Content "config.env" | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim())
    }
}

$dataset_repo_id = [System.Environment]::GetEnvironmentVariable("DATASET_REPO_ID")
if (-not $dataset_repo_id) { $dataset_repo_id = "k1000dai/so101-writei" }

$args_list = @("--repo_id", $dataset_repo_id, "--mode", $Mode)

if ($Crop) {
    # Crop is a space-separated string, e.g. "100 100 300 300"
    $crop_vals = $Crop.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    $args_list += "--crop"
    $args_list += $crop_vals
}

if ($NewRepoId) {
    $args_list += @("--new_repo_id", $NewRepoId, "--push_to_hub")
}

# Set PYTHONPATH to include the src directory
$env:PYTHONPATH = "src;" + $env:PYTHONPATH

# Detect python executable (either local .venv or system python)
$python_cmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $python_cmd = ".venv\Scripts\python.exe"
}

& $python_cmd src/lerobot/scripts/lerobot_extract_patterns.py $args_list
