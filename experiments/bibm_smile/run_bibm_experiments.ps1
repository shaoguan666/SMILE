param(
  [switch]$DryRun,
  [string[]]$Models = @("Backbone", "SMILE-Full", "SMILE w/o Density", "SMILE w/o CoMiss", "SMILE w/o TimeMod", "SMILE w/o T-MNAR", "SMILE w/o T-PE", "SMILE w/o All-MNAR"),
  [string[]]$Datasets = @("c12", "c19", "mimic_mortality", "mimic_decompensation", "mimic_phenotyping", "mimic_lengthofstay"),
  [int[]]$Seeds = @(1, 42, 3407),
  [string]$Python = "python",
  [string]$Devices = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$argsList = @(
  "$scriptDir\run_bibm_experiments.py",
  "--python-executable", $Python,
  "--models"
) + $Models + @(
  "--datasets"
) + $Datasets + @(
  "--seeds"
) + ($Seeds | ForEach-Object { "$_" })

if ($DryRun) { $argsList += "--dry-run" }
if ($Devices -ne "") { $argsList += @("--devices", $Devices) }

& $Python @argsList
