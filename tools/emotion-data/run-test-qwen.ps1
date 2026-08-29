param(
  [string]$Candidates = "outputs\emotion-data\speaker-candidates\candidates.jsonl",
  [string]$ModelDir = "checkpoints\qwen0.6bemo4-merge",
  [string]$OutputRoot = "outputs\emotion-data\qwen-test"
)

. "$env:USERPROFILE\.codex\skills\powershell-utf8\scripts\Set-CodexPowerShellUtf8.ps1" -Quiet
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$candidatePath = (Resolve-Path $Candidates).Path
$modelPath = (Resolve-Path $ModelDir).Path
$root = [IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $root | Out-Null

foreach ($variant in @("a", "b")) {
  $output = Join-Path $root "annotations-$variant.jsonl"
  & $python -m indextts.emotion.cli annotate-test-qwen `
    --candidates $candidatePath `
    --output $output `
    --model-dir $modelPath `
    --batch-size 8 `
    --max-new-tokens 128 `
    --prompt-variant $variant
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $python -m indextts.emotion.cli merge-annotations `
  --candidates $candidatePath `
  --annotation-a (Join-Path $root "annotations-a.jsonl") `
  --annotation-b (Join-Path $root "annotations-b.jsonl") `
  --output (Join-Path $root "merged") `
  --test-only
exit $LASTEXITCODE
