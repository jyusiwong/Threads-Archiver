# Quick Setup Script for Disney Content Classifier
# Run this to create and test your custom Ollama model

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Disney Content Classifier Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Ollama is running
Write-Host "[1/5] Checking Ollama..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  ✓ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Ollama not running!" -ForegroundColor Red
    Write-Host "  Start it with: ollama serve" -ForegroundColor White
    exit 1
}

# Check if base model exists
Write-Host "`n[2/5] Checking base model (mistral)..." -ForegroundColor Yellow
$models = ollama list | Out-String
if ($models -match "mistral") {
    Write-Host "  ✓ Mistral model found" -ForegroundColor Green
} else {
    Write-Host "  ✗ Mistral not found. Pulling..." -ForegroundColor Yellow
    ollama pull mistral
    Write-Host "  ✓ Mistral downloaded" -ForegroundColor Green
}

# Create custom model
Write-Host "`n[3/5] Creating disney-classifier model..." -ForegroundColor Yellow
try {
    ollama create disney-classifier -f Modelfile
    Write-Host "  ✓ Model created successfully" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Model creation failed" -ForegroundColor Red
    Write-Host "  Check Modelfile syntax" -ForegroundColor White
    exit 1
}

# Test model
Write-Host "`n[4/5] Testing model..." -ForegroundColor Yellow
$testPrompt = "Evaluate this post: 'Just finished my Judy Hopps cosplay! So excited to wear it to the Disney convention!'"
Write-Host "  Test prompt: $testPrompt" -ForegroundColor Gray

$testResult = ollama run disney-classifier $testPrompt --verbose=false
Write-Host "  Model response:" -ForegroundColor Gray
Write-Host "  $testResult" -ForegroundColor White

if ($testResult -match "CONFIDENCE:\s*([0-9.]+)") {
    $confidence = $matches[1]
    if ([float]$confidence -gt 0.7) {
        Write-Host "  ✓ Model working correctly (confidence: $confidence)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Unexpected confidence score: $confidence (expected >0.7)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ Response format unexpected" -ForegroundColor Yellow
}

# Configuration instructions
Write-Host "`n[5/5] Next steps:" -ForegroundColor Yellow
Write-Host "  1. Update phase2 scripts to use 'disney-classifier':" -ForegroundColor White
Write-Host "     MODEL_NAME = 'disney-classifier'" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Run Phase 2 classification:" -ForegroundColor White
Write-Host "     python phase2_ai_prefilter.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Check results in _sorting/ directory" -ForegroundColor White
Write-Host ""
Write-Host "  For detailed help, see: MODELFILE_GUIDE.md" -ForegroundColor Cyan

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
