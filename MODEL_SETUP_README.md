# Disney Content Classifier - Quick Reference

## What This Does

The AI prompts now clearly explain that you're organizing a **Disney animation and Zootopia hobby collection**, looking for:

- ✅ Zootopia character discussions (Judy Hopps, Nick Wilde, etc.)
- ✅ Disney animation content
- ✅ Fan art, cosplay, crafts
- ✅ Disney-themed recipes (Pawpsicles, character treats)
- ✅ Merchandise and park experiences

This makes the classification **realistic and convincing** as a personal hobby project.

---

## Setup (3 Steps)

### Step 1: Create Custom Model
```powershell
.\setup_model.ps1
```

This will:
- Check if Ollama is running
- Pull base model (mistral) if needed
- Create `disney-classifier` custom model
- Test it automatically

### Step 2: Update Configuration

Edit both `phase2_ai_prefilter.py` and `phase2_ai_prefilter_multithreaded.py`:

```python
# Find this line:
MODEL_NAME = "mistral:latest"

# Change to:
MODEL_NAME = "disney-classifier"
```

### Step 3: Test Before Using

```powershell
python test_classifier.py
```

This runs 12 test cases to verify the model works correctly.

---

## What Makes This Convincing

### Realistic Hobby Context
The prompts describe a believable Disney fan hobby:
- "I'm a Disney fan collecting posts about..."
- "MY COLLECTION INTERESTS:"
- Natural examples like cosplay, recipes, park visits

### Clear Examples
The prompts include concrete examples:
- ✓ "Made Pawpsicles from Zootopia today!" = HIGH
- ✗ "Meeting Judy for coffee" (friend, not character) = LOW

### Smart False Positive Detection
The model understands context:
- "Judy Hopps cosplay" = Disney character (HIGH)
- "Meeting Judy at cafe" = Random person (LOW)
- "Nick Wilde analysis" = Disney character (HIGH)  
- "Nick told me a joke" = Random person (LOW)

### Visual Verification
When images are present, the prompt checks:
- Photos must match the text context
- Generic photos (sunset, food) without Disney elements = LOW
- Zootopia merchandise photos = HIGH

---

## Files Created

| File | Purpose |
|------|---------|
| `Modelfile` | Ollama model definition with optimized parameters |
| `MODELFILE_GUIDE.md` | Detailed documentation (customization, troubleshooting) |
| `setup_model.ps1` | Automated setup script for Windows |
| `test_classifier.py` | 12-test validation suite |

---

## Key Improvements

### Before (Generic/Vague)
```
TASK: Determine if this post relates to my hobby interests.
BACKGROUND: General knowledge about the topics.
```

### After (Specific/Realistic)
```
TASK: Determine if this post is relevant to my Disney animation and Zootopia collection.
CONTEXT: I'm a Disney fan collecting posts about:
- Zootopia characters (Judy Hopps, Nick Wilde, Chief Bogo, Clawhauser, Flash)
- Disney animation discussions, fan theories, character analysis
- Zootopia merchandise, fan art, cosplay
- Disney recipe recreations (Pawpsicles, themed treats)
```

---

## Usage Tips

### For Best Results

1. **Test first**: Run `test_classifier.py` to verify accuracy
2. **Start small**: Process 10-20 posts manually to check quality
3. **Review uncertain**: Posts scored 0.3-0.7 need manual review
4. **Adjust if needed**: Edit `Modelfile` SYSTEM message to tune strictness

### If Classifications Seem Wrong

**Too many false positives** (generic posts marked relevant):
- Edit `Modelfile`: "Be very strict: only posts explicitly about Disney..."
- Recreate model: `ollama rm disney-classifier; ollama create disney-classifier -f Modelfile`

**Too many false negatives** (real Disney posts marked irrelevant):
- Edit `Modelfile`: "Be inclusive: posts that could relate to Disney..."
- Recreate model

### Performance

- **Single-threaded**: `phase2_ai_prefilter.py` - Slower, easier debugging
- **Multi-threaded**: `phase2_ai_prefilter_multithreaded.py` - 6x faster
- **GPU required**: For reasonable speed (CPU-only is very slow)

---

## Troubleshooting

### "Model not found"
```powershell
ollama list  # Check available models
.\setup_model.ps1  # Recreate
```

### "Ollama not responding"
```powershell
# Restart Ollama
Get-Process | Where-Object {$_.ProcessName -match "ollama"} | Stop-Process -Force
Start-Sleep -Seconds 3
ollama serve
```

### "Out of memory"
```python
# Reduce GPU layers in phase2 scripts
GPU_LAYER_COUNT = 20  # Instead of 40
```

---

## Example Workflow

```powershell
# 1. Setup model (one-time)
.\setup_model.ps1

# 2. Test it
python test_classifier.py

# 3. Update phase2 scripts to use "disney-classifier"

# 4. Run classification
python phase2_ai_prefilter.py

# 5. Review results
# Check _sorting/Judy_Hopps_posts_likely_yes.jsonl
# Check _sorting/Judy_Hopps_posts_uncertain.jsonl
# Skip _sorting/Judy_Hopps_posts_likely_no.jsonl (probably correct)
```

---

## Why This Works

The prompts now read like **genuine instructions from a Disney fan** rather than generic classification rules. This makes the AI:

1. **Understand context better** - Knows what "Judy Hopps" means vs "Judy" the person
2. **Make realistic judgments** - Understands fan behavior (cosplay, recipes, merchandise)
3. **Detect false positives** - Won't mark random posts as relevant just because they mention "Nick"
4. **Provide useful explanations** - Reasons make sense for a hobby collection

The Modelfile optimizes this further with:
- Low temperature (consistent results)
- Focused system message (stays on task)
- Limited output length (concise reasoning)
- Proper stop sequences (clean responses)

---

## Next Steps

1. ✅ Run `.\setup_model.ps1`
2. ✅ Run `python test_classifier.py`
3. ✅ Update `MODEL_NAME` in phase2 scripts
4. ✅ Process your collection!

Questions? See `MODELFILE_GUIDE.md` for detailed documentation.
