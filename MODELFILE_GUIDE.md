# Ollama Model Setup Guide

## Quick Start

### 1. Create the Custom Model

```powershell
# Navigate to project directory
cd "d:\Github\Hong-Kong-Fire-Threads-Archive\Hong-Kong-Fire-Threads-Archive"

# Create the Disney classifier model
ollama create disney-classifier -f Modelfile
```

### 2. Update Configuration

Edit `phase2_ai_prefilter.py` and `phase2_ai_prefilter_multithreaded.py`:

```python
class AppConfig:  # or SystemConfig
    # Change this line:
    MODEL_NAME = "mistral:latest"
    
    # To this:
    MODEL_NAME = "disney-classifier"
```

### 3. Test the Model

```powershell
# Test with sample prompt
ollama run disney-classifier "Evaluate this post: 'Just finished watching Zootopia again! Judy Hopps is such an inspiring character.'"
```

Expected output:
```
CONFIDENCE: 0.95
REASON: Explicitly discusses Zootopia and character Judy Hopps with appreciation context.
```

---

## Advanced Configuration

### Creating Models from Different Base Models

#### Option 1: Mistral (Balanced - Recommended)
```bash
# Default Modelfile uses mistral:latest
ollama create disney-classifier -f Modelfile
```

#### Option 2: Gemma (Faster, less accurate)
```bash
# Edit Modelfile: FROM gemma:7b
ollama create disney-classifier-fast -f Modelfile
```

#### Option 3: Llama3 (More accurate, slower)
```bash
# Edit Modelfile: FROM llama3:8b
ollama create disney-classifier-accurate -f Modelfile
```

### For Vision Support (Image Analysis)

If you want to analyze images in posts, create a vision model:

```bash
# Edit Modelfile: FROM llava:13b
ollama create disney-classifier-vision -f Modelfile
```

Then update config:
```python
class SystemConfig:
    VISION_MODEL = "disney-classifier-vision"
    ENABLE_VISION = True
```

---

## Model Customization

### Adjusting Classification Strictness

Edit `Modelfile` SYSTEM message:

**More Strict (fewer matches):**
```
Be very strict: only posts explicitly about Disney animation should score above 0.7.
Coincidental name matches should score below 0.3.
```

**More Lenient (more matches):**
```
Be inclusive: posts that could relate to Disney themes should score above 0.5.
Give benefit of doubt to borderline cases.
```

### Adjusting Response Length

In `Modelfile`:
```
PARAMETER num_predict 60  # Longer explanations
PARAMETER num_predict 30  # Shorter explanations
```

### Adjusting Creativity vs Consistency

```
PARAMETER temperature 0.0   # Most consistent (robotic)
PARAMETER temperature 0.1   # Recommended (balanced)
PARAMETER temperature 0.3   # More varied responses
```

---

## Testing Your Model

### Test Script

Create `test_model.py`:

```python
import requests
import json

def test_classification(text):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'disney-classifier',
            'prompt': f'Evaluate this post: "{text}"',
            'stream': False
        }
    )
    return response.json()['response']

# Test cases
test_posts = [
    "Just watched Zootopia again! Love Judy Hopps!",  # Should be HIGH (0.9+)
    "Meeting my friend Judy for lunch today",         # Should be LOW (0.1-)
    "Made Pawpsicles from the movie recipe",          # Should be HIGH (0.8+)
    "Beautiful sunset photo from my vacation",        # Should be LOW (0.1-)
]

for post in test_posts:
    print(f"\nPost: {post}")
    print(f"Result: {test_classification(post)}")
```

Run: `python test_model.py`

---

## Troubleshooting

### Model Not Found
```powershell
# List all models
ollama list

# Recreate if missing
ollama create disney-classifier -f Modelfile
```

### Model Too Slow
```python
# In config, reduce GPU layers
GPU_LAYER_COUNT = 20  # Instead of 40
```

### Inconsistent Results
```python
# Lower temperature in Modelfile
PARAMETER temperature 0.05
```

### Out of Memory
```bash
# Use smaller base model
# Edit Modelfile: FROM mistral:7b (instead of :latest)
ollama create disney-classifier -f Modelfile
```

---

## Model Management

### List Models
```bash
ollama list
```

### Delete Model
```bash
ollama rm disney-classifier
```

### Update Model (after editing Modelfile)
```bash
ollama rm disney-classifier
ollama create disney-classifier -f Modelfile
```

### Export Model (for backup)
```bash
# Not directly supported, but you can copy:
# Windows: %USERPROFILE%\.ollama\models\
# Linux/Mac: ~/.ollama/models/
```

---

## Performance Optimization

### GPU Acceleration
Ensure Ollama uses GPU:
```bash
# Check GPU usage while running
nvidia-smi

# Should show ollama process using VRAM
```

### Batch Processing
In `phase2_ai_prefilter_multithreaded.py`:
```python
WORKER_COUNT = 6          # Increase for more parallelism
BATCH_SIZE_PER_WORKER = 5 # Adjust based on RAM
```

### Context Window
```python
# In Modelfile for longer posts
PARAMETER num_ctx 4096  # Instead of 2048
```

---

## Production Tips

1. **Test thoroughly** with your actual data before processing everything
2. **Monitor GPU temperature** during long processing runs
3. **Save checkpoints frequently** (already implemented)
4. **Review uncertain posts** manually (scores 0.3-0.7)
5. **Keep backups** of your Modelfile and configurations

---

## Example Workflow

```powershell
# 1. Create model
ollama create disney-classifier -f Modelfile

# 2. Test it
ollama run disney-classifier "Test: Judy Hopps fan art"

# 3. Update config files (MODEL_NAME)

# 4. Run Phase 2
python phase2_ai_prefilter.py

# 5. Review results in _sorting/ directory
```

---

## Need Help?

- Ollama docs: https://github.com/ollama/ollama
- Model library: https://ollama.com/library
- Check logs in `_sorting/ai_organizer_*.log`
