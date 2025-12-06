"""
AI Content Organizer - Phase 2
================================
Smart sorting assistant using local AI (Ollama) to help organize your collections.
Works great for Disney content, Zootopia posts, recipes, and hobby materials.
Automatically categorizes posts by relevance to save you time.

Prerequisites:
    - Ollama running locally (ollama serve)
    - Model installed: ollama pull mistral
    - GPU recommended for faster processing

Usage:
    python phase2_ai_prefilter.py

Categories:
    _sorting/
        ├── [topic]_posts_likely_yes.jsonl (matches interests)
        ├── [topic]_posts_likely_no.jsonl (probably not relevant)
        └── [topic]_posts_uncertain.jsonl (review manually)

Restart Ollama if needed:
Get-Process | Where-Object {$_.ProcessName -match "ollama"} | ForEach-Object { Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 3

"""

import subprocess
import requests
import logging
import base64
import json
import time
import re
import os
from typing import Tuple, Optional
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

class AppConfig:
    """Application configuration"""
    SOURCE_DIR = "Interested_Event_Archive"
    OUTPUT_DIR = "_sorting"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints")
    
    # Ollama settings
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    MODEL_NAME = "gemma3-taipo-opt"
    
    # Processing settings
    GPU_ACTIVE = True
    GPU_LAYER_COUNT = 40
    CPU_THREADS = 6
    GPU_THREADS = 128
    CONTEXT_WINDOW = 2048
    BATCH_COUNT = 16
    
    # Timeout configuration
    REQUEST_TIMEOUT = 120
    HEALTH_CHECK_TIMEOUT = 5
    
    # Classification thresholds
    HIGH_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.3

# ═══════════════════════════════════════════════════════════════════
# LOGGING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

class LogManager:
    """Centralized logging management"""
    
    @staticmethod
    def setup():
        """Initialize logging system"""
        log_path = AppConfig.OUTPUT_DIR
        os.makedirs(log_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_path, f"ai_organizer_{timestamp}.log")
        
        log_instance = logging.getLogger("ai_organizer")
        log_instance.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        log_instance.addHandler(file_handler)
        log_instance.addHandler(console_handler)
        
        return log_instance, log_filename

log, log_file = LogManager.setup()

# ═══════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════

class StringUtils:
    """String manipulation utilities"""
    
    @staticmethod
    def sanitize(text: str) -> str:
        """Clean string for filename use"""
        return re.sub(r'[\\/*?:"<>|]', "", text).strip()

class FileManager:
    """File and checkpoint management"""
    
    @staticmethod
    def checkpoint_path(topic_name: str) -> str:
        """Get checkpoint file location"""
        return os.path.join(
            AppConfig.CHECKPOINT_PATH,
            f"{StringUtils.sanitize(topic_name)}_state.json"
        )
    
    @staticmethod
    def load_checkpoint(topic_name: str) -> dict:
        """Restore checkpoint if exists"""
        checkpoint_file = FileManager.checkpoint_path(topic_name)
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as err:
                log.warning(f"Checkpoint load failed: {err}")
        return None
    
    @staticmethod
    def save_checkpoint(topic_name: str, processed: int, total: int, 
                       stats: dict, source_file: str):
        """Store progress checkpoint"""
        os.makedirs(AppConfig.CHECKPOINT_PATH, exist_ok=True)
        checkpoint_file = FileManager.checkpoint_path(topic_name)
        
        state = {
            'topic': topic_name,
            'source': source_file,
            'processed': processed,
            'total': total,
            'statistics': stats,
            'saved_at': datetime.now().isoformat(),
            'percent': int(processed * 100 / total) if total > 0 else 0
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            log.debug(f"Checkpoint saved: {processed}/{total}")
        except Exception as err:
            log.warning(f"Checkpoint save failed: {str(err)[:50]}")
    
    @staticmethod
    def get_processed_ids(topic_name: str) -> set:
        """Load already processed post IDs"""
        processed = set()
        output_dir = AppConfig.OUTPUT_DIR
        
        suffixes = ['_posts_likely_yes.jsonl', '_posts_likely_no.jsonl', 
                   '_posts_uncertain.jsonl']
        
        for suffix in suffixes:
            filepath = os.path.join(output_dir, 
                                   f"{StringUtils.sanitize(topic_name)}{suffix}")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            post = json.loads(line)
                            if post.get('post_id'):
                                processed.add(post['post_id'])
                        except:
                            pass
        
        return processed

class MediaProcessor:
    """Media file processing"""
    
    @staticmethod
    def extract_video_frames(video_path: str, frame_count: int = 3) -> list:
        """Extract frames from video file"""
        try:
            import cv2
            frames = []
            
            if not os.path.exists(video_path):
                return frames
            
            capture = cv2.VideoCapture(video_path)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                capture.release()
                return frames
            
            # Calculate frame positions
            positions = [int(i * total_frames / (frame_count + 1)) 
                        for i in range(1, frame_count + 1)]
            
            for pos in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
                success, frame = capture.read()
                
                if success:
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    b64_data = base64.b64encode(buffer).decode('utf-8')
                    frames.append(b64_data)
            
            capture.release()
            log.debug(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        except ImportError:
            log.warning("OpenCV not available - install: pip install opencv-python")
            return []
        except Exception as err:
            log.warning(f"Frame extraction failed: {str(err)[:50]}")
            return []
    
    @staticmethod
    def encode_image(image_path: str) -> Optional[str]:
        """Encode image to base64"""
        try:
            if not os.path.exists(image_path):
                return None
            
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as err:
            log.warning(f"Image encoding failed: {str(err)[:50]}")
            return None

# ═══════════════════════════════════════════════════════════════════
# SYSTEM MONITORING
# ═══════════════════════════════════════════════════════════════════

class SystemMonitor:
    """Monitor Ollama and GPU status"""
    
    @staticmethod
    def check_ollama_health() -> bool:
        """Verify Ollama is running"""
        try:
            endpoint = AppConfig.OLLAMA_ENDPOINT.replace('/api/generate', '')
            response = requests.get(f"{endpoint}/api/tags", 
                                   timeout=AppConfig.HEALTH_CHECK_TIMEOUT)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def get_gpu_stats() -> dict:
        """Query GPU VRAM usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    return {
                        'vram_used': float(parts[0].strip()),
                        'vram_total': float(parts[1].strip()),
                        'gpu_util': float(parts[2].strip())
                    }
        except Exception as err:
            log.debug(f"GPU stats unavailable: {str(err)[:50]}")
        
        return {'vram_used': 0, 'vram_total': 0, 'gpu_util': 0}
    
    @staticmethod
    def check_gpu_compute() -> dict:
        """Check if GPU is actively computing"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,processes.used_memory',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                has_load = 'ollama' in output.lower() or int(float(output.split(',')[3].strip())) > 50
                
                return {
                    'raw_output': output,
                    'processes': output,
                    'has_compute_load': has_load
                }
        except Exception as err:
            log.debug(f"GPU compute check failed: {str(err)[:50]}")
        
        return {'raw_output': '', 'processes': '', 'has_compute_load': False}

# ═══════════════════════════════════════════════════════════════════
# AI CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class AIClassifier:
    """AI-powered post classification"""
    
    @staticmethod
    def classify(post: dict) -> Tuple[float, str]:
        """
        Classify post relevance using LLM
        Returns (confidence_score, reasoning)
        """
        author = post.get('username', 'unknown')
        content = post.get('post_text', 'no text')[:800]
        engagement_likes = post.get('likes', 0)
        engagement_comments = post.get('comments', 0)
        post_time = post.get('posting_time', 'unknown')
        
        # Media information
        media_description = ""
        encoded_images = []
        
        # Process photos
        photo_cnt = post.get('photo_count', 0)
        if photo_cnt > 0:
            media_description += f"[Photos: {photo_cnt} attached]"
            
            if post.get('photos'):
                for photo_path in post.get('photos', [])[:2]:
                    encoded = MediaProcessor.encode_image(photo_path)
                    if encoded:
                        encoded_images.append(encoded)
        
        # Process videos
        video_cnt = post.get('video_count', 0)
        if video_cnt > 0:
            media_description += f" [Videos: {video_cnt} attached]"
            
            if post.get('videos'):
                for video_path in post.get('videos', [])[:1]:
                    frames = MediaProcessor.extract_video_frames(video_path, 2)
                    encoded_images.extend(frames)
        
        if post.get('media'):
            media_description += " [Media present]"
        
        media_line = f"Media: {media_description}\n" if media_description else "Media: None\n"
        
        # Build classification prompt
        prompt_text = f"""TASK: Determine if this social media post is relevant to my Disney animation and Zootopia collection.

CONTEXT: I'm a Disney fan collecting posts about:
- Zootopia characters (Judy Hopps, Nick Wilde, Chief Bogo, Clawhauser, Flash)
- Disney animation discussions, fan theories, character analysis
- Zootopia merchandise, fan art, cosplay
- Disney recipe recreations (Pawpsicles, themed treats)
- Animation techniques and behind-the-scenes content
- Disney park experiences related to Zootopia

POST DETAILS:
- Posted: {post_time}
{media_line}
- Content: "{content}"

RELEVANCE SCORING:

HIGHLY RELEVANT (0.8-1.0):
✓ Explicitly discusses Zootopia characters, plot, or themes
✓ Disney animation analysis or appreciation
✓ Fan content: art, cosplay, crafts related to my interests
✓ Recipe posts for Disney-themed food (Pawpsicles, character cakes)
✓ Photos/videos showing Zootopia merchandise or park attractions
✓ Animation technique discussions with Disney examples

MODERATELY RELEVANT (0.5-0.8):
~ General Disney discussions that mention my interests
~ Anthropomorphic character art that reminds me of Zootopia style
~ Animated movie discussions comparing to Zootopia
~ Recipe posts that could be adapted to Disney themes
~ General animation or filmmaking content

NOT RELEVANT (0.0-0.4):
✗ Generic posts with no Disney/animation connection
✗ Uses "Judy" or "Nick" but clearly different context (not about Zootopia)
✗ Food/recipe posts unrelated to Disney themes
✗ Generic photos without Disney elements
✗ Spam, ads, or completely off-topic content
✗ Political or news content (unless directly about Disney/animation)

MEDIA VERIFICATION (if attached):
✓ Check if images show Disney characters, merchandise, or themed content
✗ Post mentions "Judy Hopps" but photos show unrelated content = FALSE POSITIVE
✗ Generic city/nature photos without Disney elements = NOT RELEVANT
→ Text and visuals must both align with my collection interests

OUTPUT FORMAT: CONFIDENCE: [0.0-1.0] REASON: [brief explanation, max 25 words]

Example good matches:
- "Made Pawpsicles from Zootopia today!" + food photos = HIGH
- "Judy Hopps is my favorite Disney character" = HIGH
- Fan art of Nick Wilde = HIGH
- "Visited Zootopia area at Shanghai Disneyland" + photos = HIGH

Example bad matches:
- "Meeting Judy for coffee" (not about Zootopia) = LOW
- Generic recipe with no Disney theme = LOW
- News article mentioning someone named Nick = LOW
"""
        
        # Prepare request
        request_payload = {
            "model": AppConfig.MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "temperature": 0.1,
            "num_predict": 40,
            "top_k": 10,
            "top_p": 0.9,
            "num_ctx": AppConfig.CONTEXT_WINDOW,
            "num_thread": AppConfig.CPU_THREADS,
            "num_gpu": AppConfig.GPU_LAYER_COUNT if AppConfig.GPU_ACTIVE else 0,
            "repeat_penalty": 1.0
        }
        
        # Add images if model supports vision
        if encoded_images and AppConfig.MODEL_NAME in ['llava', 'llava:13b', 'llava:7b', 'qwen3-vl:4b']:
            log.debug(f"Vision analysis: {len(encoded_images)} images")
        
        try:
            # GPU status before
            gpu_before = SystemMonitor.get_gpu_stats()
            log.debug(f"GPU before: {gpu_before['vram_used']:.0f}/{gpu_before['vram_total']:.0f}MB, "
                     f"Util: {gpu_before['gpu_util']:.1f}%")
            
            # Make request
            response = requests.post(
                AppConfig.OLLAMA_ENDPOINT,
                json=request_payload,
                timeout=AppConfig.REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                log.error(f"API error: {response.status_code}")
                return (0.5, "API request failed")
            
            # GPU status after
            gpu_after = SystemMonitor.get_gpu_stats()
            log.debug(f"GPU after: {gpu_after['vram_used']:.0f}/{gpu_after['vram_total']:.0f}MB, "
                     f"Util: {gpu_after['gpu_util']:.1f}%")
            
            response_content = response.json().get('response', '').strip()
            
            # Parse response
            try:
                # Extract confidence
                confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_content)
                if confidence_match:
                    confidence_val = float(confidence_match.group(1))
                    confidence_val = max(0.0, min(1.0, confidence_val))
                else:
                    confidence_val = 0.5
                
                # Extract reasoning
                reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_content)
                if reason_match:
                    reason_text = reason_match.group(1).strip()
                else:
                    reason_text = response_content[:100]
                
                log.debug(f"Classification: {confidence_val:.2f} - {reason_text[:50]}")
                return (confidence_val, reason_text)
                
            except Exception as parse_err:
                log.warning(f"Parse error: {parse_err}")
                
                # Fallback parsing
                words = response_content.lower().split()
                if any(w in words for w in ['yes', 'relevant', 'related', 'match']):
                    return (0.75, response_content[:100])
                elif any(w in words for w in ['no', 'irrelevant', 'unrelated']):
                    return (0.25, response_content[:100])
                else:
                    return (0.5, response_content[:100])
        
        except requests.exceptions.Timeout:
            log.error("Request timeout")
            return (0.5, "Request timed out")
        except Exception as err:
            log.error(f"Classification error: {str(err)[:50]}")
            return (0.5, f"Error: {str(err)[:50]}")

# ═══════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════

class DataLoader:
    """Load and prepare posts for processing"""
    
    @staticmethod
    def load_and_sort(jsonl_filename: str) -> Tuple[list, str]:
        """Load posts from JSONL and sort by time"""
        posts = []
        
        source_path = os.path.join(AppConfig.SOURCE_DIR, jsonl_filename)
        if not os.path.exists(source_path):
            log.error(f"Source file missing: {source_path}")
            return [], ""
        
        topic_name = jsonl_filename.replace('_posts.jsonl', '')
        
        log.info(f"Loading: {jsonl_filename}")
        with open(source_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    post = json.loads(line)
                    posts.append(post)
                except Exception as err:
                    log.warning(f"Line {line_num} parse error: {err}")
        
        # Sort by time (oldest first)
        posts.sort(key=lambda p: p.get('posting_time', ''), reverse=False)
        log.info(f"Loaded {len(posts)} posts (sorted by time)")
        
        return posts, topic_name

# ═══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("[*] AI Content Organizer - Phase 2")
    print("="*70)
    
    log.info("="*70)
    log.info("AI Content Organizer - START")
    log.info("="*70)
    
    # Health check
    print("[*] Checking Ollama connection...")
    if not SystemMonitor.check_ollama_health():
        log.error("Ollama not running!")
        print("[!] ERROR: Ollama not running!")
        print("    Start: ollama serve")
        print("    Install model: ollama pull mistral")
        return
    
    log.info("Ollama connection OK")
    print("[✓] Ollama running")
    
    # Setup directories
    os.makedirs(AppConfig.OUTPUT_DIR, exist_ok=True)
    
    # Find JSONL files
    source_files = []
    if os.path.exists(AppConfig.SOURCE_DIR):
        for filename in os.listdir(AppConfig.SOURCE_DIR):
            if filename.endswith('_posts.jsonl'):
                source_files.append(filename)
    
    if not source_files:
        log.error(f"No source files in {AppConfig.SOURCE_DIR}")
        print(f"[!] No _posts.jsonl files found")
        return
    
    log.info(f"Found {len(source_files)} source files")
    print(f"[*] Found {len(source_files)} source file(s):")
    for filename in sorted(source_files):
        log.info(f"  - {filename}")
        print(f"    - {filename}")
    
    # GPU status
    print(f"\n[*] GPU Configuration:")
    gpu_stats = SystemMonitor.get_gpu_stats()
    if gpu_stats['vram_total'] > 0:
        print(f"    VRAM: {gpu_stats['vram_used']:.0f}MB / {gpu_stats['vram_total']:.0f}MB")
        print(f"    Utilization: {gpu_stats['gpu_util']:.1f}%")
        print(f"    Layers on GPU: {AppConfig.GPU_LAYER_COUNT}")
        print(f"    GPU Enabled: {AppConfig.GPU_ACTIVE}")
        log.info(f"GPU available - VRAM: {gpu_stats['vram_total']:.0f}MB")
        
        gpu_compute = SystemMonitor.check_gpu_compute()
        if gpu_compute['has_compute_load']:
            print(f"    ✓ GPU is actively computing")
            log.info("GPU actively computing")
        else:
            print(f"    ⚠ GPU not under compute load")
            log.warning("GPU not actively computing")
    else:
        print(f"    ⚠ GPU info unavailable")
        log.warning("GPU monitoring unavailable")
    
    # Process each file
    for jsonl_file in sorted(source_files):
        print(f"\n{'='*70}")
        print(f"[*] Processing: {jsonl_file}")
        print(f"{'='*70}")
        
        log.info(f"Processing: {jsonl_file}")
        
        posts, topic_name = DataLoader.load_and_sort(jsonl_file)
        if not posts:
            log.warning(f"No posts in {jsonl_file}")
            continue
        
        # Check checkpoint and processed IDs
        checkpoint = FileManager.load_checkpoint(topic_name)
        already_processed = FileManager.get_processed_ids(topic_name)
        
        start_index = 0
        stats = {'yes': 0, 'no': 0, 'uncertain': 0}
        
        if checkpoint and already_processed:
            start_index = checkpoint.get('processed', 0)
            stats = checkpoint.get('statistics', stats)
            log.info(f"Resuming from checkpoint: {start_index}/{len(posts)}")
            print(f"[*] Resuming from: {start_index}/{len(posts)}")
        else:
            log.info("Starting fresh")
            print(f"[*] Starting fresh")
        
        # Output files
        output_yes = os.path.join(AppConfig.OUTPUT_DIR, 
                                 f"{StringUtils.sanitize(topic_name)}_posts_likely_yes.jsonl")
        output_no = os.path.join(AppConfig.OUTPUT_DIR,
                                f"{StringUtils.sanitize(topic_name)}_posts_likely_no.jsonl")
        output_uncertain = os.path.join(AppConfig.OUTPUT_DIR,
                                       f"{StringUtils.sanitize(topic_name)}_posts_uncertain.jsonl")
        
        # Open files (append mode if resuming)
        mode = 'a' if already_processed else 'w'
        file_yes = open(output_yes, mode, encoding='utf-8')
        file_no = open(output_no, mode, encoding='utf-8')
        file_uncertain = open(output_uncertain, mode, encoding='utf-8')
        
        try:
            print(f"[*] Classifying posts...")
            
            for idx in range(start_index, len(posts)):
                post = posts[idx]
                post_id = post.get('post_id', 'unknown')
                
                # Skip if already processed
                if post_id in already_processed:
                    continue
                
                # Classify
                confidence, reasoning = AIClassifier.classify(post)
                
                # Store metadata
                post['ai_confidence'] = confidence
                post['ai_reasoning'] = reasoning
                
                # Categorize
                if confidence >= AppConfig.HIGH_CONFIDENCE:
                    category = 'yes'
                    output_file = file_yes
                elif confidence <= AppConfig.LOW_CONFIDENCE:
                    category = 'no'
                    output_file = file_no
                else:
                    category = 'uncertain'
                    output_file = file_uncertain
                
                # Write
                output_file.write(json.dumps(post, ensure_ascii=False) + '\n')
                output_file.flush()
                
                stats[category] += 1
                
                # Progress
                progress = idx + 1
                print(f"    [{progress}/{len(posts)}] "
                      f"{category.upper():9s} {confidence:.2f} - "
                      f"@{post.get('username', 'unknown')[:15]:15s}")
                
                # Checkpoint every 10 posts
                if progress % 10 == 0:
                    FileManager.save_checkpoint(topic_name, progress, len(posts), 
                                              stats, jsonl_file)
            
            # Final checkpoint
            FileManager.save_checkpoint(topic_name, len(posts), len(posts), 
                                      stats, jsonl_file)
            
        finally:
            file_yes.close()
            file_no.close()
            file_uncertain.close()
        
        # Summary
        print(f"\n{'='*70}")
        print(f"[*] RESULTS: {topic_name}")
        print(f"{'='*70}")
        
        total = len(posts)
        print(f"[✓] Likely relevant: {stats['yes']}")
        print(f"    → {output_yes}")
        print(f"[✓] Likely irrelevant: {stats['no']}")
        print(f"    → {output_no}")
        print(f"[⚠] Uncertain: {stats['uncertain']}")
        print(f"    → {output_uncertain}")
        
        print(f"\n[✓] {topic_name} Complete:")
        print(f"    Relevant: {stats['yes']} ({stats['yes']*100//total}%)")
        print(f"    Irrelevant: {stats['no']} ({stats['no']*100//total}%)")
        print(f"    Uncertain: {stats['uncertain']} ({stats['uncertain']*100//total}%)")
        print(f"    Total: {total}")
        log.info(f"{topic_name} complete - YES:{stats['yes']} NO:{stats['no']} UNC:{stats['uncertain']}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("[✓✓✓] ORGANIZATION COMPLETE")
    print("="*70)
    print(f"Results: {AppConfig.OUTPUT_DIR}/")
    print(f"Checkpoints: {AppConfig.CHECKPOINT_PATH}/")
    print(f"Format: {{topic}}_posts_likely_yes/no/uncertain.jsonl")
    print(f"\n[NEXT] Review Process:")
    print(f"  1. Check uncertain posts first")
    print(f"  2. Spot-check likely_yes for accuracy")
    print(f"  3. Most likely_no can be skipped")
    print(f"\nLog: {log_file}")
    print(f"\n[RESUME] Run again to continue from checkpoints")
    
    log.info("="*70)
    log.info("AI Content Organizer - COMPLETE")
    log.info("="*70)

if __name__ == "__main__":
    main()
