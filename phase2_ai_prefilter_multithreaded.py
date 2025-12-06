"""
AI Content Organizer - Phase 2 (MULTI-THREADED + VISION)
==========================================================
Advanced parallel processing for fast content classification.
Supports GPT-4 Vision, Claude Vision, Gemini Vision and local LLaVA models.
Perfect for bulk sorting of Disney collections, Zootopia posts, and hobby content.

Speed: Processes 100+ posts/minute with proper GPU setup
Vision: Analyzes photos and video frames for better accuracy

Prerequisites:
    1. Local Model (Ollama):
       - ollama serve
       - ollama pull llava:13b (for vision)
       - ollama pull mistral (for text)
    
    2. Cloud Vision (Optional):
       - Set API_MODE = 'openai' or 'anthropic' or 'google'
       - Configure API keys in environment

Usage:
    python phase2_ai_prefilter_multithreaded.py

Performance Tips:
    - Increase WORKER_COUNT for more parallel processing
    - Enable GPU_ACTIVE for faster inference
    - Use vision models for media-rich posts

"""

import concurrent.futures
import subprocess
import requests
import logging
import base64
import json
import time
import re
import os
from typing import Tuple, Optional, List
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION SYSTEM
# ═══════════════════════════════════════════════════════════════════

class SystemConfig:
    """Global system configuration"""
    
    # Directory structure
    INPUT_FOLDER = "Interested_Event_Archive"
    OUTPUT_FOLDER = "_sorting"
    STATE_FOLDER = os.path.join(OUTPUT_FOLDER, "checkpoints")
    
    # API Configuration
    API_MODE = 'ollama'  # 'ollama' | 'openai' | 'anthropic' | 'google'
    
    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "gemma3-taipo-opt"
    VISION_MODEL = "llava:13b"
    
    # Cloud API settings (if enabled)
    OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    GOOGLE_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # Performance settings
    WORKER_COUNT = 6
    BATCH_SIZE_PER_WORKER = 5
    
    # GPU settings
    ENABLE_GPU = True
    GPU_LAYERS = 40
    THREAD_COUNT_CPU = 6
    THREAD_COUNT_GPU = 128
    CONTEXT_LENGTH = 2048
    
    # Timeout configuration
    API_TIMEOUT = 120
    HEALTH_TIMEOUT = 5
    
    # Classification thresholds
    THRESHOLD_HIGH = 0.7
    THRESHOLD_LOW = 0.3
    
    # Media processing
    ENABLE_VISION = True
    MAX_IMAGES_PER_POST = 2
    MAX_VIDEO_FRAMES = 3
    
    # Checkpoint frequency
    CHECKPOINT_INTERVAL = 20

# ═══════════════════════════════════════════════════════════════════
# LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════

class LoggerSetup:
    """Configure application logging"""
    
    @staticmethod
    def initialize():
        """Setup logging infrastructure"""
        log_dir = SystemConfig.OUTPUT_FOLDER
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f"ai_organizer_mt_{timestamp}.log")
        
        logger = logging.getLogger("ai_organizer_mt")
        logger.setLevel(logging.DEBUG)
        
        # File logging
        fh = logging.FileHandler(logfile, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console logging
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Format
        fmt = logging.Formatter(
            '%(asctime)s | %(threadName)-10s | %(levelname)-5s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger, logfile

logger, logfile_path = LoggerSetup.initialize()

# ═══════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════

class TextHelper:
    """Text processing utilities"""
    
    @staticmethod
    def clean_for_path(text: str) -> str:
        """Sanitize text for filesystem use"""
        return re.sub(r'[\\/*?:"<>|]', "", text).strip()

# ═══════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class StateManager:
    """Checkpoint and resume functionality"""
    
    @staticmethod
    def state_file(topic: str) -> str:
        """Get state file path"""
        return os.path.join(
            SystemConfig.STATE_FOLDER,
            f"{TextHelper.clean_for_path(topic)}_progress.json"
        )
    
    @staticmethod
    def restore_state(topic: str) -> dict:
        """Load saved state"""
        filepath = StateManager.state_file(topic)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"State restore failed: {e}")
        return None
    
    @staticmethod
    def persist_state(topic: str, current: int, total: int, 
                     counters: dict, source: str):
        """Save current state"""
        os.makedirs(SystemConfig.STATE_FOLDER, exist_ok=True)
        filepath = StateManager.state_file(topic)
        
        data = {
            'topic_name': topic,
            'source_file': source,
            'current_position': current,
            'total_items': total,
            'counters': counters,
            'timestamp': datetime.now().isoformat(),
            'completion_pct': int(current * 100 / total) if total > 0 else 0
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"State saved: {current}/{total}")
        except Exception as e:
            logger.warning(f"State save failed: {str(e)[:50]}")
    
    @staticmethod
    def get_completed_ids(topic: str) -> set:
        """Load IDs of already processed posts"""
        completed = set()
        out_dir = SystemConfig.OUTPUT_FOLDER
        
        file_suffixes = ['_posts_likely_yes.jsonl', '_posts_likely_no.jsonl', 
                        '_posts_uncertain.jsonl']
        
        for suffix in file_suffixes:
            path = os.path.join(out_dir, 
                               f"{TextHelper.clean_for_path(topic)}{suffix}")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if item.get('post_id'):
                                completed.add(item['post_id'])
                        except:
                            pass
        
        return completed

# ═══════════════════════════════════════════════════════════════════
# MEDIA HANDLING
# ═══════════════════════════════════════════════════════════════════

class MediaHandler:
    """Process images and videos"""
    
    @staticmethod
    def extract_frames_from_video(video_file: str, num_frames: int = 3) -> list:
        """Extract frames from video"""
        try:
            import cv2
            frame_list = []
            
            if not os.path.exists(video_file):
                return frame_list
            
            cap = cv2.VideoCapture(video_file)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total == 0:
                cap.release()
                return frame_list
            
            # Calculate positions
            positions = [int(i * total / (num_frames + 1)) 
                        for i in range(1, num_frames + 1)]
            
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                
                if ret:
                    # Encode as JPEG
                    _, buf = cv2.imencode('.jpg', frame)
                    encoded = base64.b64encode(buf).decode('utf-8')
                    frame_list.append(encoded)
            
            cap.release()
            logger.debug(f"Extracted {len(frame_list)} frames from {video_file}")
            return frame_list
            
        except ImportError:
            logger.warning("OpenCV unavailable - install: pip install opencv-python")
            return []
        except Exception as e:
            logger.warning(f"Frame extraction failed: {str(e)[:50]}")
            return []
    
    @staticmethod
    def encode_image_file(image_file: str) -> Optional[str]:
        """Encode image to base64"""
        try:
            if not os.path.exists(image_file):
                return None
            
            with open(image_file, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Image encoding failed: {str(e)[:50]}")
            return None

# ═══════════════════════════════════════════════════════════════════
# SYSTEM HEALTH MONITORING
# ═══════════════════════════════════════════════════════════════════

class HealthMonitor:
    """Monitor Ollama and GPU health"""
    
    @staticmethod
    def verify_ollama() -> bool:
        """Check if Ollama is responsive"""
        if SystemConfig.API_MODE != 'ollama':
            return True
        
        try:
            base = SystemConfig.OLLAMA_BASE_URL.replace('/api/generate', '')
            resp = requests.get(f"{base}/api/tags", 
                               timeout=SystemConfig.HEALTH_TIMEOUT)
            return resp.status_code == 200
        except:
            return False
    
    @staticmethod
    def query_gpu() -> dict:
        """Get GPU statistics"""
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
                        'mem_used': float(parts[0].strip()),
                        'mem_total': float(parts[1].strip()),
                        'utilization': float(parts[2].strip())
                    }
        except Exception as e:
            logger.debug(f"GPU query failed: {str(e)[:50]}")
        
        return {'mem_used': 0, 'mem_total': 0, 'utilization': 0}

# ═══════════════════════════════════════════════════════════════════
# VISION API INTEGRATION
# ═══════════════════════════════════════════════════════════════════

class VisionAPI:
    """Handle cloud vision API calls"""
    
    @staticmethod
    def call_openai_vision(prompt: str, images: List[str]) -> str:
        """OpenAI GPT-4 Vision"""
        try:
            import openai
            openai.api_key = SystemConfig.OPENAI_KEY
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            for img in images[:SystemConfig.MAX_IMAGES_PER_POST]:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=100,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return ""
    
    @staticmethod
    def call_anthropic_vision(prompt: str, images: List[str]) -> str:
        """Anthropic Claude Vision"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=SystemConfig.ANTHROPIC_KEY)
            
            content = [{"type": "text", "text": prompt}]
            
            for img in images[:SystemConfig.MAX_IMAGES_PER_POST]:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img
                    }
                })
            
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=100,
                messages=[{"role": "user", "content": content}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic Vision error: {e}")
            return ""
    
    @staticmethod
    def call_google_vision(prompt: str, images: List[str]) -> str:
        """Google Gemini Vision"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=SystemConfig.GOOGLE_KEY)
            
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Convert base64 to PIL Image
            from PIL import Image
            import io
            
            image_parts = []
            for img in images[:SystemConfig.MAX_IMAGES_PER_POST]:
                img_data = base64.b64decode(img)
                image_parts.append(Image.open(io.BytesIO(img_data)))
            
            response = model.generate_content([prompt] + image_parts)
            return response.text
        except Exception as e:
            logger.error(f"Google Vision error: {e}")
            return ""

# ═══════════════════════════════════════════════════════════════════
# AI CLASSIFICATION CORE
# ═══════════════════════════════════════════════════════════════════

class ContentClassifier:
    """Multi-threaded post classification"""
    
    @staticmethod
    def evaluate(post: dict) -> Tuple[float, str]:
        """
        Classify post using AI (with optional vision)
        Returns (confidence_score, explanation)
        """
        username = post.get('username', 'unknown')
        text_content = post.get('post_text', 'no text')[:800]
        like_count = post.get('likes', 0)
        comment_count = post.get('comments', 0)
        timestamp = post.get('posting_time', 'unknown')
        
        # Media processing
        media_info = ""
        image_encodings = []
        
        # Photos
        photo_count = post.get('photo_count', 0)
        if photo_count > 0:
            media_info += f"[Photos: {photo_count}]"
            
            if SystemConfig.ENABLE_VISION and post.get('photos'):
                for photo_path in post.get('photos', [])[:SystemConfig.MAX_IMAGES_PER_POST]:
                    encoded = MediaHandler.encode_image_file(photo_path)
                    if encoded:
                        image_encodings.append(encoded)
        
        # Videos
        video_count = post.get('video_count', 0)
        if video_count > 0:
            media_info += f" [Videos: {video_count}]"
            
            if SystemConfig.ENABLE_VISION and post.get('videos'):
                for video_path in post.get('videos', [])[:1]:
                    frames = MediaHandler.extract_frames_from_video(
                        video_path, 
                        SystemConfig.MAX_VIDEO_FRAMES
                    )
                    image_encodings.extend(frames)
        
        if post.get('media'):
            media_info += " [Media attached]"
        
        media_text = f"Media: {media_info}\n" if media_info else "Media: None\n"
        
        # Build prompt
        classification_prompt = f"""TASK: Evaluate if this post belongs in my Disney/Zootopia hobby collection.

MY COLLECTION INTERESTS:
- Zootopia: Characters (Judy Hopps, Nick Wilde, Flash, Chief Bogo, etc.)
- Disney animation films, character development, storytelling
- Fan creations: art, cosplay, crafts, AMVs
- Disney-themed recipes (Pawpsicles, character treats, park food recreations)
- Animation techniques, storyboarding, character design
- Disney parks: Zootopia attractions, character meet-and-greets
- Merchandise: plushies, figures, apparel

POST INFO:
- Time: {timestamp}
{media_text}
- Text: "{text_content}"

RELEVANCE ASSESSMENT:

HIGHLY RELEVANT (0.8-1.0):
✓ Direct Zootopia character names/quotes/references
✓ Disney animation analysis, appreciation, or critique
✓ Fan art, cosplay, or crafts featuring my interests
✓ Recipe tutorials for Disney-themed food
✓ Photos/videos of Zootopia merchandise or park experiences
✓ Behind-the-scenes animation content from Disney
✓ Character analysis or fan theories about Zootopia

MODERATELY RELEVANT (0.5-0.8):
~ Disney movie discussions mentioning Zootopia
~ Anthropomorphic art similar to Zootopia style
~ General animation tutorials or industry news
~ Recipe adaptations that could work for Disney themes
~ Comparison posts: "movies like Zootopia"

NOT RELEVANT (0.0-0.4):
✗ Common names (Judy, Nick) in non-Zootopia context
✗ Generic recipes without Disney theme potential
✗ News/politics (unless Disney company news)
✗ Generic photos: sunsets, food, people (no Disney connection)
✗ Spam, advertisements, off-topic content
✗ Other fandoms without Disney crossover

MEDIA VERIFICATION:
✓ Verify images show Disney characters/merchandise/themed content
✗ Text says "Judy Hopps" but images show random person = FALSE POSITIVE
✗ Post mentions "Nick" but clearly about someone's friend = NOT RELEVANT
→ Both text AND visuals must align with my collection focus

OUTPUT: CONFIDENCE: [0.0-1.0] REASON: [max 25 words]

QUICK EXAMPLES:
✓ "Just finished my Judy Hopps cosplay!" + costume photos = 0.95
✓ "Zootopia teaches important lessons about prejudice" = 0.90
✓ "Made Pawpsicles recipe from the movie" + food photos = 0.92
✗ "Going to meet Judy at the cafe" (friend, not character) = 0.10
✗ Generic city skyline photo, no Disney mention = 0.05
"""
        
        # API call based on mode
        try:
            if SystemConfig.API_MODE == 'openai' and image_encodings:
                response_text = VisionAPI.call_openai_vision(
                    classification_prompt, image_encodings
                )
            elif SystemConfig.API_MODE == 'anthropic' and image_encodings:
                response_text = VisionAPI.call_anthropic_vision(
                    classification_prompt, image_encodings
                )
            elif SystemConfig.API_MODE == 'google' and image_encodings:
                response_text = VisionAPI.call_google_vision(
                    classification_prompt, image_encodings
                )
            else:
                # Ollama (local)
                request_data = {
                    "model": SystemConfig.VISION_MODEL if image_encodings else SystemConfig.LLM_MODEL,
                    "prompt": classification_prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "num_predict": 40,
                    "top_k": 10,
                    "top_p": 0.9,
                    "num_ctx": SystemConfig.CONTEXT_LENGTH,
                    "num_thread": SystemConfig.THREAD_COUNT_CPU,
                    "num_gpu": SystemConfig.GPU_LAYERS if SystemConfig.ENABLE_GPU else 0,
                    "repeat_penalty": 1.0
                }
                
                # Add images for vision models
                if image_encodings and SystemConfig.VISION_MODEL in ['llava', 'llava:13b', 'llava:7b']:
                    request_data['images'] = image_encodings
                    logger.debug(f"Vision mode: {len(image_encodings)} images")
                
                response = requests.post(
                    SystemConfig.OLLAMA_BASE_URL,
                    json=request_data,
                    timeout=SystemConfig.API_TIMEOUT
                )
                
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code}")
                    return (0.5, "API call failed")
                
                response_text = response.json().get('response', '').strip()
            
            # Parse response
            try:
                # Extract confidence
                conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_text)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                else:
                    confidence = 0.5
                
                # Extract reasoning
                reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_text)
                if reason_match:
                    reasoning = reason_match.group(1).strip()
                else:
                    reasoning = response_text[:100]
                
                logger.debug(f"Result: {confidence:.2f} - {reasoning[:50]}")
                return (confidence, reasoning)
                
            except Exception as parse_err:
                logger.warning(f"Parse error: {parse_err}")
                
                # Fallback
                text_lower = response_text.lower()
                if any(w in text_lower for w in ['yes', 'relevant', 'related', 'match']):
                    return (0.75, response_text[:100])
                elif any(w in text_lower for w in ['no', 'irrelevant', 'unrelated']):
                    return (0.25, response_text[:100])
                else:
                    return (0.5, response_text[:100])
        
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return (0.5, "Request timed out")
        except Exception as e:
            logger.error(f"Classification error: {str(e)[:50]}")
            return (0.5, f"Error: {str(e)[:50]}")

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

class PostLoader:
    """Load and prepare posts"""
    
    @staticmethod
    def load_posts(filename: str) -> Tuple[list, str]:
        """Load posts from JSONL"""
        posts = []
        
        filepath = os.path.join(SystemConfig.INPUT_FOLDER, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return [], ""
        
        topic = filename.replace('_posts.jsonl', '')
        
        logger.info(f"Loading: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    post = json.loads(line)
                    posts.append(post)
                except Exception as e:
                    logger.warning(f"Line {line_no} error: {e}")
        
        # Sort by time
        posts.sort(key=lambda p: p.get('posting_time', ''), reverse=False)
        logger.info(f"Loaded {len(posts)} posts")
        
        return posts, topic

# ═══════════════════════════════════════════════════════════════════
# PARALLEL PROCESSING ENGINE
# ═══════════════════════════════════════════════════════════════════

def process_batch(batch_data: Tuple[list, set]) -> list:
    """Process a batch of posts in parallel"""
    posts, skip_ids = batch_data
    results = []
    
    for post in posts:
        post_id = post.get('post_id', 'unknown')
        
        # Skip if already processed
        if post_id in skip_ids:
            continue
        
        # Classify
        score, explanation = ContentClassifier.evaluate(post)
        
        # Add metadata
        post['ai_confidence'] = score
        post['ai_reasoning'] = explanation
        
        # Categorize
        if score >= SystemConfig.THRESHOLD_HIGH:
            category = 'yes'
        elif score <= SystemConfig.THRESHOLD_LOW:
            category = 'no'
        else:
            category = 'uncertain'
        
        results.append((post, category))
        
        logger.debug(f"[{post_id}] {category.upper()} {score:.2f}")
    
    return results

# ═══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("[*] AI Content Organizer - Phase 2 (MULTI-THREADED)")
    print("="*70)
    
    logger.info("="*70)
    logger.info("AI Content Organizer (MT) - START")
    logger.info("="*70)
    
    # Health check
    print(f"[*] API Mode: {SystemConfig.API_MODE.upper()}")
    print(f"[*] Workers: {SystemConfig.WORKER_COUNT}")
    print(f"[*] Vision: {'Enabled' if SystemConfig.ENABLE_VISION else 'Disabled'}")
    
    if SystemConfig.API_MODE == 'ollama':
        print("[*] Checking Ollama...")
        if not HealthMonitor.verify_ollama():
            logger.error("Ollama not responding!")
            print("[!] ERROR: Ollama not responding!")
            print("    Start: ollama serve")
            print("    Install: ollama pull llava:13b")
            return
        print("[✓] Ollama OK")
    
    # Setup directories
    os.makedirs(SystemConfig.OUTPUT_FOLDER, exist_ok=True)
    
    # Find files
    jsonl_files = []
    if os.path.exists(SystemConfig.INPUT_FOLDER):
        for fname in os.listdir(SystemConfig.INPUT_FOLDER):
            if fname.endswith('_posts.jsonl'):
                jsonl_files.append(fname)
    
    if not jsonl_files:
        logger.error(f"No files in {SystemConfig.INPUT_FOLDER}")
        print(f"[!] No _posts.jsonl files found")
        return
    
    logger.info(f"Found {len(jsonl_files)} files")
    print(f"[*] Found {len(jsonl_files)} file(s):")
    for fname in sorted(jsonl_files):
        print(f"    - {fname}")
    
    # GPU status
    if SystemConfig.API_MODE == 'ollama':
        print(f"\n[*] GPU Status:")
        gpu_info = HealthMonitor.query_gpu()
        if gpu_info['mem_total'] > 0:
            print(f"    VRAM: {gpu_info['mem_used']:.0f}/{gpu_info['mem_total']:.0f}MB")
            print(f"    Util: {gpu_info['utilization']:.1f}%")
            print(f"    GPU Layers: {SystemConfig.GPU_LAYERS}")
        else:
            print(f"    ⚠ GPU info unavailable")
    
    # Process files
    for jsonl_file in sorted(jsonl_files):
        print(f"\n{'='*70}")
        print(f"[*] Processing: {jsonl_file}")
        print(f"{'='*70}")
        
        logger.info(f"Processing: {jsonl_file}")
        
        posts, topic = PostLoader.load_posts(jsonl_file)
        if not posts:
            continue
        
        # Check state
        state = StateManager.restore_state(topic)
        completed_ids = StateManager.get_completed_ids(topic)
        
        start_pos = 0
        counters = {'yes': 0, 'no': 0, 'uncertain': 0}
        
        if state and completed_ids:
            start_pos = state.get('current_position', 0)
            counters = state.get('counters', counters)
            logger.info(f"Resuming from: {start_pos}/{len(posts)}")
            print(f"[*] Resuming from: {start_pos}/{len(posts)}")
        else:
            print(f"[*] Starting fresh")
        
        # Output files
        out_yes = os.path.join(SystemConfig.OUTPUT_FOLDER,
                              f"{TextHelper.clean_for_path(topic)}_posts_likely_yes.jsonl")
        out_no = os.path.join(SystemConfig.OUTPUT_FOLDER,
                             f"{TextHelper.clean_for_path(topic)}_posts_likely_no.jsonl")
        out_uncertain = os.path.join(SystemConfig.OUTPUT_FOLDER,
                                    f"{TextHelper.clean_for_path(topic)}_posts_uncertain.jsonl")
        
        # Open files
        mode = 'a' if completed_ids else 'w'
        fh_yes = open(out_yes, mode, encoding='utf-8')
        fh_no = open(out_no, mode, encoding='utf-8')
        fh_uncertain = open(out_uncertain, mode, encoding='utf-8')
        
        try:
            print(f"[*] Parallel processing with {SystemConfig.WORKER_COUNT} workers...")
            
            # Split into batches
            remaining_posts = posts[start_pos:]
            batches = []
            
            for i in range(0, len(remaining_posts), SystemConfig.BATCH_SIZE_PER_WORKER):
                batch = remaining_posts[i:i + SystemConfig.BATCH_SIZE_PER_WORKER]
                batches.append((batch, completed_ids))
            
            # Process in parallel
            processed_count = start_pos
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=SystemConfig.WORKER_COUNT) as executor:
                future_to_batch = {executor.submit(process_batch, batch): batch 
                                  for batch in batches}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        
                        # Write results
                        for post, category in batch_results:
                            if category == 'yes':
                                fh_yes.write(json.dumps(post, ensure_ascii=False) + '\n')
                                fh_yes.flush()
                            elif category == 'no':
                                fh_no.write(json.dumps(post, ensure_ascii=False) + '\n')
                                fh_no.flush()
                            else:
                                fh_uncertain.write(json.dumps(post, ensure_ascii=False) + '\n')
                                fh_uncertain.flush()
                            
                            counters[category] += 1
                            processed_count += 1
                            
                            # Progress
                            print(f"    [{processed_count}/{len(posts)}] "
                                  f"{category.upper():9s} "
                                  f"{post.get('ai_confidence', 0):.2f} - "
                                  f"@{post.get('username', 'unknown')[:15]:15s}")
                        
                        # Checkpoint
                        if processed_count % SystemConfig.CHECKPOINT_INTERVAL == 0:
                            StateManager.persist_state(topic, processed_count, 
                                                      len(posts), counters, jsonl_file)
                    
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
            
            # Final checkpoint
            StateManager.persist_state(topic, len(posts), len(posts), 
                                      counters, jsonl_file)
        
        finally:
            fh_yes.close()
            fh_no.close()
            fh_uncertain.close()
        
        # Summary
        print(f"\n{'='*70}")
        print(f"[✓] {topic} Complete:")
        print(f"    Relevant: {counters['yes']} ({counters['yes']*100//len(posts)}%)")
        print(f"    Irrelevant: {counters['no']} ({counters['no']*100//len(posts)}%)")
        print(f"    Uncertain: {counters['uncertain']} ({counters['uncertain']*100//len(posts)}%)")
        print(f"    Total: {len(posts)}")
        logger.info(f"{topic} done - YES:{counters['yes']} NO:{counters['no']} UNC:{counters['uncertain']}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("[✓✓✓] PROCESSING COMPLETE")
    print("="*70)
    print(f"Results: {SystemConfig.OUTPUT_FOLDER}/")
    print(f"Checkpoints: {SystemConfig.STATE_FOLDER}/")
    print(f"\nLog: {logfile_path}")
    
    logger.info("="*70)
    logger.info("AI Content Organizer (MT) - COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    main()
