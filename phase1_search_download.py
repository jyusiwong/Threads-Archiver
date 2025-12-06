"""
Personal Content Collector - Phase 1
=====================================
Collects posts from Threads about your hobbies and interests.
Perfect for Disney fans, Zootopia enthusiasts, recipe collectors, and more.

Each topic gets its own organized collection with media files.

Usage:
    python phase1_search_download.py

Output Structure:
    Interested_Event_Archive/
        ├── [Topic]_posts.jsonl
        └── [Topic]_media/

Configure your interests in TOPICS below.
Educational & personal use only. Respect platform terms.
"""

from playwright.sync_api import sync_playwright
from datetime import datetime
import requests
import json
import time
import re
import os

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION SECTION
# ═══════════════════════════════════════════════════════════════════

class Config:
    """Central configuration for the collector"""
    TOPICS = ["Judy Hopps", "Nick Wilde", "Zootopia", "Disney Recipes"]
    STORAGE_PATH = "Interested_Event_Archive"
    AUTH_CACHE = "thread_sessions"
    POST_LIMIT = 2000  # Per topic
    SCROLL_DELAY = 3
    LOAD_WAIT = 2
    TIMEOUT_MS = 60000
    ASK_DATE_FILTER = True
    NO_PROGRESS_LIMIT = 10
    BATCH_THRESHOLD = 5

# ═══════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════

class FileUtils:
    """File and path utilities"""
    
    @staticmethod
    def clean_name(text):
        """Make filename safe"""
        return re.sub(r'[\\/*?:"<>|]', "", text).replace(" ", "_").strip()
    
    @staticmethod
    def ensure_path(directory):
        """Create directory if missing"""
        os.makedirs(directory, exist_ok=True)
        return directory

class AuthManager:
    """Handle authentication and session persistence"""
    
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        FileUtils.ensure_path(cache_dir)
    
    def _get_user_identifier(self, cookie_list):
        """Extract user ID from cookies"""
        for item in cookie_list:
            if item.get('name') == 'ds_user_id':
                return item.get('value')
        return None
    
    def store(self, cookie_list, browser_agent, date_params=None):
        """Persist authentication session"""
        user_id = self._get_user_identifier(cookie_list)
        if not user_id:
            print("[!] Cannot identify user from cookies")
            return False
        
        cache_path = os.path.join(self.cache_dir, f"session_{user_id}.json")
        
        try:
            data = {
                "user_id": user_id,
                "cookies": cookie_list,
                "user_agent": browser_agent,
                "date_filter": date_params,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2)
            
            print(f"[✓] Auth cached to '{self.cache_dir}/'")
            if date_params:
                print(f"    Dates: before={date_params.get('before')} after={date_params.get('after')}")
            return True
        except Exception as err:
            print(f"[!] Cache error: {err}")
            return False
    
    def retrieve(self):
        """Load most recent authentication session"""
        if not os.path.exists(self.cache_dir):
            return None, None, None
        
        cache_files = [f for f in os.listdir(self.cache_dir) 
                      if f.startswith('session_') and f.endswith('.json')]
        
        if not cache_files:
            return None, None, None
        
        newest = max(cache_files, 
                    key=lambda f: os.path.getctime(os.path.join(self.cache_dir, f)))
        cache_path = os.path.join(self.cache_dir, newest)
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            print(f"[✓] Loaded cached session: {newest}")
            print(f"    From: {data.get('timestamp', 'Unknown')}")
            
            date_params = data.get('date_filter')
            if date_params:
                print(f"    Dates: before={date_params.get('before')} after={date_params.get('after')}")
            
            return data.get('cookies'), data.get('user_agent'), date_params
        except Exception as err:
            print(f"[!] Load error: {err}")
            return None, None, None

class StorageManager:
    """Manage file storage structure"""
    
    def __init__(self, base_path):
        self.base = base_path
        FileUtils.ensure_path(base_path)
    
    def get_media_path(self, topic):
        """Get media directory for topic"""
        clean_topic = FileUtils.clean_name(topic)
        media_path = os.path.join(self.base, f"{clean_topic}_media")
        FileUtils.ensure_path(media_path)
        return media_path
    
    def get_posts_file(self, topic):
        """Get JSONL file path for topic"""
        clean_topic = FileUtils.clean_name(topic)
        return os.path.join(self.base, f"{clean_topic}_posts.jsonl")

class MediaDownloader:
    """Handle media file downloads"""
    
    def __init__(self, cookie_list, browser_agent):
        self.cookies = cookie_list
        self.agent = browser_agent
    
    def fetch(self, url, post_identifier, target_dir, fallback_id):
        """Download media file"""
        if not url:
            return None
        
        try:
            # Detect media type
            is_video = bool(re.search(r'\.(mp4|mov|m3u8|gif)', url, re.I) or 
                          re.search(r'm84|f2/m84|video', url, re.I))
            extension = "mp4" if is_video else "jpg"
            
            # Generate unique filename
            url_signature = str(abs(hash(url)))[:10]
            filename = f"{post_identifier}_{url_signature}.{extension}"
            filepath = os.path.join(target_dir, filename)
            
            # Skip if exists
            if os.path.exists(filepath):
                return filename
            
            # Prepare authenticated session
            session = requests.Session()
            csrf = ''
            
            for cookie in self.cookies:
                session.cookies.set(cookie['name'], cookie['value'], 
                                  domain=cookie['domain'])
                if cookie['name'] == 'csrftoken':
                    csrf = cookie['value']
            
            headers = {
                "User-Agent": self.agent,
                "Referer": "https://www.threads.net/",
                "X-CSRFToken": csrf,
                "X-IG-App-ID": "238260118697666",
            }
            
            # Download
            with session.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(filepath, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            
            return filename
        
        except Exception as err:
            print(f"      [!] Download failed for {post_identifier}: {str(err)[:50]}")
            return None

# ═══════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════

class PostExtractor:
    """Extract post data from API responses"""
    
    @staticmethod
    def _find_post_node(data):
        """Recursively locate post data in JSON"""
        if isinstance(data, dict):
            # Check if this looks like a post
            if (data.get('id') and data.get('user', {}).get('username') and 
                (data.get('caption') or data.get('like_count') is not None)):
                return data
            
            # Search nested
            for value in data.values():
                result = PostExtractor._find_post_node(value)
                if result:
                    return result
        
        elif isinstance(data, list):
            for item in data:
                result = PostExtractor._find_post_node(item)
                if result:
                    return result
        
        return None
    
    @staticmethod
    def _extract_media_urls(node):
        """Extract all media URLs from post node"""
        urls = []
        
        def safe_url(versions):
            if versions and isinstance(versions, list) and len(versions) > 0:
                return versions[0].get('url')
            return None
        
        # Video versions
        video_data = node.get('video_versions')
        if video_data:
            url = safe_url(video_data)
            if url:
                urls.append(url)
        
        # Carousel media
        carousel = node.get('carousel_media')
        if carousel and isinstance(carousel, list):
            for item in carousel:
                # Check video first
                video_url = safe_url(item.get('video_versions'))
                if video_url:
                    urls.append(video_url)
                    continue
                
                # Then images
                image_url = safe_url(item.get('image_versions2', {}).get('candidates'))
                if image_url:
                    urls.append(image_url)
        
        # Single image fallback
        if not urls:
            image_data = node.get('image_versions2', {}).get('candidates')
            url = safe_url(image_data)
            if url:
                urls.append(url)
        
        return urls
    
    @classmethod
    def parse(cls, json_data):
        """Parse JSON response to extract posts"""
        try:
            post_node = cls._find_post_node(json_data)
            
            if isinstance(post_node, dict):
                nodes = [post_node]
            elif isinstance(post_node, list):
                nodes = post_node
            else:
                return []
            
            extracted = []
            timestamp = datetime.now().isoformat()
            
            for node in nodes:
                if not node:
                    continue
                
                # Extract identifiers
                long_identifier = node.get('id')
                author = node.get('user', {}).get('username')
                text_content = node.get('caption', {}).get('text')
                short_identifier = node.get('shortcode') or node.get('short_id')
                
                post_id = short_identifier if short_identifier else long_identifier
                
                # Timestamp
                unix_time = node.get('taken_at')
                post_time = (datetime.fromtimestamp(unix_time).isoformat() 
                           if unix_time else None)
                
                # Media URLs
                media_urls = cls._extract_media_urls(node)
                
                # Build post object
                post = {
                    "type": "main",
                    "username": author,
                    "post_id": post_id,
                    "long_id": long_identifier,
                    "post_text": text_content,
                    "posting_time": post_time,
                    "retrieved_at": timestamp,
                    "timestamp": unix_time,
                    "likes": node.get('like_count', 0),
                    "comments": node.get('comment_count', 0),
                    "reposts": node.get('reshare_count', 0),
                    "permalink": f"https://www.threads.net/@{author}/post/{post_id}",
                    "local_media": [],
                    "media_urls": media_urls,
                    "replies": [],
                    "total_nested_comments": 0
                }
                extracted.append(post)
            
            return extracted
        
        except Exception as err:
            print(f"[!!!] Extraction error: {err}")
            return []

# ═══════════════════════════════════════════════════════════════════
# DEDUPLICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════

class DuplicateFilter:
    """Track and filter duplicate posts"""
    
    def __init__(self):
        self.id_set = set()
        self.text_signatures = set()
    
    @staticmethod
    def _text_signature(text):
        """Create signature from post text"""
        if not text:
            return ""
        words = text.split()[:5]
        sig = " ".join(words)[:50]
        return sig.lower().strip()
    
    def is_duplicate(self, post):
        """Check if post is duplicate"""
        pid = post.get('post_id')
        lid = post.get('long_id')
        text_sig = self._text_signature(post.get('post_text', ''))
        
        # Check IDs
        if pid in self.id_set or lid in self.id_set:
            return True
        
        # Check text signature
        if text_sig and text_sig in self.text_signatures:
            return True
        
        return False
    
    def add(self, post):
        """Add post to tracking"""
        pid = post.get('post_id')
        lid = post.get('long_id')
        text_sig = self._text_signature(post.get('post_text', ''))
        
        if pid:
            self.id_set.add(pid)
        if lid:
            self.id_set.add(lid)
        if text_sig:
            self.text_signatures.add(text_sig)
    
    def load_from_file(self, filepath):
        """Load existing posts from JSONL"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    post = json.loads(line)
                    self.add(post)
                except:
                    pass

# ═══════════════════════════════════════════════════════════════════
# MAIN COLLECTION ENGINE
# ═══════════════════════════════════════════════════════════════════

class TopicCollector:
    """Collects posts for a single topic"""
    
    def __init__(self, context, topic, storage_mgr, downloader, date_params=None):
        self.context = context
        self.topic = topic
        self.storage = storage_mgr
        self.downloader = downloader
        self.date_params = date_params
        
        self.media_dir = storage_mgr.get_media_path(topic)
        self.output_file = storage_mgr.get_posts_file(topic)
        
        self.duplicate_filter = DuplicateFilter()
        self.duplicate_filter.load_from_file(self.output_file)
        
        self.pending_posts = []
        self.collected = []
        self.uncertain_ids = []
        self.count = 0
    
    def _build_search_url(self):
        """Construct search URL with filters"""
        url = "https://www.threads.com/search?"
        
        if self.date_params:
            if self.date_params.get('after'):
                url += f"after_date={self.date_params['after']}&"
            if self.date_params.get('before'):
                url += f"before_date={self.date_params['before']}&"
        
        url += f"q={self.topic}&serp_type=default"
        return url
    
    def _setup_api_interceptor(self):
        """Set up response handler for API calls"""
        def handle_response(response):
            if "graphql" not in response.url or response.request.method != "POST":
                return
            
            try:
                json_data = response.json()
                edges = json_data.get('data', {}).get('searchResults', {}).get('edges', [])
                
                if not edges:
                    return
                
                posts_data = []
                for edge in edges:
                    node = edge.get('node')
                    if not node:
                        continue
                    
                    thread = node.get('thread')
                    if not thread:
                        continue
                    
                    items = thread.get('thread_items')
                    if not (items and isinstance(items, list) and len(items) > 0):
                        continue
                    
                    post_data = items[0].get('post')
                    if post_data and post_data.get('pk'):
                        posts_data.append(post_data)
                
                if posts_data:
                    print(f"    [API] Found {len(posts_data)} posts")
                
                # Parse posts
                for post_dict in posts_data:
                    parsed = PostExtractor.parse({'post': post_dict})
                    for post in parsed:
                        if not self.duplicate_filter.is_duplicate(post):
                            self.pending_posts.append(post)
                            self.duplicate_filter.add(post)
            
            except Exception:
                pass
        
        return handle_response
    
    def _extract_short_ids_from_dom(self, page):
        """Extract short IDs from page DOM"""
        script = """
            Array.from(document.querySelectorAll('a[href*="/post/"]')).map(a => {
                const url = a.href;
                const parts = url.split('/').filter(p => p.length > 0);
                
                if (parts.length >= 4 && parts[parts.length - 2] === 'post') {
                    return {
                        username: parts[parts.length - 3].replace('@', ''),
                        short_id: parts[parts.length - 1]
                    };
                }
                return null;
            }).filter(item => item !== null);
        """
        
        try:
            return page.evaluate(script)
        except:
            return []
    
    def _process_pending_posts(self, page):
        """Process posts waiting in queue"""
        new_count = 0
        
        while self.pending_posts and self.count < Config.POST_LIMIT:
            post = self.pending_posts.pop(0)
            media_urls = post.pop('media_urls', [])
            
            # Try to get short ID from DOM
            try:
                dom_posts = self._extract_short_ids_from_dom(page)
                for dp in dom_posts:
                    if dp['username'] == post['username']:
                        post['post_id'] = dp['short_id']
                        break
            except:
                # Fallback to long ID
                if not post.get('post_id') and post.get('long_id'):
                    post['post_id'] = post['long_id']
                    post['short_id_uncertain'] = True
                    self.uncertain_ids.append({
                        "post_id": post['post_id'],
                        "username": post['username'],
                        "reason": "DOM extraction failed - using long_id"
                    })
            
            # Download media
            local_files = []
            for url in media_urls:
                filename = self.downloader.fetch(url, post['post_id'], 
                                                self.media_dir, post['long_id'])
                if filename:
                    local_files.append(filename)
            
            post['local_media'] = local_files
            post['permalink'] = f"https://www.threads.net/@{post['username']}/post/{post['post_id']}"
            
            # Save to file
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
            
            self.collected.append(post)
            self.count += 1
            new_count += 1
            
            print(f"    [{self.count}/{Config.POST_LIMIT}] "
                  f"{post['username'][:15]:15} | "
                  f"ID: {post['post_id'][:8]:8} | "
                  f"Media: {len(local_files)}")
        
        return new_count
    
    def collect(self):
        """Main collection loop"""
        print("\n" + "="*70)
        print(f"[>>>] Collecting '{self.topic}' (Target: {Config.POST_LIMIT} posts)")
        print("="*70)
        
        if len(self.duplicate_filter.id_set) > 0:
            print(f"    [i] Found {len(self.duplicate_filter.id_set)} existing IDs, "
                  f"{len(self.duplicate_filter.text_signatures)} text signatures")
        
        # Create page and set up interceptor
        page = self.context.new_page()
        page.on("response", self._setup_api_interceptor())
        
        # Navigate to search
        search_url = self._build_search_url()
        print(f"[-] Navigating: {search_url}")
        page.goto(search_url, timeout=Config.TIMEOUT_MS, wait_until='domcontentloaded')
        time.sleep(Config.LOAD_WAIT)
        
        # Initial scroll
        print("[-] Initial scroll...")
        page.mouse.wheel(0, 500)
        time.sleep(Config.SCROLL_DELAY)
        
        print(f"[*] Starting collection for: {self.topic}")
        print("[-] Scrolling and capturing...")
        
        # Scroll loop
        no_progress_count = 0
        loop_count = 0
        last_height = 0
        
        while self.count < Config.POST_LIMIT:
            loop_count += 1
            
            # Process pending
            new_posts = self._process_pending_posts(page)
            
            if self.count >= Config.POST_LIMIT:
                print("    [✓] Target reached")
                break
            
            # Flow control
            if len(self.pending_posts) >= Config.BATCH_THRESHOLD:
                print(f"    [i] Pausing - {len(self.pending_posts)} posts queued")
                time.sleep(Config.SCROLL_DELAY)
                continue
            
            # Check progress
            current_height = page.evaluate("document.body.scrollHeight")
            height_changed = current_height != last_height
            last_height = current_height
            
            if new_posts > 0 or height_changed:
                no_progress_count = 0
                if new_posts > 0:
                    print(f"    [i] Loop {loop_count}: {new_posts} new posts")
            else:
                no_progress_count += 1
                print(f"    [!] Loop {loop_count}: No progress ({no_progress_count}/{Config.NO_PROGRESS_LIMIT})")
            
            if no_progress_count >= Config.NO_PROGRESS_LIMIT:
                print(f"    [!!!] Stopping - no progress for {Config.NO_PROGRESS_LIMIT} loops")
                break
            
            # Scroll
            page.mouse.wheel(0, 5000)
            time.sleep(Config.SCROLL_DELAY)
            
            # Occasional bounce scroll
            if loop_count % 5 == 0:
                page.mouse.wheel(0, -500)
                time.sleep(0.5)
                page.mouse.wheel(0, 1000)
        
        page.close()
        print(f"\n[✓] '{self.topic}' Complete - Collected {self.count} posts")
        
        # Report uncertainties
        if self.uncertain_ids:
            print(f"\n[⚠️  WARNING] {len(self.uncertain_ids)} posts with uncertain IDs:")
            for entry in self.uncertain_ids[:5]:
                print(f"    - @{entry['username']}: {entry['post_id'][:20]}... ({entry['reason']})")
            if len(self.uncertain_ids) > 5:
                print(f"    ... and {len(self.uncertain_ids) - 5} more")
        
        return self.collected

# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main orchestration function"""
    print("[*] Initializing storage...")
    storage = StorageManager(Config.STORAGE_PATH)
    print(f"    [✓] Storage: {Config.STORAGE_PATH}")
    
    print("\n[*] Checking authentication...")
    auth_mgr = AuthManager(Config.AUTH_CACHE)
    cookies, user_agent, date_params = auth_mgr.retrieve()
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        
        # Restore or create session
        if cookies and user_agent:
            print("[*] Restoring cached session...")
            try:
                context.add_cookies(cookies)
                print("[✓] Session restored")
                
                # Verify session
                print("[*] Verifying session...")
                verify_page = context.new_page()
                
                api_error = False
                def check_api(response):
                    nonlocal api_error
                    if "graphql" in response.url and response.status in [401, 403]:
                        print(f"[!] API rejected (Status {response.status})")
                        api_error = True
                
                verify_page.on("response", check_api)
                verify_page.goto("https://www.threads.net/", wait_until='networkidle')
                
                if api_error:
                    print("[!] Session expired - need re-login")
                    cookies = None
                    date_params = None
                else:
                    print("[✓] Session valid")
                
                verify_page.close()
            except Exception as err:
                print(f"[!] Session error: {str(err)[:50]}")
                cookies = None
                user_agent = None
                date_params = None
        
        # New login if needed
        if not cookies or not user_agent:
            login_page = context.new_page()
            print("\n[*] Content Collector - Login Required")
            print("[-] Opening Threads login...")
            login_page.goto("https://www.threads.net/login", timeout=Config.TIMEOUT_MS)
            input("[!] Press Enter AFTER logging in...")
            
            print("[-] Capturing credentials...")
            cookies = context.cookies()
            user_agent = login_page.evaluate("navigator.userAgent")
            login_page.close()
            
            # Date filter setup
            print("\n[*] Date Filter Setup (Optional)")
            use_filter = input("[?] Enable date filtering? (y/n): ").lower().strip()
            date_params = None
            
            if use_filter == 'y':
                before = input("    [?] Before date (YYYY-MM-DD): ").strip()
                after = input("    [?] After date (YYYY-MM-DD, optional): ").strip()
                
                date_params = {
                    "before": before if before else None,
                    "after": after if after else None
                }
                
                # Validate format
                import re
                for date_val in [date_params["before"], date_params["after"]]:
                    if date_val and not re.match(r'^\d{4}-\d{2}-\d{2}$', date_val):
                        print(f"[!] Invalid format: {date_val}")
                        date_params = None
                        break
            
            print("[*] Caching session...")
            auth_mgr.store(cookies, user_agent, date_params)
        
        print("[✓] Authentication ready")
        
        # Handle date filter prompting
        if Config.ASK_DATE_FILTER:
            print("\n[*] Date Filter Configuration")
            use_filter = input("[?] Enable date filtering? (y/n): ").lower().strip()
            
            if use_filter == 'y':
                before = input("    [?] Before date (YYYY-MM-DD): ").strip()
                after = input("    [?] After date (YYYY-MM-DD, optional): ").strip()
                
                date_params = {
                    "before": before if before else None,
                    "after": after if after else None
                }
                
                # Validate
                import re
                for date_val in [date_params["before"], date_params["after"]]:
                    if date_val and not re.match(r'^\d{4}-\d{2}-\d{2}$', date_val):
                        print(f"[!] Invalid format: {date_val}")
                        date_params = None
                        break
            else:
                date_params = None
        elif not date_params:
            print("\n[*] Date Filter Configuration (Optional)")
            use_filter = input("[?] Enable date filtering? (y/n): ").lower().strip()
            
            if use_filter == 'y':
                before = input("    [?] Before date (YYYY-MM-DD): ").strip()
                after = input("    [?] After date (YYYY-MM-DD, optional): ").strip()
                date_params = {
                    "before": before if before else None,
                    "after": after if after else None
                }
        
        # Run collection for each topic
        print(f"\n[*] Starting collection for {len(Config.TOPICS)} topics...")
        
        downloader = MediaDownloader(cookies, user_agent)
        all_collected = []
        
        for idx, topic in enumerate(Config.TOPICS, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(Config.TOPICS)}] TOPIC: {topic}")
            print(f"{'='*70}")
            
            collector = TopicCollector(context, topic, storage, downloader, date_params)
            posts = collector.collect()
            all_collected.extend(posts)
        
        # Save date filter
        if date_params:
            auth_mgr.store(cookies, user_agent, date_params)
        
        # Summary
        print("\n" + "="*70)
        print("[✓✓✓] COLLECTION COMPLETE")
        print("="*70)
        
        for idx, topic in enumerate(Config.TOPICS, 1):
            posts_file = storage.get_posts_file(topic)
            if os.path.exists(posts_file):
                with open(posts_file, 'r', encoding='utf-8') as f:
                    post_count = sum(1 for line in f if json.loads(line).get('type') == 'main')
                print(f"  [{idx}] {topic}: {post_count} posts → {posts_file}")
        
        print(f"\nTotal collected: {len(all_collected)}")
        print(f"Storage: {Config.STORAGE_PATH}")
        print(f"Media: Per-topic directories")
        print("\n[NEXT] Run: python phase2_fetch_comments.py")
        
        browser.close()

if __name__ == "__main__":
    main()
