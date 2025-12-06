#!/usr/bin/env python3
"""
Test script for Disney Content Classifier
Tests the custom Ollama model with various post examples
"""

import requests
import json
from typing import Tuple

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "disney-classifier"

def test_classification(text: str) -> Tuple[float, str]:
    """Send test post to classifier"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': MODEL_NAME,
                'prompt': f'Evaluate this post: "{text}"',
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            
            # Parse confidence
            import re
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', result)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            # Parse reason
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', result)
            reason = reason_match.group(1).strip() if reason_match else result[:50]
            
            return confidence, reason
        else:
            return 0.5, f"API error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return 0.0, "Ollama not running"
    except Exception as e:
        return 0.5, f"Error: {str(e)[:30]}"

def run_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("Disney Content Classifier - Test Suite")
    print("="*70)
    
    # Test cases: (post_text, expected_range, description)
    test_cases = [
        # High confidence cases (should be 0.8+)
        (
            "Just finished watching Zootopia for the 10th time! Judy Hopps is such an amazing character!",
            (0.8, 1.0),
            "Direct Zootopia character mention with appreciation"
        ),
        (
            "Made Pawpsicles today using the recipe from Zootopia! Kids loved them!",
            (0.8, 1.0),
            "Disney-themed recipe from the movie"
        ),
        (
            "My Judy Hopps cosplay is finally complete! Can't wait for the Disney convention! 🐰",
            (0.8, 1.0),
            "Zootopia cosplay creation"
        ),
        (
            "Nick Wilde's character development in Zootopia is one of the best in Disney animation",
            (0.8, 1.0),
            "Character analysis of Zootopia"
        ),
        (
            "Found this amazing Zootopia merchandise at the Disney Store! Flash sloth plushie!",
            (0.8, 1.0),
            "Zootopia merchandise discussion"
        ),
        
        # Medium confidence cases (should be 0.4-0.7)
        (
            "Anthropomorphic character design is so interesting in animation",
            (0.4, 0.7),
            "General animation topic (could relate)"
        ),
        (
            "This reminds me of a Disney movie with animals in a city",
            (0.4, 0.7),
            "Vague Disney reference"
        ),
        
        # Low confidence cases (should be 0.0-0.3)
        (
            "Meeting Judy for coffee at 3pm today",
            (0.0, 0.3),
            "Common name, not character"
        ),
        (
            "Nick told me a funny joke yesterday",
            (0.0, 0.3),
            "Common name, not character"
        ),
        (
            "Beautiful sunset at the beach today",
            (0.0, 0.3),
            "Generic content, no Disney connection"
        ),
        (
            "Check out this new cryptocurrency investment opportunity!",
            (0.0, 0.3),
            "Spam/off-topic"
        ),
        (
            "Political debate happening downtown",
            (0.0, 0.3),
            "News/politics (no Disney)"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for idx, (post_text, (min_conf, max_conf), description) in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {idx}/{len(test_cases)}: {description}")
        print(f"{'─'*70}")
        print(f"Post: \"{post_text[:60]}{'...' if len(post_text) > 60 else ''}\"")
        print(f"Expected: {min_conf:.1f} - {max_conf:.1f}")
        
        confidence, reason = test_classification(post_text)
        
        print(f"Result:   {confidence:.2f}")
        print(f"Reason:   {reason}")
        
        if min_conf <= confidence <= max_conf:
            print("✓ PASS", end="")
            passed += 1
        else:
            print("✗ FAIL", end="")
            failed += 1
        
        print(f" (Confidence: {confidence:.2f})")
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Results Summary")
    print("="*70)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed:      {passed} ({passed*100//len(test_cases)}%)")
    print(f"Failed:      {failed} ({failed*100//len(test_cases)}%)")
    
    if failed == 0:
        print("\n🎉 All tests passed! Model is working correctly.")
    elif passed >= len(test_cases) * 0.8:
        print("\n⚠ Most tests passed. Review failures above.")
    else:
        print("\n❌ Many tests failed. Check model configuration.")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    print(f"\nTesting model: {MODEL_NAME}")
    print(f"Endpoint: {OLLAMA_URL}")
    
    # Quick connection test
    try:
        response = requests.get(OLLAMA_URL.replace('/api/generate', '/api/tags'), timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
        else:
            print("✗ Ollama connection issue")
            exit(1)
    except:
        print("✗ Cannot connect to Ollama")
        print("  Start it with: ollama serve")
        exit(1)
    
    run_tests()
