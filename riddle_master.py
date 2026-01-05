"""
RIDDLE MASTER
=============
Handles riddles for Locked Doors ("L").
Now features an AI Judge for lenient, rubric-based adjudication.
"""

import random
import json
import time
import os
import urllib.request
import urllib.error
from typing import Dict, Tuple, Optional, List

class RiddleMaster:
    """Adjudicates riddles for the dungeon using an AI Judge."""
    
    def __init__(self):
        # Configuration
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
        
        self.riddles = [
            {
                "id": 0,
                "question": "Look at me its you you see, turn me over you're upside down.",
                "answers": ["spoon", "reflection in a spoon", "silverware", "cutlery"]
            },
            {
                "id": 1,
                "question": "Billions of ancient wheels turning in my brain, dropping acid makes them spin, I can't think or do anything without them.",
                "answers": ["atpace", "atpase", "atp", "atp synthase", "adenosine triphosphate"]
            }
        ]
    
    def get_random_riddle(self) -> Dict:
        """Get a random riddle assignment."""
        return random.choice(self.riddles)
    
    def get_riddle_text(self, riddle_id: int) -> str:
        """Get text for a specific riddle ID."""
        for r in self.riddles:
            if r["id"] == riddle_id:
                return r["question"]
        return "A mysterious lock with no inscription."
        
    def get_riddle_answer(self, riddle_id: int) -> str:
        """Get the primary answer for a specific riddle ID (for Oracle view)."""
        for r in self.riddles:
            if r["id"] == riddle_id:
                return r["answers"][0] if r["answers"] else "???"
        return "???"
        
    def check_answer(self, riddle_id: int, answer: str) -> bool:
        """
        Check if the answer is correct.
        Attempts AI adjudication first. Falls back to strict matching on failure.
        """
        if not answer:
            return False
            
        riddle = next((r for r in self.riddles if r["id"] == riddle_id), None)
        if not riddle:
            return False
            
        # 1. Try AI Adjudication
        try:
            is_correct = self._call_ai_judge(riddle["question"], riddle["answers"], answer)
            if is_correct is not None:
                print(f"[RiddleMaster] AI Judge decided: '{answer}' is {is_correct}")
                return is_correct
        except Exception as e:
            print(f"[RiddleMaster] AI Judge failed: {e}. Falling back to strict match.")
            
        # 2. Fallback: Strict matching (case-insensitive)
        clean_answer = answer.lower().strip()
        for accepted in riddle["answers"]:
            if clean_answer == accepted.lower():
                return True
                
        return False

    def _call_ai_judge(self, question: str, correct_answers: List[str], user_answer: str) -> Optional[bool]:
        """
        Ask the LLM to judge the answer leniently.
        Returns True/False, or None if the call fails.
        """
        system_prompt = """You are the Riddle Master Judge. 
Your job is to determine if a traveler's answer to a riddle is correct based on the Accepted Answers.
BE LENIENT and EASY GOING.
- Accept synonyms (e.g., "spoon" == "soup spoon").
- Accept typos if the intent is clear.
- Accept descriptions of the answer (e.g., "it is a spoon").
- Ignore capitalization and punctuation.

Respond with JSON ONLY:
{
  "correct": boolean,
  "reason": "short explanation"
}"""

        user_prompt = f"""
Riddle: "{question}"
Accepted Answers: {correct_answers}
Traveler's Answer: "{user_answer}"

Is this correct? JSON response:"""

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1, # Low temp for consistent judging
                "num_ctx": 2048
            }
        }
        
        url = f"{self.base_url}/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            raw_response = result.get("response", "").strip()
            
            # Parse JSON from response
            try:
                # Try direct parse
                parsed = json.loads(raw_response)
                return parsed.get("correct", False)
            except:
                # Try finding JSON block
                import re
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        return parsed.get("correct", False)
                    except:
                        pass
                        
        return None