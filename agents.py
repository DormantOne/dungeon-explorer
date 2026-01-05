#=====================================================================
# 🐍 FILE: agents.py (Updated: Oracle Theories & Learning)
#=====================================================================

"""
DUNGEON EXPLORER AGENTS v4.8 (Network & Economy Edition)
============================
Single-agent system with direct Oracle Scroll interface:
1. EXPLORER AGENT - Reads Oracle Scroll directly, decides actions, reports completion.
2. ORACLE GIAS SCROLL - Holds user advice until AI marks it done.

CHANGELOG:
- ADDED: Automatic Local IP detection.
- ADDED: Clickable 'OPEN DASHBOARD' links in emails.
- ADDED: Detailed 'Oracle Price List' in AI Prompt so Grimshaw knows costs.
- Integrated Microsoft Graph API for reliable email delivery.
- Added email triggers for ASK_ORACLE and STUCK events.
- UPDATED: Grimshaw chooses specific Oracle (Aespg vs Sohnenae).
- ADDED: 10-Minute Timeout for Oracle Requests.
- ADDED: Oracle Theory & Learning System (Grimshaw forms opinions on Oracles).
"""

import json
import re
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import urllib.request
import urllib.error
import requests # REQUIRED for Microsoft Graph API
import datetime
import socket # REQUIRED for Local IP detection

from game_state import DungeonGame, ExplorerState, OracleGiasScroll, OracleRequest, GameStatus
from maze_gen import Dungeon, DIRECTIONS
from knowledge_graph import KnowledgeGraph, Scratchpad, NodeType


# ============================================================================
# MICROSOFT GRAPH EMAIL CONFIGURATION
# ============================================================================

# Microsoft OAuth2 Credentials (Provided by User)
CLIENT_ID = 'xxxx'
CLIENT_SECRET = 'xxxx'
TENANT_ID = 'xxxx'
AUTHORITY = f'xxxx{TENANT_ID}'
SENDER_USER_ID = 'xxxx'

# Recipients
ORACLE_RECIPIENTS = ["xxxxx@gmail.com", "xxxxx@gmail.com"]

# Toggle to enable/disable
EMAIL_ENABLED = True 

# Costs for AI Reference (Must match app.py)
ORACLE_COSTS_REF = {
    "Identify Box": 10,
    "Restore Oil (Small)": 15,
    "Restore Energy (Small)": 15,
    "Restore Health (Small)": 20,
    "Cure Poison": 25,
    "Restore Oil (Large)": 30,
    "Restore Energy (Large)": 30,
    "Restore Health (Large)": 40
}

def get_local_ip():
    """Detect the local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect, just picks the interface used for routing
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def get_oauth2_token():
    """Fetch OAuth2 token from Microsoft."""
    token_url = f'{AUTHORITY}/oauth2/v2.0/token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'https://graph.microsoft.com/.default'
    }
    
    try:
        response = requests.post(token_url, data=token_data, timeout=5)
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"[Email] ❌ Failed to get Token: {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"[Email] ❌ Token Error: {e}")
        return None

def send_graph_email(subject: str, body_html: str, recipients: List[str] = None):
    """
    Send email via Microsoft Graph API with Dashboard Link.
    If 'recipients' is None, sends to all ORACLE_RECIPIENTS.
    """
    if not EMAIL_ENABLED:
        print(f"[Email] 🚫 Disabled. Would send: {subject}")
        return

    access_token = get_oauth2_token()
    if not access_token:
        return

    # Use specific recipients or default to broadcast list
    final_recipients = recipients if recipients else ORACLE_RECIPIENTS

    local_ip = get_local_ip()
    dashboard_url = f"http://{local_ip}:5055"
    
    # Append Dashboard Link
    full_body = f"""
    {body_html}
    <br><hr>
    <p style="text-align: center; font-size: 18px;">
        <a href="{dashboard_url}" style="background-color: #9b59b6; color: white; padding: 14px 25px; text-decoration: none; border-radius: 4px; display: inline-block;">
            🔮 OPEN ORACLE DASHBOARD
        </a>
    </p>
    <p style="text-align: center; font-size: 12px; color: #666;">
        (Must be on the same Wi-Fi network: {dashboard_url})
    </p>
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    email_data = {
        "message": {
            "subject": subject,
            "body": {"contentType": "HTML", "content": full_body},
            "toRecipients": [{"emailAddress": {"address": email}} for email in final_recipients]
        },
        "saveToSentItems": "true"
    }

    try:
        print(f"[Email] 📧 Sending '{subject}' to {final_recipients}...")
        response = requests.post(
            f"https://graph.microsoft.com/v1.0/users/{SENDER_USER_ID}/sendMail",
            headers=headers,
            json=email_data,
            timeout=10
        )

        if response.status_code == 202:
            print(f"[Email] ✅ Sent successfully!")
        else:
            print(f"[Email] ❌ Failed to Send: {response.status_code} {response.text}")
            
    except Exception as e:
        print(f"[Email] ❌ Send Error: {e}")


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"
    timeout: int = 120
    temperature: float = 0.3
    num_ctx: int = 16384


def call_ollama(prompt: str, system: str = "", config: OllamaConfig = None) -> str:
    """Call Ollama API and return raw response text."""
    config = config or OllamaConfig()
    url = f"{config.base_url}/api/generate"
    
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
        }
    }
    
    if system:
        payload["system"] = system
    
    data = json.dumps(payload).encode("utf-8")
    
    print(f"[Ollama] Calling {config.model}...")
    start = time.time()
    
    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            raw = result.get("response", "")
            
            elapsed = time.time() - start
            print(f"[Ollama] Response in {elapsed:.1f}s, {len(raw)} chars")
            
            if raw:
                preview = raw[:200].replace('\n', ' ')
                print(f"[Ollama] Preview: {preview}...")
            else:
                print("[Ollama] WARNING: Empty response!")
            
            return raw
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"[Ollama] ERROR after {elapsed:.1f}s: {e}")
        return ""


def parse_json_response(raw: str) -> Dict:
    """Extract JSON from LLM response."""
    if not raw:
        return {}
    
    try:
        return json.loads(raw.strip())
    except:
        pass
    
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    return {}


def extract_position_from_text(text: str) -> Optional[Tuple[int, int]]:
    """Extract a position like (5, 3) from text."""
    if not text:
        return None
    
    match = re.search(r'\((\d+)\s*,\s*(\d+)\)', text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    match = re.search(r'(?:at|position)\s*(\d+)\s*,\s*(\d+)', text, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    return None


def extract_oracle_context(question: str, current_pos: Tuple[int, int], 
                           game: 'DungeonGame') -> Dict:
    """Extract context from an oracle question."""
    context = {
        "about_box": None,
        "about_monster": None,
        "about_direction": None,
    }
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["box", "container", "crate", "food", "poison", "medicine"]):
        pos = extract_position_from_text(question)
        if pos:
            context["about_box"] = pos
        else:
            current_cell = game.get_current_cell()
            if current_cell.terrain == "B":
                context["about_box"] = current_pos
            else:
                for dir_name, (dx, dy) in DIRECTIONS.items():
                    adj_cell = game.get_cell(current_pos[0] + dx, current_pos[1] + dy)
                    if adj_cell.terrain == "B":
                        context["about_box"] = (current_pos[0] + dx, current_pos[1] + dy)
                        break
    
    if any(word in question_lower for word in ["monster", "creature", "enemy", "danger", "fight"]):
        pos = extract_position_from_text(question)
        if pos:
            context["about_monster"] = pos
        else:
            heard = game.can_hear_monsters()
            if heard:
                closest = min(heard, key=lambda m: m[3])
                context["about_monster"] = (closest[0], closest[1])
    
    for direction in ["north", "south", "east", "west", "n", "s", "e", "w"]:
        if direction in question_lower:
            dir_map = {"north": "N", "south": "S", "east": "E", "west": "W",
                      "n": "N", "s": "S", "e": "E", "w": "W"}
            context["about_direction"] = dir_map.get(direction, direction.upper())
            break
    
    if any(phrase in question_lower for phrase in ["which way", "where should", "how do i get", "path to"]):
        context["about_direction"] = "exit"
    
    return context


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXPLORER_SYSTEM = """You are GRIMSHAW, a dungeon explorer. You must find the EXIT to escape.

## THE ORACLES: AESPG & SOHNENAE
You are being watched by two powerful Oracles: **Aespg** and **Sohnenae**.
- You must build a THEORY about their personalities based on their responses.
- Are they generous? Do they like boxes? Are they slow (timeout)?
- You must CHOOSE which Oracle to ask. Do not spam the same one if they ignore you.
- **Mix it up!** If you are unsure, pick randomly to test them.

## RESOURCES & MANA
- Oracles need MANA to help you.
- SACRIFICE your stats (Health/Energy/Oil) to refill their Mana pool.

## TERRAIN & SURVIVAL
- 'B' = Box (UNKNOWN contents). Can be Food (+Energy), Poison (-Health), Medicine (+Health), or Oil.
- 'M' = MONSTER. Fighting costs Energy & Health.
- 'L' = Locked Door. Requires "SOLVE <answer>".

## ACTIONS
- "MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W"
- "OPEN" (Only if box identified!)
- "DISCARD" (If box is poison)
- "REST" (Recover energy)
- "ASK_ORACLE" (Ask specific Oracle for help)
- "SOLVE <answer>" (For riddles)
- "SACRIFICE_HEALTH" / "SACRIFICE_ENERGY" / "SACRIFICE_OIL"

## RESPONSE FORMAT (JSON)
{
  "inner_dialogue": "Thoughts on the Oracles and my theory",
  "thinking": "Analysis",
  "action": "ASK_ORACLE",
  "riddle_answer": null,
  "oracle_question": "Can you identify this box?",
  "target_oracle": "Aespg", // MUST BE "Aespg" or "Sohnenae"
  "oracle_theory_update": "Aespg seems to like boxes...", // Optional note on your theory
  "confidence": 0.8
}
"""


# ============================================================================
# EXPLORER AGENT
# ============================================================================

class ExplorerAgent:
    """The main decision-making agent."""
    
    def __init__(self, game: DungeonGame, ollama_config: OllamaConfig = None):
        self.game = game
        self.config = ollama_config or OllamaConfig()
        self.call_count = 0
        self.total_latency = 0
        
        self.last_prompt = ""
        self.last_response = {}
        
        # Initialize Starting Theories (Randomized per run)
        # This gives Grimshaw a "hunch" to start with
        theories = [
            ("The Box Master", "The Light Bringer"),
            ("The Generous", "The Stingy"),
            ("The Guide", "The Warrior"),
            ("The Silent", "The Chatty")
        ]
        chosen_pair = random.choice(theories)
        # Randomly assign roles
        if random.random() > 0.5:
            self.starting_theories = {"Aespg": chosen_pair[0], "Sohnenae": chosen_pair[1]}
        else:
            self.starting_theories = {"Aespg": chosen_pair[1], "Sohnenae": chosen_pair[0]}
            
    def _analyze_oracle_reputation(self) -> str:
        """Analyze memory to build a profile for each Oracle."""
        if not self.game.explorer.knowledge_graph:
            return "No data yet."
            
        memories = self.game.explorer.knowledge_graph.find_by_type(NodeType.ORACLE_ADVICE)
        
        stats = {
            "Aespg": {"asks": 0, "timeouts": 0, "replies": 0, "supplies": []},
            "Sohnenae": {"asks": 0, "timeouts": 0, "replies": 0, "supplies": []}
        }
        
        for m in memories:
            author = m.metadata.get("author", "Unknown")
            outcome = m.metadata.get("outcome", "")
            
            # Normalize author names
            target = None
            if "aespg" in author.lower(): target = "Aespg"
            elif "sohnen" in author.lower(): target = "Sohnenae"
            
            if target:
                stats[target]["asks"] += 1
                if outcome == "timeout":
                    stats[target]["timeouts"] += 1
                else:
                    stats[target]["replies"] += 1
                    
                # Analyze content for supply hints (crude text analysis)
                content_lower = m.content.lower()
                if "box" in content_lower: stats[target]["supplies"].append("boxes")
                if "oil" in content_lower or "light" in content_lower: stats[target]["supplies"].append("light")
                if "health" in content_lower or "heal" in content_lower: stats[target]["supplies"].append("healing")
        
        # Build Report
        report = "## ORACLE INTELLIGENCE REPORT (Grimshaw's Notebook)\n"
        
        for name in ["Aespg", "Sohnenae"]:
            s = stats[name]
            start_theory = self.starting_theories.get(name, "Unknown")
            
            reliability = "Unknown"
            if s["asks"] > 0:
                rel_pct = (s["replies"] / s["asks"]) * 100
                if rel_pct < 40: reliability = "UNRELIABLE (Ignored me often)"
                elif rel_pct < 80: reliability = "MODERATE"
                else: reliability = "HIGH (Reliable)"
            
            # Determine specialty from data, or fallback to theory
            specialty = start_theory + " (Theory)"
            if s["supplies"]:
                # Find most common
                from collections import Counter
                c = Counter(s["supplies"])
                most_common = c.most_common(1)[0][0]
                specialty = f"Proven source of {most_common.upper()}"
            
            report += f"- **{name.upper()}**: {specialty}. Reliability: {reliability}. (Asks: {s['asks']}, Timeouts: {s['timeouts']})\n"
            
        return report

    def build_prompt(self) -> str:
        """Build the prompt showing current state."""
        exp = self.game.explorer
        x, y = exp.pos
        
        # Status summary
        health_status = "CRITICAL" if exp.health < 25 else ("LOW" if exp.health < 50 else "OK")
        energy_status = "CRITICAL" if exp.energy < 20 else ("LOW" if exp.energy < 40 else "OK")
        oil_status = "EMPTY" if exp.oil <= 0 else ("CRITICAL" if exp.oil < 10 else ("LOW" if exp.oil < 20 else "OK"))
        
        # Calculate direction to exit
        exit_x, exit_y = self.game.dungeon.exit
        dx_to_exit = exit_x - x
        dy_to_exit = exit_y - y
        exit_hints = []
        if dx_to_exit > 0:
            exit_hints.append(f"EAST (+{dx_to_exit})")
        elif dx_to_exit < 0:
            exit_hints.append(f"WEST ({dx_to_exit})")
        if dy_to_exit > 0:
            exit_hints.append(f"SOUTH (+{dy_to_exit})")
        elif dy_to_exit < 0:
            exit_hints.append(f"NORTH ({dy_to_exit})")
        exit_direction = " and ".join(exit_hints) if exit_hints else "HERE!"
        
        # COST AWARENESS INJECTION
        cost_list = "\n".join([f"- {k}: {v} Mana" for k, v in ORACLE_COSTS_REF.items()])
        
        # ORACLE INTELLIGENCE REPORT
        oracle_report = self._analyze_oracle_reputation()
        
        status = f"""═══ YOUR STATUS ═══
Position: ({x}, {y})
Exit is at: ({exit_x}, {exit_y}) - go {exit_direction}

Health: {exp.health}/{exp.max_health} ({health_status})
Energy: {exp.energy}/{exp.max_energy} ({energy_status})
Lamp Oil: {exp.oil}/{exp.max_oil} ({oil_status})
Vision: {exp.vision_range()} cells
Turn: {exp.turn}

═══ ORACLE STATUS ═══
Current Mana Pool: {self.game.oracle_mana}

{oracle_report}

[COST MENU - What Oracles can do for you]
{cost_list}
"""
        
        # Warnings
        warnings = []
        if exp.health < 25:
            warnings.append("⚠️ HEALTH CRITICAL!")
        if exp.energy < 20:
            warnings.append("⚠️ ENERGY CRITICAL - find food!")
        if exp.oil <= 0:
            warnings.append("🔴 LAMP OUT - YOU ARE BLIND!")
        elif exp.oil < 10:
            warnings.append("⚠️ LAMP ALMOST OUT!")
        if exp.poisoned:
            warnings.append("☠️ POISONED - losing health!")
        if self.game.oracle_mana < 10:
             warnings.append("⚠️ ORACLE MANA LOW! Sacrifice stats so they can help you!")
        
        loop_warning = exp.detect_loop()
        if loop_warning:
            warnings.append(f"🔄 {loop_warning}")
            warnings.append("💡 TRY A DIFFERENT DIRECTION!")
        
        if warnings:
            status += "\n\n" + "\n".join(warnings)
        
        # Exploration stats
        floor_cells = sum(1 for (x, y), cell in self.game.dungeon.cells.items() 
                        if cell.terrain != "#")
        stats = exp.exploration_stats(floor_cells)
        
        exploration_info = f"""

═══ EXPLORATION PROGRESS ═══
Explored: {stats['explored']}/{stats['total']} cells ({stats['percentage']}%)"""
        
        # Current cell
        current_cell = self.game.get_current_cell()
        cell_info = f"\n\n═══ CURRENT CELL ({x}, {y}) ═══\nTerrain: '{current_cell.terrain}'"
        
        if current_cell.terrain == "B":
            if exp.pos in exp.known_boxes:
                contents = exp.known_boxes[exp.pos]
                cell_info += f"\n📦 BOX here - Oracle identified as: {contents.upper()}"
                if contents == "food":
                    cell_info += " - SAFE TO OPEN!"
                elif contents == "poison":
                    cell_info += " - DISCARD IT!"
                elif contents == "medicine":
                    cell_info += " - SAFE TO OPEN!"
                elif contents == "oil":
                    cell_info += " - SAFE TO OPEN!"
            else:
                cell_info += "\n📦 UNKNOWN BOX here! Ask Oracle before opening!"
        elif current_cell.terrain == "D":
            cell_info += "\n🌑 DARKNESS - extra oil consumption!"
        elif current_cell.terrain == "R":
            cell_info += "\n🪨 RUBBLE - movement costs extra energy"
        elif current_cell.terrain == "M":
            cell_info += f"\n👹 MONSTER HERE! Size: {current_cell.monster_size}"
        elif exp.pos == self.game.dungeon.exit:
            cell_info += "\n🌟 EXIT PORTAL! You can escape!"
        
        # Adjacent cells
        adjacent = self.game.get_adjacent_cells()
        adj_info = "\n\n═══ ADJACENT CELLS ═══"
        
        for direction, cell in adjacent.items():
            dx, dy = DIRECTIONS[direction]
            adj_x, adj_y = x + dx, y + dy
            symbol = cell.terrain
            desc = {
                "#": "Wall (blocked)",
                ".": "Floor (safe)",
                "B": "Box (unknown contents!)",
                "D": "Darkness",
                "R": "Rubble",
                "M": f"MONSTER ({cell.monster_size or 'unknown'})",
                "E": "EXIT!",
                "L": "LOCKED DOOR"
            }.get(symbol, symbol)

            # --- NEW: Dynamic Riddle Lookup ---
            if symbol == "L":
                riddle_text = self.game.riddle_master.get_riddle_text(cell.riddle_id)
                desc = f"LOCKED DOOR - RIDDLE: '{riddle_text}'"
            # ----------------------------------
            
            if (adj_x, adj_y) == self.game.dungeon.exit:
                desc = "EXIT! 🌟"
            
            if symbol == "B" and (adj_x, adj_y) in exp.known_boxes:
                contents = exp.known_boxes[(adj_x, adj_y)]
                if contents == "food":
                    desc = "Box - FOOD (safe to open!)"
                elif contents == "poison":
                    desc = "Box - POISON (discard it!)"
                elif contents == "medicine":
                    desc = "Box - MEDICINE (safe to open!)"
                elif contents == "oil":
                    desc = "Box - OIL (safe to open!)"
            
            visit_info = ""
            if symbol != "#" and symbol != "L":
                visits = exp.position_visits.get((adj_x, adj_y), 0)
                if visits == 0:
                    visit_info = " ✨NEW!"
                else:
                    visit_info = f" (visited {visits}x)"
            
            adj_info += f"\n  {direction} ({adj_x},{adj_y}): '{symbol}' - {desc}{visit_info}"
        
        # Known information
        known_info = "\n\n═══ ORACLE KNOWLEDGE ═══"
        if exp.known_boxes:
            known_info += "\n📦 IDENTIFIED BOXES:"
            for box_pos, contents in exp.known_boxes.items():
                action = "OPEN (safe!)" if contents in ("food", "medicine", "oil") else "DISCARD (danger!)"
                known_info += f"\n  • Box at ({box_pos[0]},{box_pos[1]}): {contents.upper()} → {action}"
        else:
            known_info += "\n  No boxes identified yet."
        
        # Local map - only show what's visible with line of sight
        vision = exp.vision_range()
        map_info = f"\n\n═══ YOUR VIEW (radius {vision}, line-of-sight) ═══\n"
        if vision == 0:
            map_info += "YOU ARE BLIND! Cannot see anything!\nYou can only feel: "
            # Describe what we can feel (adjacent cells by touch)
            current = self.game.get_current_cell()
            map_info += f"Standing on '{current.terrain}'. "
            for dir_name, (dx, dy) in DIRECTIONS.items():
                adj = self.game.get_cell(x + dx, y + dy)
                if adj.terrain == "#":
                    map_info += f"Wall to {dir_name}. "
                elif adj.terrain == "L":
                    map_info += f"Locked Door to {dir_name}. "
                else:
                    map_info += f"Open space to {dir_name}. "
        else:
            # Get visible cells using line of sight
            visible_cells = self.game.get_visible_cells()
            visible_set = {(vx, vy) for vx, vy, _ in visible_cells}
            
            header = "    "
            for dx in range(-vision, vision + 1):
                if abs(dx) <= vision:
                    header += f"{(x + dx) % 10}"
            map_info += header + "\n"
            
            for dy in range(-vision, vision + 1):
                row = f"{(y + dy):2d}: "
                for dx in range(-vision, vision + 1):
                    nx, ny = x + dx, y + dy
                    
                    if abs(dx) + abs(dy) > vision:
                        row += " "
                    elif dx == 0 and dy == 0:
                        row += "@"
                    elif (nx, ny) not in visible_set:
                        # Can't see this cell (blocked by walls)
                        row += "?"
                    else:
                        if (nx, ny) == self.game.dungeon.exit:
                            row += "E"
                        else:
                            cell = self.game.get_cell(nx, ny)
                            terrain = cell.terrain
                            
                            if terrain == "B" and (nx, ny) in exp.known_boxes:
                                contents = exp.known_boxes[(nx, ny)]
                                if contents == "food":
                                    terrain = "F"
                                elif contents == "poison":
                                    terrain = "P"
                                elif contents == "medicine":
                                    terrain = "+"
                                elif contents == "oil":
                                    terrain = "O"
                            elif terrain == "M":
                                terrain = "X"
                            
                            row += terrain
                map_info += row + "\n"
            
            map_info += "\n  Legend: @ = You, ? = Blocked by walls"
        
        # Knowledge Graph
        kg_info = "\n\n═══ 🧠 YOUR MEMORY ═══"
        if exp.knowledge_graph and exp.knowledge_graph.nodes:
            kg_info += "\n" + exp.knowledge_graph.render_for_ai(exp.turn, max_nodes=10)
        else:
            kg_info += "\nNo memories yet."
        
        # NEW: Oracle Gias Scroll Display
        scroll_info = ""
        if exp.active_scroll:
            history_list = "\n".join([f"  - {act}" for act in exp.active_scroll.actions_taken[-5:]])
            if not history_list:
                history_list = "  (Just received)"
            
            scroll_info = f"""
\n═══ 📜 THE ORACLE GIAS SCROLL ═══
Signed by: {exp.active_scroll.author}
The Oracle has handed you a scroll with direct instructions:
"{exp.active_scroll.advice_text}"

YOUR ACTIONS SO FAR UNDER THIS ADVICE:
{history_list}

INSTRUCTION:
1. Interpret the advice and decide your next immediate action.
2. Determine the status of this scroll:
   - "continue": You are still working on this advice.
   - "completed": You have finished what was asked.
   - "abandoned": The advice is impossible, dangerous, or fully rejected.
"""
        else:
            scroll_info = "\n\n═══ 📜 THE ORACLE GIAS SCROLL ═══\n(No active scroll. You are exploring on your own.)"

        instruction = """

═══ YOUR TURN ═══
Choose your action wisely.
Actions: MOVE_N, MOVE_E, MOVE_S, MOVE_W, OPEN, DISCARD, REST, ASK_ORACLE, SOLVE <answer>
SACRIFICE ACTIONS: SACRIFICE_HEALTH, SACRIFICE_ENERGY, SACRIFICE_OIL

JSON RESPONSE:
{
  "inner_dialogue": "How I feel",
  "thinking": "My analysis",
  "memory_updates": ["Facts to remember"],
  "scratchpad_goal": "Current objective",
  "action": "MOVE_E",
  "riddle_answer": "spoon",  // Include ONLY if action is SOLVE
  "oracle_question": null,
  "target_oracle": "Aespg", // "Aespg" or "Sohnenae" (Required for ASK_ORACLE)
  "scroll_status": "continue",
  "confidence": 0.8
}

Respond with JSON only:"""
        
        return status + exploration_info + cell_info + adj_info + known_info + map_info + kg_info + scroll_info + instruction
    
    def decide(self) -> Dict:
        """Make a decision about what to do next."""
        self.call_count += 1
        
        prompt = self.build_prompt()
        self.last_prompt = prompt
        
        start = time.time()
        raw = call_ollama(prompt, EXPLORER_SYSTEM, self.config)
        latency = int((time.time() - start) * 1000)
        self.total_latency += latency
        
        parsed = parse_json_response(raw)
        
        if not parsed:
            parsed = self._extract_from_text(raw)
        
        parsed["_raw"] = raw[:500]
        parsed["_latency_ms"] = latency
        
        self.last_response = parsed
        
        # Update explorer's inner state
        if "inner_dialogue" in parsed:
            self.game.explorer.inner_dialogue = parsed["inner_dialogue"]
        if "thinking" in parsed:
            self.game.explorer.thinking = parsed["thinking"]
        if "notes_to_self" in parsed and isinstance(parsed["notes_to_self"], list):
            for note in parsed["notes_to_self"]:
                self.game.explorer.add_note(str(note))
        
        # Process memory updates
        exp = self.game.explorer
        if "memory_updates" in parsed and isinstance(parsed["memory_updates"], list):
            for memory in parsed["memory_updates"]:
                if isinstance(memory, str) and memory.strip():
                    mem_lower = memory.lower()
                    if "box" in mem_lower or "food" in mem_lower or "poison" in mem_lower or "medicine" in mem_lower or "oil" in mem_lower:
                        node_type = NodeType.BOX
                    elif "monster" in mem_lower:
                        node_type = NodeType.MONSTER
                    elif "path" in mem_lower or "direction" in mem_lower or "go" in mem_lower:
                        node_type = NodeType.PATH
                    elif "danger" in mem_lower or "avoid" in mem_lower:
                        node_type = NodeType.DANGER
                    else:
                        node_type = NodeType.OBSERVATION
                    
                    exp.add_memory(node_type, memory, position=exp.pos)
        
        # Process scratchpad updates
        if exp.scratchpad:
            if "scratchpad_goal" in parsed and parsed["scratchpad_goal"]:
                exp.scratchpad.set_goal(str(parsed["scratchpad_goal"]))
            
            if "scratchpad_lesson" in parsed and parsed["scratchpad_lesson"]:
                exp.scratchpad.add_lesson(str(parsed["scratchpad_lesson"]))
            
            if "thinking" in parsed:
                exp.scratchpad.add_thought(str(parsed["thinking"])[:100])
        
        # Handle SOLVE action aggregation
        if parsed.get("action") == "SOLVE" and "riddle_answer" in parsed:
            parsed["action"] = f"SOLVE {parsed['riddle_answer']}"
        
        return parsed
    
    def _extract_from_text(self, raw: str) -> Dict:
        """Fallback: extract action from plain text."""
        result = {
            "inner_dialogue": "",
            "thinking": raw[:300] if raw else "No response from AI",
            "action": None,
            "oracle_question": None,
            "target_oracle": None,
            "scroll_status": "continue",
            "confidence": 0.3,
            "parse_error": True,
        }
        
        if not raw:
            return result
        
        raw_upper = raw.upper()
        
        for action in ["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "OPEN", "DISCARD", "REST", "ASK_ORACLE", "SOLVE", 
                      "SACRIFICE_HEALTH", "SACRIFICE_ENERGY", "SACRIFICE_OIL"]:
            if action in raw_upper:
                result["action"] = action
                break
        
        if "ASK_ORACLE" in raw_upper or "ORACLE" in raw_upper:
            questions = re.findall(r'[^.!?]*\?', raw)
            if questions:
                result["oracle_question"] = questions[-1].strip()
                result["action"] = "ASK_ORACLE"
            
            # Attempt to extract target if missing from JSON
            if "AESPG" in raw_upper:
                result["target_oracle"] = "Aespg"
            elif "SOHNEN" in raw_upper:
                result["target_oracle"] = "Sohnenae"
        
        return result


# ============================================================================
# GAME CONTROLLER
# ============================================================================

class GameController:
    """Main controller that orchestrates the game loop."""
    
    def __init__(self, dungeon: Dungeon, ollama_config: OllamaConfig = None):
        self.game = DungeonGame(dungeon)
        self.config = ollama_config or OllamaConfig()
        
        self.explorer_agent = ExplorerAgent(self.game, self.config)
        
        self.tick_messages: List[str] = []
        self.last_action_result: Dict = {}
        self.oracle_answered: bool = False
    
    def tick(self) -> Dict:
        """Process one game tick."""
        exp = self.game.explorer
        
        # Check game over
        if exp.status in (GameStatus.VICTORY, GameStatus.DEAD):
            # If scroll was active, mark abandoned/completed based on outcome
            if exp.active_scroll:
                final_status = "completed" if exp.status == GameStatus.VICTORY else "abandoned"
                self._handle_scroll_completion(final_status)
            return {"done": True, "status": exp.status.value}
        
        # Check if waiting for oracle
        if exp.pending_oracle and not exp.pending_oracle.response and not exp.pending_oracle.supplies:
             # Wait for response OR supplies. If responded_at is set, we process.
             if not getattr(exp.pending_oracle, 'responded_at', None):
                 
                 # NEW: TIMEOUT CHECK (10 minutes = 600 seconds)
                 elapsed = time.time() - exp.pending_oracle.timestamp
                 if elapsed > 600:
                     target = getattr(exp.pending_oracle, 'target_oracle', 'The Oracle')
                     timeout_msg = f"🚫 No response from {target}... You are on your own."
                     self.tick_messages.append(timeout_msg)
                     print(f"[Controller] Request to {target} timed out after {int(elapsed)}s.")
                     
                     # Record negative memory
                     if exp.knowledge_graph:
                         exp.add_memory(NodeType.ORACLE_ADVICE, 
                             f"Asked {target} for help but they did not respond.",
                             metadata={"author": target, "outcome": "timeout"})
                     
                     # Cancel request and resume play
                     exp.pending_oracle = None
                     exp.status = GameStatus.PLAYING
                     
                     return {
                         "action": "TIMEOUT",
                         "result_summary": timeout_msg,
                         "turn": exp.turn,
                         "messages": self.tick_messages[-10:],
                         "status": exp.status.value,
                         "done": False,
                         "last_action": self.last_action_result
                     }
                 
                 return {"waiting_oracle": True, "question": exp.pending_oracle.question}
        
        # If oracle just responded (Checked by responded_at to allow empty text with supplies)
        if exp.pending_oracle and getattr(exp.pending_oracle, 'responded_at', None):
            print("[Controller] Oracle responded. Creating Gias Scroll...")
            
            # 1. Apply explicitly granted supplies (Power Buttons)
            if exp.pending_oracle.supplies:
                messages = self.game.apply_oracle_supplies(exp.pending_oracle.supplies)
                self.tick_messages.extend(messages)
                
                # Update memory for box identifications from Powers
                for supply in exp.pending_oracle.supplies:
                    if supply.get("type") == "identify_box":
                        pos = supply.get("position", [])
                        contents = supply.get("contents", "unknown")
                        if exp.knowledge_graph and pos:
                            memory_text = f"Oracle identified box at ({pos[0]},{pos[1]}) as {contents.upper()}"
                            exp.add_memory(NodeType.BOX, memory_text, position=tuple(pos))
            
            # 2. Parse TEXT response for implicit box identification
            # This fixes the issue where user types "It's medicine" but doesn't use the Power Button
            response_text = exp.pending_oracle.response or ""
            response_lower = response_text.lower()
            box_context_pos = exp.pending_oracle.about_box
            
            # If the user is talking about a box, check if they identified it in text
            if box_context_pos:
                identified_content = None
                if "medicine" in response_lower:
                    identified_content = "medicine"
                elif "poison" in response_lower:
                    identified_content = "poison"
                elif "food" in response_lower or "eat" in response_lower:
                    identified_content = "food"
                elif "oil" in response_lower:
                    identified_content = "oil"
                
                # If identified, forcefully update known_boxes so OPEN action succeeds
                if identified_content:
                    cell = self.game.get_cell(*box_context_pos)
                    if cell.terrain == "B":
                        # We update known_boxes with the identified content
                        exp.known_boxes[box_context_pos] = identified_content
                        print(f"[Controller] Text Parsing: Identified box at {box_context_pos} as {identified_content}")
                        
                        # Add memory
                        exp.add_memory(NodeType.BOX, f"Oracle said box at {box_context_pos} is {identified_content.upper()}", position=box_context_pos)

            # 3. Create Oracle Gias Scroll directly (ONLY IF TEXT EXISTS)
            if response_text.strip():
                advice_id = exp.generate_advice_id()
                exp.active_scroll = OracleGiasScroll(
                    advice_text=response_text,
                    advice_id=advice_id,
                    turn_received=exp.turn,
                    start_health=exp.health,
                    start_energy=exp.energy,
                    start_oil=exp.oil,
                    author=exp.pending_oracle.author # Pass the author
                )
            
            # Track advice history
            exp.pending_oracle.start_health = exp.health
            exp.oracle_history.append(exp.pending_oracle)
            exp.pending_oracle = None
            self.oracle_answered = False
        
        # Increment turn
        exp.turn += 1
        
        # Get explorer decision
        decision = self.explorer_agent.decide()
        action = decision.get("action", "REST")
        scroll_status = decision.get("scroll_status", "continue")
        
        if not action:
            action = "REST"
        
        # Handle ASK_ORACLE
        if action == "ASK_ORACLE":
            question = decision.get("oracle_question", "I need help!")
            target_oracle = decision.get("target_oracle")
            
            # Prevent rapid-fire asking without movement/action
            # Allow asking if we just got a response but need more info, 
            # BUT if we are in a loop of asking, block it.
            if exp.oracle_history and (time.time() - exp.oracle_history[-1].timestamp) < 5:
                print("[Controller] Asked Oracle too rapidly, defaulting to REST")
                action = "REST"
            else:
                context_info = extract_oracle_context(question, exp.pos, self.game)
                
                exp.pending_oracle = OracleRequest(
                    question=question,
                    context=str(exp.pos),
                    turn=exp.turn,
                    position=exp.pos,
                    about_box=context_info.get("about_box"),
                    about_monster=context_info.get("about_monster"),
                    about_direction=context_info.get("about_direction"),
                )
                
                # Attach target for timeout tracking
                exp.pending_oracle.target_oracle = target_oracle
                
                exp.status = GameStatus.AWAITING_ORACLE
                
                # *** RESOLVE EMAIL RECIPIENT ***
                recipient_email = None
                
                if target_oracle and isinstance(target_oracle, str):
                    clean_target = target_oracle.lower().strip()
                    if "aespg" in clean_target:
                        recipient_email = "aespgh@gmail.com"
                    elif "sohnen" in clean_target:
                        recipient_email = "sohnenae@gmail.com"
                
                # If AI failed to choose or chose invalid, we must guess/randomize to ensure email is sent
                if not recipient_email:
                    recipient_email = random.choice(ORACLE_RECIPIENTS)
                    print(f"[Controller] Oracle target unclear ('{target_oracle}'). Guessed: {recipient_email}")
                    target_oracle = f"Unknown (Sent to {recipient_email})"
                    exp.pending_oracle.target_oracle = target_oracle
                else:
                    print(f"[Controller] Oracle target selected: {target_oracle} -> {recipient_email}")
                
                # *** EMAIL TRIGGER: ASK_ORACLE ***
                send_graph_email(
                    f"🔮 GRIMSHAW ASKS ({target_oracle}) - Turn {exp.turn}",
                    f"""<html><body>
                        <h2>Grimshaw Needs Guidance from {target_oracle}</h2>
                        <p><strong>Turn:</strong> {exp.turn}</p>
                        <p><strong>Position:</strong> {exp.pos}</p>
                        <p><strong>Question:</strong> "{question}"</p>
                        <p>Please respond via the Dashboard.</p>
                    </body></html>""",
                    recipients=[recipient_email]
                )
                
                return {
                    "action": "ASK_ORACLE",
                    "question": question,
                    "target_oracle": target_oracle,
                    "waiting_oracle": True,
                    "turn": exp.turn,
                    "context": context_info,
                }
        
        # Store position before action
        old_pos = exp.pos
        
        # Loop-breaking logic
        loop_warning = exp.detect_loop()
        action_source = "explorer"
        
        # *** EMAIL TRIGGER: DISTRESS SIGNAL (STUCK) ***
        if loop_warning and "STUCK" in loop_warning:
            send_graph_email(
                f"🆘 GRIMSHAW DISTRESS - Turn {exp.turn}",
                f"""<html><body>
                    <h2 style="color: red;">GRIMSHAW IS STUCK</h2>
                    <p><strong>Turn:</strong> {exp.turn}</p>
                    <p><strong>Position:</strong> {exp.pos}</p>
                    <p><strong>Reason:</strong> {loop_warning}</p>
                    <p>Immediate intervention required.</p>
                </body></html>"""
            )
        
        if loop_warning and action.startswith("MOVE_"):
            direction = action.replace("MOVE_", "")
            dx, dy = DIRECTIONS.get(direction, (0, 0))
            target_pos = (exp.pos[0] + dx, exp.pos[1] + dy)
            
            if exp.position_visits.get(target_pos, 0) >= 3:
                print(f"[Controller] LOOP OVERRIDE: {target_pos} visited {exp.position_visits.get(target_pos, 0)} times")
                
                adjacent = self.game.get_adjacent_cells()
                best_dir = None
                min_visits = 999
                
                for d, cell in adjacent.items():
                    if cell.terrain != "#" and cell.terrain != "L": # Avoid locked doors in loop breaking
                        d_dx, d_dy = DIRECTIONS[d]
                        d_pos = (exp.pos[0] + d_dx, exp.pos[1] + d_dy)
                        visits = exp.position_visits.get(d_pos, 0)
                        if visits < min_visits:
                            min_visits = visits
                            best_dir = d
                
                if best_dir and best_dir != direction:
                    new_action = f"MOVE_{best_dir}"
                    print(f"[Controller] Overriding {action} → {new_action}")
                    action = new_action
                    action_source = "loop_breaker"

        # *** HANDLE SACRIFICE ACTIONS ***
        if action == "SACRIFICE_HEALTH":
            result = self.game.execute_sacrifice("health", 10)
        elif action == "SACRIFICE_ENERGY":
            result = self.game.execute_sacrifice("energy", 20)
        elif action == "SACRIFICE_OIL":
            result = self.game.execute_sacrifice("oil", 10)
        else:
            # Execute normal action
            result = self.game.execute_action(action)
        
        # Update visible cells (line of sight exploration)
        self.game.get_visible_cells()
        
        # Build result message
        result_summary = f"{action}"
        if result.get("success"):
            if action.startswith("MOVE_"):
                direction = action.replace("MOVE_", "")
                result_summary = f"Moved {direction}: ({old_pos[0]},{old_pos[1]}) → ({exp.pos[0]},{exp.pos[1]})"
            elif action == "OPEN":
                result_summary = f"Opened box: {result.get('message', '')}"
            elif action == "DISCARD":
                result_summary = f"Discarded box"
            elif action == "REST":
                result_summary = f"Rested (no energy gain)"
            elif action.startswith("SOLVE"):
                result_summary = f"Solved Riddle: {result.get('message', '')}"
            elif action.startswith("SACRIFICE"):
                result_summary = f"Sacrificed: {result.get('message', '')}"
        else:
            result_summary = f"{action} FAILED: {result.get('message', 'Unknown error')}"
        
        # Update Active Scroll History
        if exp.active_scroll:
            exp.active_scroll.add_action(result_summary)
            if scroll_status in ["completed", "abandoned"]:
                self._handle_scroll_completion(scroll_status)
        
        # Record in history
        exp.action_history.append({
            "turn": exp.turn,
            "action": action,
            "source": action_source,
            "old_pos": list(old_pos),
            "new_pos": list(exp.pos),
            "result": result_summary,
            "success": result.get("success", False),
            "events": result.get("events", []),
        })
        
        # Auto-memory for significant events
        if exp.knowledge_graph:
            if action == "OPEN" and result.get("success"):
                memory = f"Turn {exp.turn}: Opened box at ({old_pos[0]},{old_pos[1]}): {result.get('message', 'success')}"
                exp.add_memory(NodeType.ACTION, memory, position=tuple(old_pos))
            elif action == "OPEN" and not result.get("success"):
                memory = f"Turn {exp.turn}: FAILED to open box: {result.get('message', 'unknown error')}"
                exp.add_memory(NodeType.ACTION, memory, position=tuple(old_pos))
            elif action == "DISCARD" and result.get("success"):
                memory = f"Turn {exp.turn}: Discarded box at ({old_pos[0]},{old_pos[1]})"
                exp.add_memory(NodeType.ACTION, memory, position=tuple(old_pos))
            elif action.startswith("SOLVE"):
                 exp.add_memory(NodeType.ACTION, f"Turn {exp.turn}: {result.get('message', 'Tried riddle')}", position=tuple(old_pos))

        
        # Process turn effects
        turn_messages = self.game.process_turn_effects()
        self.tick_messages.extend(result.get("events", []))
        self.tick_messages.extend(turn_messages)
        
        # Store last action result
        self.last_action_result = {
            "action": action,
            "source": action_source,
            "old_pos": list(old_pos),
            "new_pos": list(exp.pos),
            "success": result.get("success", False),
            "message": result_summary,
            "events": result.get("events", []) + turn_messages,
        }
        
        return {
            "action": action,
            "result": result,
            "result_summary": result_summary,
            "turn": exp.turn,
            "messages": self.tick_messages[-10:],
            "done": exp.status in (GameStatus.VICTORY, GameStatus.DEAD),
            "status": exp.status.value,
            "last_action": self.last_action_result,
        }
    
    def _handle_scroll_completion(self, status: str):
        """Handle the completion or abandonment of an Oracle Scroll."""
        exp = self.game.explorer
        if not exp.active_scroll:
            return

        scroll = exp.active_scroll
        
        # Assess outcomes
        health_change = exp.health - scroll.start_health
        energy_change = exp.energy - scroll.start_energy
        
        # Determine rating based on AI's decision
        rating = "neutral"
        if status == "completed":
            rating = "good" if health_change >= -10 else "neutral"
        elif status == "abandoned":
            rating = "bad"

        # Form memory - WITH AUTHOR
        memory_text = f"Oracle {scroll.author} Scroll (Turn {scroll.turn_received}): '{scroll.advice_text[:50]}...' → {status.upper()}"
        exp.add_memory(NodeType.ORACLE_ADVICE, memory_text, metadata={
            "advice_id": scroll.advice_id,
            "rating": rating,
            "outcome": status,
            "author": scroll.author
        })
        
        # Add to scratchpad history if needed
        if exp.scratchpad:
             exp.scratchpad.add_lesson(f"Oracle {scroll.author} advice '{scroll.advice_text[:20]}...' was {status}")

        print(f"[Controller] Scroll {status}: {scroll.advice_text[:30]}...")
        
        # Clear the scroll
        exp.active_scroll = None
    
    def provide_oracle_response(self, response: str, supplies: List[Dict] = None, mana_cost: int = 0, author: str = "Unknown"):
        """Provide the Oracle's response."""
        exp = self.game.explorer
        
        if not exp.pending_oracle:
            print("[Controller] No pending oracle request!")
            return
        
        exp.pending_oracle.response = response
        exp.pending_oracle.supplies = supplies or []
        exp.pending_oracle.responded_at = time.time()
        exp.pending_oracle.mana_cost = mana_cost
        exp.pending_oracle.author = author  # Set Author
        
        exp.status = GameStatus.PLAYING
        
        print(f"[Controller] Oracle {author} responded (cost: {mana_cost} mana): {response[:100]}...")
    
    def force_opinion(self) -> str:
        """Force Grimshaw to judge the Oracles based on memory."""
        if not self.game.explorer.knowledge_graph:
            return "I have no memories to form an opinion."

        # Fetch all Oracle interactions
        memories = self.game.explorer.knowledge_graph.find_by_type(NodeType.ORACLE_ADVICE)
        if not memories:
            return "I have not received enough advice from Aespg or Sohnenae to judge them."
            
        # Construct prompt for the LLM
        memory_text = "\n".join([f"- {m.content} (Author: {m.metadata.get('author', 'Unknown')}, Outcome: {m.metadata.get('outcome', 'Unknown')})" for m in memories])
        
        prompt = f"""
        Review your memories of the Oracles (Aespg and Sohnenae):
        {memory_text}
        
        Based ONLY on these memories, who has been more helpful? Why?
        Respond in 1-2 sentences as Grimshaw.
        """
        
        response = call_ollama(prompt, "You are Grimshaw.", self.config)
        return response

    def get_state(self) -> Dict:
        """Get full game state for API/UI."""
        state = self.game.get_state_dict()
        
        state["llm"] = {
            "explorer_calls": self.explorer_agent.call_count,
            "last_prompt": self.explorer_agent.last_prompt[-1500:],
            "last_response": self.explorer_agent.last_response,
        }
        
        state["messages"] = self.tick_messages[-20:]
        state["last_action"] = self.last_action_result
        state["oracle_answered"] = self.oracle_answered
        
        return state