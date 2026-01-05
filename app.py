"""
DUNGEON EXPLORER FLASK APP v4.3
================================
Web interface for the dungeon explorer.
"""

import os
import json
import time
import threading
from flask import Flask, render_template, request, jsonify

from maze_gen import generate_dungeon
from game_state import GameStatus
from agents import GameController, OllamaConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)

# Global state
controller: GameController = None
config: OllamaConfig = None
auto_running = False
auto_thread = None
tick_interval = 3.0

# Oracle mana costs
ORACLE_POWER_COSTS = {
    "identify_box": 10,
    "restore_oil_small": 15,
    "restore_oil_large": 30,
    "restore_health_small": 20,
    "restore_health_large": 40,
    "restore_energy_small": 15,
    "restore_energy_large": 30,
    "cure_poison": 25,
}

MANA_PER_CHAR = 0.05
MANA_MIN_TEXT = 3
MANA_MAX_TEXT = 50


def calculate_text_mana_cost(text: str) -> int:
    """Calculate mana cost based on text length."""
    if not text or not text.strip():
        return 0
    text_len = len(text.strip())
    cost = max(MANA_MIN_TEXT, int(text_len * MANA_PER_CHAR))
    return min(cost, MANA_MAX_TEXT)


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_game(seed=None, size=21):
    """Initialize or reset the game."""
    global controller, config
    
    config = OllamaConfig(
        base_url=OLLAMA_URL,
        model=MODEL_NAME,
    )
    
    dungeon = generate_dungeon(size=size, seed=seed, maze_style=True)
    controller = GameController(dungeon, config)
    
    # Initial observation
    controller.game.explorer.explored.add(dungeon.start)
    controller.game.get_visible_cells()
    
    print(f"[App] Game initialized. Dungeon: {dungeon.width}x{dungeon.height}, seed={dungeon.seed}")
    print(f"[App] Start: {dungeon.start}, Exit: {dungeon.exit}")
    print(f"[App] Model: {MODEL_NAME}")
    
    return dungeon.seed


# Initialize on startup
init_game()


# ============================================================================
# AUTO-RUN LOOP
# ============================================================================

def auto_run_loop():
    """Background loop for automatic ticks."""
    global auto_running
    
    while auto_running:
        if controller.game.explorer.status == GameStatus.AWAITING_ORACLE:
            time.sleep(0.5)
            continue
        
        if controller.game.explorer.status in (GameStatus.VICTORY, GameStatus.DEAD):
            auto_running = False
            break
        
        try:
            controller.tick()
        except Exception as e:
            print(f"[Auto] Error: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(tick_interval)


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Get full game state."""
    state = controller.get_state()
    state["auto_running"] = auto_running
    state["oracle_power_costs"] = ORACLE_POWER_COSTS
    state["mana_per_char"] = MANA_PER_CHAR
    state["mana_min_text"] = MANA_MIN_TEXT
    state["mana_max_text"] = MANA_MAX_TEXT
    state["model"] = MODEL_NAME
    return jsonify(state)


@app.route("/api/control", methods=["POST"])
def api_control():
    """Control commands: resume, pause, step, reset."""
    global auto_running, auto_thread
    
    data = request.get_json(force=True) or {}
    cmd = data.get("cmd", "")
    
    if cmd == "resume":
        if not auto_running:
            auto_running = True
            controller.game.explorer.status = GameStatus.PLAYING
            auto_thread = threading.Thread(target=auto_run_loop, daemon=True)
            auto_thread.start()
        return jsonify({"ok": True, "status": "running"})
    
    elif cmd == "pause":
        auto_running = False
        controller.game.explorer.status = GameStatus.PAUSED
        return jsonify({"ok": True, "status": "paused"})
    
    elif cmd == "step":
        auto_running = False
        if controller.game.explorer.status == GameStatus.PAUSED:
            controller.game.explorer.status = GameStatus.PLAYING
        result = controller.tick()
        return jsonify({"ok": True, "result": result})
    
    elif cmd == "reset":
        auto_running = False
        seed = data.get("seed")
        size = data.get("size", 15)
        new_seed = init_game(seed=seed, size=size)
        return jsonify({"ok": True, "seed": new_seed})
    
    return jsonify({"ok": False, "error": f"Unknown command: {cmd}"})


@app.route("/api/oracle/respond", methods=["POST"])
def api_oracle_respond():
    """Handle Oracle response with proportional mana costs."""
    data = request.get_json(force=True) or {}
    
    recommendation = data.get("recommendation", "").strip()
    actions = data.get("actions", [])
    identity = data.get("identity", "Unknown")  # NEW: Extract Identity
    
    if not controller.game.explorer.pending_oracle:
        return jsonify({"ok": False, "error": "No pending oracle request"})
    
    supplies = []
    total_cost = 0
    
    # Calculate text advice cost
    text_cost = calculate_text_mana_cost(recommendation)
    if text_cost > 0:
        if controller.game.oracle_mana >= text_cost:
            controller.game.oracle_mana -= text_cost
            total_cost += text_cost
        else:
            return jsonify({"ok": False, "error": f"Not enough mana for text advice (need {text_cost})"})
    
    # Process oracle powers
    for action_id in actions:
        if action_id not in ORACLE_POWER_COSTS:
            continue
        
        cost = ORACLE_POWER_COSTS[action_id]
        if controller.game.oracle_mana >= cost:
            controller.game.oracle_mana -= cost
            total_cost += cost
            
            if action_id == "identify_box":
                pending = controller.game.explorer.pending_oracle
                box_pos = None
                
                if pending.about_box:
                    box_pos = pending.about_box
                else:
                    pos = controller.game.explorer.pos
                    cell = controller.game.get_current_cell()
                    if cell.terrain == "B" and cell.box_contents:
                        box_pos = pos
                    else:
                        from maze_gen import DIRECTIONS
                        for dir_name, (dx, dy) in DIRECTIONS.items():
                            adj_cell = controller.game.get_cell(pos[0] + dx, pos[1] + dy)
                            if adj_cell.terrain == "B" and adj_cell.box_contents:
                                box_pos = (pos[0] + dx, pos[1] + dy)
                                break
                
                if box_pos:
                    cell = controller.game.get_cell(box_pos[0], box_pos[1])
                    if cell.terrain == "B" and cell.box_contents:
                        supplies.append({
                            "type": "identify_box",
                            "position": list(box_pos),
                            "contents": cell.box_contents,
                        })
                    else:
                        controller.game.oracle_mana += cost
                        total_cost -= cost
                else:
                    controller.game.oracle_mana += cost
                    total_cost -= cost
                    
            elif action_id == "restore_oil_small":
                supplies.append({"type": "oil", "amount": 25})
            elif action_id == "restore_oil_large":
                supplies.append({"type": "oil", "amount": 60})
            elif action_id == "restore_health_small":
                supplies.append({"type": "health", "amount": 25})
            elif action_id == "restore_health_large":
                supplies.append({"type": "health", "amount": 50})
            elif action_id == "restore_energy_small":
                supplies.append({"type": "energy", "amount": 25})
            elif action_id == "restore_energy_large":
                supplies.append({"type": "energy", "amount": 50})
            elif action_id == "cure_poison":
                supplies.append({"type": "cure_poison"})
    
    # Provide response to controller with mana cost
    controller.provide_oracle_response(recommendation, supplies, mana_cost=total_cost, author=identity)
    
    # Mark oracle as having answered
    controller.oracle_answered = True
    
    return jsonify({
        "ok": True,
        "recommendation": recommendation,
        "supplies": supplies,
        "text_cost": text_cost,
        "power_cost": total_cost - text_cost,
        "total_mana_spent": total_cost,
        "mana_remaining": controller.game.oracle_mana,
    })


@app.route("/api/oracle/sacrifice", methods=["POST"])
def api_oracle_sacrifice():
    """Handle sacrifices for mana."""
    data = request.get_json(force=True) or {}
    resource = data.get("resource")
    amount = data.get("amount", 0)
    
    result = controller.game.execute_sacrifice(resource, amount)
    
    return jsonify(result)

@app.route("/api/oracle/opinion", methods=["POST"])
def api_oracle_opinion():
    """Force an opinion on the Oracles."""
    opinion = controller.force_opinion()
    return jsonify({"ok": True, "opinion": opinion})


@app.route("/api/oracle/calculate_cost", methods=["POST"])
def api_oracle_calculate_cost():
    """Calculate the cost of a text message (for live preview)."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    
    cost = calculate_text_mana_cost(text)
    return jsonify({
        "text_length": len(text.strip()) if text else 0,
        "cost": cost,
        "can_afford": controller.game.oracle_mana >= cost,
    })


@app.route("/api/debug")
def api_debug():
    """Debug endpoint for LLM prompts/responses."""
    return jsonify({
        "explorer_prompt": controller.explorer_agent.last_prompt,
        "explorer_response": controller.explorer_agent.last_response,
        "advisor_prompt": "",
        "advisor_response": {},
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dungeon Explorer v4.3")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--tick-interval", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=15, help="Dungeon size (square)")
    
    args = parser.parse_args()
    
    OLLAMA_URL = args.ollama_url
    MODEL_NAME = args.model
    tick_interval = args.tick_interval
    
    seed = init_game(seed=args.seed, size=args.size)
    
    print("=" * 60)
    print("🏰 DUNGEON EXPLORER v4.3 (Sacrifice & Identity)")
    print("=" * 60)
    print(f"\n🌐 Starting server at http://localhost:{args.port}")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)