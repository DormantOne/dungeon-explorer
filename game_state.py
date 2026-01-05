#=====================================================================
# 🐍 FILE: game_state.py
#=====================================================================

"""
DUNGEON EXPLORER STATE v4.3 (Sacrifice & Identity Edition)
===========================
Explorer state, resources, game logic, Knowledge Graph memory, and monster movement.

CHANGELOG:
- Added 'author' field to OracleRequest and OracleGiasScroll to track (Aespg/Sohnenae).
- Added execute_sacrifice() to DungeonGame to allow trading stats for Mana.
- Replaced ActionPlan with OracleGiasScroll (Direct Oracle Conduit)
- Explorer now holds an active scroll containing raw user advice
- Added logic for tracking actions taken under specific advice
- Added RiddleMaster integration for Locked Doors ("L")
- Added Oracle View for Locked Doors/Riddles
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import time
import random

from maze_gen import Dungeon, DungeonCell, DIRECTIONS
from knowledge_graph import KnowledgeGraph, Scratchpad, NodeType
from riddle_master import RiddleMaster


class GameStatus(Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    AWAITING_ORACLE = "awaiting_oracle"
    VICTORY = "victory"
    DEAD = "dead"


@dataclass
class OracleRequest:
    """A pending request to the oracle with full context."""
    question: str
    context: str
    turn: int
    timestamp: float = field(default_factory=time.time)
    
    # Position when question was asked
    position: Tuple[int, int] = (0, 0)
    
    # What the question is about
    about_box: Optional[Tuple[int, int]] = None
    about_monster: Optional[Tuple[int, int]] = None
    about_direction: Optional[str] = None
    
    # Response
    response: Optional[str] = None
    author: str = "Unknown"  # NEW: Tracks if Aespg or Sohnenae answered
    supplies: List[Dict] = field(default_factory=list)
    responded_at: Optional[float] = None
    
    # Tracking for advice quality assessment
    advice_id: str = ""  # Unique ID for this advice session
    mana_cost: int = 0  # How much mana the oracle spent
    start_health: int = 0  # Health when advice was received
    start_energy: int = 0  # Energy when advice was received
    start_oil: int = 0  # Oil when advice was received
    outcome: str = ""  # "good", "bad", "neutral", or ""


@dataclass
class OracleGiasScroll:
    """
    The Oracle Gias Scroll: Direct guidance from the Oracle (User).
    Persists until the Explorer marks it as completed or abandoned.
    """
    advice_text: str
    advice_id: str
    turn_received: int
    start_health: int
    start_energy: int
    start_oil: int
    author: str = "Unknown"  # NEW: Tracks if Aespg or Sohnenae gave this scroll
    actions_taken: List[str] = field(default_factory=list)
    
    def add_action(self, action_summary: str):
        self.actions_taken.append(action_summary)


@dataclass
class ExplorerState:
    """The explorer's current state."""
    
    # Position
    pos: Tuple[int, int] = (1, 1)
    
    # Resources
    health: int = 100
    max_health: int = 100
    energy: int = 100
    max_energy: int = 100
    oil: int = 50
    max_oil: int = 50
    
    # Status effects
    poisoned: bool = False
    poison_damage: int = 2
    
    # Game state
    status: GameStatus = GameStatus.PAUSED
    turn: int = 0
    
    # Memory / Knowledge
    explored: Set[Tuple[int, int]] = field(default_factory=set)
    known_boxes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    known_monsters: Dict[Tuple[int, int], Dict] = field(default_factory=dict)
    
    # Position tracking for loop detection
    position_visits: Dict[Tuple[int, int], int] = field(default_factory=dict)
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    last_progress_turn: int = 0
    closest_to_exit: int = 999
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    # Oracle system
    pending_oracle: Optional[OracleRequest] = None
    oracle_history: List[OracleRequest] = field(default_factory=list)
    
    # NEW: The Oracle Gias Scroll (Direct Advice)
    active_scroll: Optional[OracleGiasScroll] = None
    
    # History
    action_history: List[Dict] = field(default_factory=list)
    
    # Inner state
    inner_dialogue: str = ""
    thinking: str = ""
    fear_level: float = 0.0
    mood: str = "cautious"
    
    # Knowledge Graph and Scratchpad
    knowledge_graph: Optional['KnowledgeGraph'] = None
    scratchpad: Optional['Scratchpad'] = None
    
    # Advice counter for unique IDs
    advice_counter: int = 0
    
    def __post_init__(self):
        if self.knowledge_graph is None:
            self.knowledge_graph = KnowledgeGraph(decay_rate=3.0, forget_threshold=10.0)
        if self.scratchpad is None:
            self.scratchpad = Scratchpad()
    
    def generate_advice_id(self) -> str:
        """Generate a unique advice ID."""
        self.advice_counter += 1
        return f"advice_{self.turn}_{self.advice_counter}"
    
    def add_memory(self, node_type: NodeType, content: str, position: Tuple[int, int] = None, metadata: Dict = None):
        """Add a memory to the knowledge graph."""
        if self.knowledge_graph:
            return self.knowledge_graph.add_node(
                node_type=node_type,
                content=content,
                position=position,
                current_turn=self.turn,
                metadata=metadata or {},
            )
        return None
    
    def reinforce_memory(self, search: str, amount: float = 15.0):
        """Reinforce memories matching the search term."""
        if self.knowledge_graph:
            return self.knowledge_graph.reinforce_by_content(search, amount, self.turn)
        return 0
    
    def record_position(self, pos: Tuple[int, int]):
        """Record visiting a position for loop detection."""
        self.position_visits[pos] = self.position_visits.get(pos, 0) + 1
        self.recent_positions.append(pos)
        if len(self.recent_positions) > 20:
            self.recent_positions = self.recent_positions[-20:]
    
    def detect_loop(self) -> Optional[str]:
        """Detect if the explorer is stuck in a loop."""
        if len(self.recent_positions) < 6:
            return None
        
        last_6 = self.recent_positions[-6:]
        if len(set(last_6)) <= 2:
            positions = list(set(last_6))
            return f"LOOP DETECTED! Oscillating between {positions[0]} and {positions[1] if len(positions) > 1 else positions[0]}"
        
        current = self.recent_positions[-1] if self.recent_positions else None
        if current and self.position_visits.get(current, 0) >= 4:
            return f"WARNING: You've visited {current} {self.position_visits[current]} times!"
        
        last_10 = self.recent_positions[-10:]
        unique_last_10 = set(last_10)
        if len(unique_last_10) <= 3 and len(last_10) >= 8:
            return f"STUCK! Only visiting {len(unique_last_10)} positions in last 10 moves: {unique_last_10}"
        
        return None
    
    def get_unvisited_adjacent(self, adjacent_cells: Dict[str, 'DungeonCell']) -> List[str]:
        """Get directions to adjacent cells that haven't been visited."""
        unvisited = []
        for direction, cell in adjacent_cells.items():
            if cell.terrain != "#":
                dx, dy = DIRECTIONS[direction]
                adj_pos = (self.pos[0] + dx, self.pos[1] + dy)
                visits = self.position_visits.get(adj_pos, 0)
                if visits == 0:
                    unvisited.append(f"{direction} (NEW!)")
                elif visits == 1:
                    unvisited.append(f"{direction} (visited once)")
        return unvisited
    
    def exploration_stats(self, total_floor_cells: int) -> Dict:
        """Get exploration statistics."""
        explored_count = len(self.explored)
        unique_visited = len(self.position_visits)
        return {
            "explored": explored_count,
            "total": total_floor_cells,
            "percentage": round(100 * explored_count / max(1, total_floor_cells)),
            "unique_positions": unique_visited,
            "total_moves": len(self.recent_positions),
        }
    
    def add_note(self, note: str):
        """Add a note that persists and feeds back to the AI."""
        if note and note not in self.notes:
            self.notes.append(note)
            if len(self.notes) > 15:
                self.notes = self.notes[-15:]
    
    def vision_range(self) -> int:
        """Calculate current vision range based on oil."""
        if self.oil <= 0:
            return 0
        elif self.oil < 10:
            return 1
        elif self.oil < 20:
            return 2
        elif self.oil < 35:
            return 3
        else:
            return 4
    
    def is_alive(self) -> bool:
        return self.health > 0 and self.energy > 0
    
    def calculate_fear(self, nearby_dangers: int = 0):
        """Update fear level based on current state."""
        fear = 0.0
        
        health_pct = self.health / self.max_health
        if health_pct < 0.25:
            fear += 0.35
        elif health_pct < 0.5:
            fear += 0.15
        
        energy_pct = self.energy / self.max_energy
        if energy_pct < 0.2:
            fear += 0.25
        elif energy_pct < 0.4:
            fear += 0.1
        
        if self.oil <= 0:
            fear += 0.4
        elif self.oil < 10:
            fear += 0.2
        elif self.oil < 20:
            fear += 0.1
        
        if self.poisoned:
            fear += 0.15
        
        fear += nearby_dangers * 0.1
        
        self.fear_level = min(1.0, fear)
        
        if self.fear_level < 0.15:
            self.mood = "calm"
        elif self.fear_level < 0.3:
            self.mood = "cautious"
        elif self.fear_level < 0.5:
            self.mood = "worried"
        elif self.fear_level < 0.75:
            self.mood = "scared"
        else:
            self.mood = "terrified"


class DungeonGame:
    """Main game logic controller."""
    
    def __init__(self, dungeon: Dungeon, starting_oil: int = 50, starting_energy: int = 100):
        self.dungeon = dungeon
        self.explorer = ExplorerState(
            pos=dungeon.start,
            oil=starting_oil,
            energy=starting_energy,
        )
        self.explorer.explored.add(dungeon.start)
        
        self.riddle_master = RiddleMaster()
        
        # Oracle mana - NOW starts at 50 with no upper bound
        self.oracle_mana = 50
        self.oracle_max_mana = 999  # Display max, but no actual cap
        
        # Action costs
        self.MOVE_COST = 1
        self.RUBBLE_COST = 5
        self.DARKNESS_OIL_COST = 2
        self.NORMAL_OIL_COST = 1
        
        # REST energy restoration
        self.REST_ENERGY_GAIN = 5
        
        # Monster fight costs
        self.MONSTER_COSTS = {
            "small": {"energy": 10, "damage": 5},
            "medium": {"energy": 20, "damage": 15},
            "large": {"energy": 35, "damage": 30},
        }
        
        # Box effects
        self.BOX_EFFECTS = {
            "food": {"energy": 30, "message": "Delicious! +30 energy"},
            "poison": {"health": -25, "poisoned": True, "message": "POISON! -25 health, now poisoned!"},
            "medicine": {"health": 25, "cure_poison": True, "message": "Medicine! +25 health, poison cured"},
            "oil": {"oil": 25, "message": "Flask of Oil! +25 oil"},
        }
    
    def get_cell(self, x: int, y: int) -> DungeonCell:
        return self.dungeon.get(x, y)
    
    def get_current_cell(self) -> DungeonCell:
        return self.get_cell(*self.explorer.pos)
    
    def get_adjacent_cells(self) -> Dict[str, DungeonCell]:
        x, y = self.explorer.pos
        result = {}
        for dir_name, (dx, dy) in DIRECTIONS.items():
            result[dir_name] = self.get_cell(x + dx, y + dy)
        return result
    
    def _find_nearby_box(self) -> Optional[Tuple[int, int]]:
        """
        Find a box at current position or adjacent cells.
        """
        x, y = self.explorer.pos
        
        # First check current cell
        current = self.get_current_cell()
        if current.terrain == "B":
            return (x, y)
        
        # Then check adjacent cells (cardinal directions only)
        for dir_name, (dx, dy) in DIRECTIONS.items():
            adj_pos = (x + dx, y + dy)
            adj_cell = self.get_cell(*adj_pos)
            if adj_cell.terrain == "B":
                return adj_pos
        
        return None
    
    def _clear_box_at(self, x: int, y: int) -> Optional[str]:
        """
        Comprehensively clear a box from ALL state representations.
        """
        # Get the actual cell from the dungeon's cells dict
        if (x, y) not in self.dungeon.cells:
            return None
        
        cell = self.dungeon.cells[(x, y)]
        
        # Verify there's actually a box here
        if cell.terrain != "B":
            return None
        
        # Capture contents before clearing
        contents = cell.box_contents
        
        # CLEAR 1: Set terrain to floor
        cell.terrain = "."
        
        # CLEAR 2: Clear box contents
        cell.box_contents = None
        
        # CLEAR 3: Remove from known_boxes if present
        if (x, y) in self.explorer.known_boxes:
            del self.explorer.known_boxes[(x, y)]
        
        return contents
    
    def _has_line_of_sight(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """
        Check if there's a clear line of sight from (x0,y0) to (x1,y1).
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # If we've reached the target, we have LOS
            if x == x1 and y == y1:
                return True
            
            # Check if current cell (not the start) blocks vision
            if (x != x0 or y != y0):
                cell = self.get_cell(x, y)
                # Walls block line of sight
                if cell.terrain == "#":
                    return False
                # Locked Doors also block line of sight (like walls)
                if cell.terrain == "L":
                    return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def get_visible_cells(self) -> List[Tuple[int, int, DungeonCell]]:
        """Get cells visible from current position using line-of-sight."""
        x, y = self.explorer.pos
        vision = self.explorer.vision_range()
        visible = []
        
        # Always see current position
        current_cell = self.get_cell(x, y)
        visible.append((x, y, current_cell))
        self.explorer.explored.add((x, y))
        
        if vision == 0:
            return visible  # Blind - can only feel current cell
        
        # Check each cell within vision range for line of sight
        for dy in range(-vision, vision + 1):
            for dx in range(-vision, vision + 1):
                if dx == 0 and dy == 0:
                    continue  # Already added current position
                    
                # Manhattan distance check
                if abs(dx) + abs(dy) > vision:
                    continue
                    
                nx, ny = x + dx, y + dy
                
                # Bounds check
                if not (0 <= nx < self.dungeon.width and 0 <= ny < self.dungeon.height):
                    continue
                
                # Line of sight check
                if self._has_line_of_sight(x, y, nx, ny):
                    cell = self.get_cell(nx, ny)
                    visible.append((nx, ny, cell))
                    self.explorer.explored.add((nx, ny))
        
        return visible
    
    def count_nearby_dangers(self) -> int:
        count = 0
        for cell in self.get_adjacent_cells().values():
            if cell.terrain == "M":
                count += 1
        return count
    
    def can_hear_monsters(self) -> List[Tuple[int, int, str, int]]:
        x, y = self.explorer.pos
        heard = []
        
        hearing_range = {"small": 2, "medium": 3, "large": 5}
        
        for mx, my in self.dungeon.monsters:
            cell = self.get_cell(mx, my)
            if cell.monster_size:
                distance = abs(mx - x) + abs(my - y)
                max_hear = hearing_range.get(cell.monster_size, 3)
                if distance <= max_hear:
                    heard.append((mx, my, cell.monster_size, distance))
        
        return heard
    
    def find_nearby_box(self) -> Optional[Tuple[int, int]]:
        x, y = self.explorer.pos
        
        current = self.get_current_cell()
        if current.terrain == "B":
            return (x, y)
        
        for dir_name, (dx, dy) in DIRECTIONS.items():
            cell = self.get_cell(x + dx, y + dy)
            if cell.terrain == "B":
                return (x + dx, y + dy)
        
        return None
    
    def process_turn_effects(self) -> List[str]:
        """Process per-turn effects."""
        messages = []
        
        # Oil consumption
        current_cell = self.get_current_cell()
        oil_cost = self.DARKNESS_OIL_COST if current_cell.terrain == "D" else self.NORMAL_OIL_COST
        self.explorer.oil = max(0, self.explorer.oil - oil_cost)
        
        if current_cell.terrain == "D":
            messages.append(f"🌑 Darkness consumes extra oil! (-{oil_cost} oil)")
        
        if self.explorer.oil <= 0:
            messages.append("🔴 YOUR LAMP IS OUT! You are BLIND!")
        elif self.explorer.oil < 10:
            messages.append("⚠️ Lamp oil critical!")
        
        # Poison damage
        if self.explorer.poisoned:
            damage = self.explorer.poison_damage
            self.explorer.health -= damage
            messages.append(f"☠️ Poison damage! (-{damage} health)")
        
        # Monster movement (every 2 turns)
        if self.explorer.turn % 2 == 0:
            monster_msgs = self._move_monsters()
            messages.extend(monster_msgs)
        
        # Knowledge Graph decay
        if self.explorer.knowledge_graph:
            forgotten = self.explorer.knowledge_graph.process_decay(self.explorer.turn)
            if forgotten:
                for node in forgotten:
                    messages.append(f"💭 Forgot: {node.content[:30]}...")
        
        # Check death conditions
        if self.explorer.health <= 0:
            self.explorer.status = GameStatus.DEAD
            messages.append("💀 You have died from your wounds!")
        elif self.explorer.energy <= 0:
            self.explorer.status = GameStatus.DEAD
            messages.append("💀 You collapsed from exhaustion!")
        
        # Update fear
        self.explorer.calculate_fear(self.count_nearby_dangers())
        
        return messages
    
    def _move_monsters(self) -> List[str]:
        """Move monsters around the dungeon."""
        messages = []
        
        occupied = {self.explorer.pos}
        new_monster_positions = []
        
        for mx, my in list(self.dungeon.monsters):
            cell = self.dungeon.get(mx, my)
            if not cell.monster_size:
                continue
            
            move_chance = {"small": 0.6, "medium": 0.4, "large": 0.2}.get(cell.monster_size, 0.3)
            
            if random.random() < move_chance:
                valid_moves = []
                for dx, dy in DIRECTIONS.values():
                    nx, ny = mx + dx, my + dy
                    adj_cell = self.dungeon.get(nx, ny)
                    if adj_cell.terrain in (".", "D", "R") and (nx, ny) not in occupied:
                        if (nx, ny) != self.explorer.pos:
                            valid_moves.append((nx, ny))
                
                if valid_moves:
                    new_x, new_y = random.choice(valid_moves)
                    
                    old_terrain = "." if cell.terrain == "M" else cell.terrain
                    cell.terrain = old_terrain
                    monster_size = cell.monster_size
                    monster_hp = cell.monster_hp
                    cell.monster_size = None
                    cell.monster_hp = 0
                    
                    new_cell = self.dungeon.get(new_x, new_y)
                    new_cell.terrain = "M"
                    new_cell.monster_size = monster_size
                    new_cell.monster_hp = monster_hp
                    
                    new_monster_positions.append((new_x, new_y))
                    occupied.add((new_x, new_y))
                    
                    px, py = self.explorer.pos
                    dist = abs(new_x - px) + abs(new_y - py)
                    if dist <= self.explorer.vision_range():
                        messages.append(f"👹 A {monster_size} monster moved to ({new_x},{new_y})!")
                else:
                    new_monster_positions.append((mx, my))
                    occupied.add((mx, my))
            else:
                new_monster_positions.append((mx, my))
                occupied.add((mx, my))
        
        self.dungeon.monsters = new_monster_positions
        
        return messages
    
    def execute_move(self, direction: str) -> Dict:
        """Execute a move in the given direction."""
        if direction not in DIRECTIONS:
            return {"success": False, "message": f"Invalid direction: {direction}"}
        
        dx, dy = DIRECTIONS[direction]
        x, y = self.explorer.pos
        nx, ny = x + dx, y + dy
        
        target_cell = self.get_cell(nx, ny)
        
        if not target_cell.is_passable():
            # If it's a Locked Door ("L"), give a specific hint
            if target_cell.terrain == "L":
                riddle = self.riddle_master.get_riddle_text(target_cell.riddle_id)
                return {
                    "success": False, 
                    "message": f"LOCKED DOOR! A voice whispers: '{riddle}'. Use SOLVE <answer>."
                }
            return {"success": False, "message": f"Blocked by wall to the {direction}"}
        
        events = []
        
        energy_cost = self.MOVE_COST
        if target_cell.terrain == "R":
            energy_cost = self.RUBBLE_COST
            events.append(f"Pushed through rubble (-{energy_cost} energy)")
        
        if self.explorer.energy < energy_cost:
            return {"success": False, "message": "Not enough energy to move!"}
        
        if target_cell.terrain == "M":
            fight_result = self._fight_monster(nx, ny, target_cell)
            events.extend(fight_result["events"])
            if not fight_result["success"]:
                return {"success": False, "message": fight_result["message"], "events": events}
        
        self.explorer.energy -= energy_cost
        old_pos = self.explorer.pos
        self.explorer.pos = (nx, ny)
        self.explorer.explored.add((nx, ny))
        
        self.explorer.record_position((nx, ny))
        
        dist_to_exit = self.dungeon.distance_to_exit((nx, ny))
        if dist_to_exit < self.explorer.closest_to_exit:
            self.explorer.closest_to_exit = dist_to_exit
            self.explorer.last_progress_turn = self.explorer.turn
            events.append(f"📍 Progress! Now {dist_to_exit} steps from exit (closest yet!)")
        
        if (nx, ny) == self.dungeon.exit:
            self.explorer.status = GameStatus.VICTORY
            events.append("🏆 You found the EXIT!")
        
        return {
            "success": True,
            "message": f"Moved {direction} to ({nx}, {ny})",
            "events": events,
            "new_pos": (nx, ny),
        }
    
    def _fight_monster(self, mx: int, my: int, cell: DungeonCell) -> Dict:
        size = cell.monster_size or "medium"
        costs = self.MONSTER_COSTS.get(size, self.MONSTER_COSTS["medium"])
        
        events = [f"⚔️ Fighting {size} monster!"]
        
        if self.explorer.energy < costs["energy"]:
            return {
                "success": False,
                "message": f"Not enough energy to fight {size} monster (need {costs['energy']})",
                "events": events,
            }
        
        self.explorer.energy -= costs["energy"]
        self.explorer.health -= costs["damage"]
        
        events.append(f"Lost {costs['energy']} energy and {costs['damage']} health")
        
        cell.terrain = "."
        cell.monster_size = None
        cell.monster_hp = 0
        
        if (mx, my) in self.dungeon.monsters:
            self.dungeon.monsters.remove((mx, my))
        
        events.append(f"Monster defeated!")
        
        return {"success": True, "message": "Monster defeated", "events": events}
    
    def execute_open_box(self) -> Dict:
        """Open a box at current position OR move to adjacent box and open it."""
        # Find the nearest box (current position first, then adjacent)
        box_pos = self._find_nearby_box()
        
        if box_pos is None:
            return {"success": False, "message": "No box here or adjacent to open"}
        
        events = []
        is_adjacent = box_pos != self.explorer.pos
        
        # If box is adjacent, move to it first
        if is_adjacent:
            # Check energy for movement
            if self.explorer.energy < self.MOVE_COST:
                return {"success": False, "message": "Not enough energy to reach the box!"}
            
            # Pay movement cost and move to box position
            self.explorer.energy -= self.MOVE_COST
            old_pos = self.explorer.pos
            self.explorer.pos = box_pos
            self.explorer.explored.add(box_pos)
            self.explorer.record_position(box_pos)
            
            events.append(f"Approached box at ({box_pos[0]},{box_pos[1]})")
            
            # Check progress toward exit
            dist_to_exit = self.dungeon.distance_to_exit(box_pos)
            if dist_to_exit < self.explorer.closest_to_exit:
                self.explorer.closest_to_exit = dist_to_exit
                self.explorer.last_progress_turn = self.explorer.turn
                events.append(f"📍 Progress! Now {dist_to_exit} steps from exit")
        
        # Now we're at the box
        x, y = self.explorer.pos
        
        # Check if we know what's in this box (Oracle must have identified it)
        if (x, y) in self.explorer.known_boxes:
            contents = self.explorer.known_boxes[(x, y)]
            effect = self.BOX_EFFECTS.get(contents, {})
            
            # CLEAR THE BOX FROM ALL STATE using the authoritative method
            cleared_contents = self._clear_box_at(x, y)
            
            if cleared_contents is None:
                events.append("⚠️ Box was already gone!")
                return {
                    "success": False,
                    "message": "Box disappeared unexpectedly",
                    "events": events,
                    "new_pos": (x, y),
                }
            
            events.append(f"Opened box containing {contents}")
            
            # Apply effects
            if "energy" in effect:
                self.explorer.energy = min(self.explorer.max_energy, 
                                          self.explorer.energy + effect["energy"])
            if "health" in effect:
                self.explorer.health = min(self.explorer.max_health,
                                          self.explorer.health + effect["health"])
            if "oil" in effect:
                self.explorer.oil = min(self.explorer.max_oil,
                                       self.explorer.oil + effect["oil"])
            if effect.get("poisoned"):
                self.explorer.poisoned = True
            if effect.get("cure_poison"):
                self.explorer.poisoned = False
            
            events.append(effect.get("message", ""))
            
            return {
                "success": True, 
                "message": effect.get("message", "Opened box"), 
                "events": events,
                "new_pos": (x, y),
            }
        else:
            # Unknown box - we moved there but won't open it
            events.append("⚠️ Unknown box - could be food, poison, medicine, or oil!")
            return {
                "success": False, 
                "message": "Box contents unknown! Ask Oracle to identify before opening.",
                "events": events,
                "new_pos": (x, y) if is_adjacent else None,
            }
    
    def execute_discard_box(self) -> Dict:
        """Discard a box at current position OR move to adjacent box and discard it."""
        # Find the nearest box (current position first, then adjacent)
        box_pos = self._find_nearby_box()
        
        if box_pos is None:
            return {"success": False, "message": "No box here or adjacent to discard"}
        
        events = []
        is_adjacent = box_pos != self.explorer.pos
        
        # If box is adjacent, move to it first
        if is_adjacent:
            # Check energy for movement
            if self.explorer.energy < self.MOVE_COST:
                return {"success": False, "message": "Not enough energy to reach the box!"}
            
            # Pay movement cost and move to box position
            self.explorer.energy -= self.MOVE_COST
            old_pos = self.explorer.pos
            self.explorer.pos = box_pos
            self.explorer.explored.add(box_pos)
            self.explorer.record_position(box_pos)
            
            events.append(f"Approached box at ({box_pos[0]},{box_pos[1]})")
            
            # Check progress toward exit
            dist_to_exit = self.dungeon.distance_to_exit(box_pos)
            if dist_to_exit < self.explorer.closest_to_exit:
                self.explorer.closest_to_exit = dist_to_exit
                self.explorer.last_progress_turn = self.explorer.turn
                events.append(f"📍 Progress! Now {dist_to_exit} steps from exit")
        
        # Now we're at the box - get its known contents for messaging (before clearing)
        x, y = self.explorer.pos
        known_contents = self.explorer.known_boxes.get((x, y), "unknown")
        
        # CLEAR THE BOX FROM ALL STATE using the authoritative method
        cleared_contents = self._clear_box_at(x, y)
        
        if cleared_contents is None:
            events.append("⚠️ Box was already gone!")
            return {
                "success": False,
                "message": "No box to discard",
                "events": events,
                "new_pos": (x, y) if is_adjacent else None,
            }
        
        # Use known contents for message (what we thought it was), or actual contents
        display_contents = known_contents if known_contents != "unknown" else cleared_contents
        events.append(f"Discarded the {display_contents} box")
        
        return {
            "success": True,
            "message": f"Discarded the {display_contents} box",
            "events": events,
            "new_pos": (x, y),
        }
    
    def execute_solve_riddle(self, answer: str) -> Dict:
        """Attempt to solve the riddle of a nearby locked door."""
        x, y = self.explorer.pos
        
        # Find adjacent locked door
        target_pos = None
        target_cell = None
        
        for dx, dy in DIRECTIONS.values():
            adj_x, adj_y = x + dx, y + dy
            cell = self.get_cell(adj_x, adj_y)
            if cell.terrain == "L":
                target_pos = (adj_x, adj_y)
                target_cell = cell
                break
        
        if not target_cell:
             return {"success": False, "message": "No locked door nearby to solve."}
        
        events = []
        is_correct = self.riddle_master.check_answer(target_cell.riddle_id, answer)
        
        if is_correct:
            # Unlock the door
            target_cell.terrain = "."
            target_cell.riddle_id = None
            events.append("✨ The door dissolves into mist! You solved it!")
            self.explorer.explored.add(target_pos)
            return {
                "success": True, 
                "message": "Riddle solved! Door unlocked.", 
                "events": events
            }
        else:
            # Penalize
            self.explorer.health -= 10
            events.append("🔥 WRONG! The door glows hot. You take 10 damage.")
            return {
                "success": False,
                "message": "Wrong answer! -10 Health.",
                "events": events
            }

    def execute_rest(self) -> Dict:
        """Rest to regain some energy. Still burns oil."""
        old_energy = self.explorer.energy
        self.explorer.energy = min(self.explorer.max_energy, 
                                   self.explorer.energy + self.REST_ENERGY_GAIN)
        energy_gained = self.explorer.energy - old_energy
        
        return {
            "success": True,
            "message": f"Rested briefly (+{energy_gained} energy, lamp still burns)",
            "events": [f"Caught breath and recovered {energy_gained} energy"],
        }
    
    def execute_sacrifice(self, resource: str, amount: int) -> Dict:
        """Sacrifice resources to grant Oracle Mana."""
        resource = resource.lower()
        amount = int(amount)
        if amount <= 0:
            return {"success": False, "message": "Must sacrifice positive amount"}

        mana_gain = 0
        events = []

        if resource == "health":
            if self.explorer.health <= amount:
                return {"success": False, "message": "Cannot sacrifice fatal health!"}
            self.explorer.health -= amount
            mana_gain = amount # 1:1 ratio
            events.append(f"🩸 Sacrificed {amount} Health for {mana_gain} Mana")
            
        elif resource == "energy":
            if self.explorer.energy < amount:
                return {"success": False, "message": "Not enough energy!"}
            self.explorer.energy -= amount
            mana_gain = max(1, amount // 2) # 2:1 ratio
            events.append(f"⚡ Sacrificed {amount} Energy for {mana_gain} Mana")
            
        elif resource == "oil":
            if self.explorer.oil < amount:
                return {"success": False, "message": "Not enough oil!"}
            self.explorer.oil -= amount
            mana_gain = amount # 1:1 ratio
            events.append(f"🔥 Sacrificed {amount} Oil for {mana_gain} Mana")
            
        else:
            return {"success": False, "message": "Invalid resource. Use: health, energy, oil"}

        self.oracle_mana += mana_gain
        return {"success": True, "message": f"Sacrifice accepted! +{mana_gain} Mana", "events": events}

    def execute_action(self, action: str) -> Dict:
        action = action.strip()
        
        # Handle SOLVE action separately as it has a parameter
        if action.startswith("SOLVE"):
            # Extract answer
            parts = action.split(maxsplit=1)
            if len(parts) > 1:
                return self.execute_solve_riddle(parts[1])
            else:
                return {"success": False, "message": "What is the answer? Usage: SOLVE <answer>"}

        action_upper = action.upper()
        if action_upper.startswith("MOVE_"):
            direction = action_upper.replace("MOVE_", "")
            return self.execute_move(direction)
        elif action_upper in ("N", "E", "S", "W"):
            return self.execute_move(action_upper)
        elif action_upper == "OPEN":
            return self.execute_open_box()
        elif action_upper == "DISCARD":
            return self.execute_discard_box()
        elif action_upper == "REST":
            return self.execute_rest()
        elif action_upper == "ASK_ORACLE":
            return {"success": True, "message": "Consulting the Oracle...", "events": []}
        else:
            return {"success": False, "message": f"Unknown action: {action}"}
    
    def apply_oracle_supplies(self, supplies: List[Dict]) -> List[str]:
        """Apply supplies granted by the oracle."""
        messages = []
        
        for supply in supplies:
            supply_type = supply.get("type")
            amount = supply.get("amount", 0)
            
            if supply_type == "health":
                old = self.explorer.health
                self.explorer.health = min(self.explorer.max_health, self.explorer.health + amount)
                messages.append(f"✨ Oracle healed you! +{self.explorer.health - old} health")
            
            elif supply_type == "energy":
                old = self.explorer.energy
                self.explorer.energy = min(self.explorer.max_energy, self.explorer.energy + amount)
                messages.append(f"✨ Oracle restored energy! +{self.explorer.energy - old} energy")
            
            elif supply_type == "oil":
                old = self.explorer.oil
                self.explorer.oil = min(self.explorer.max_oil, self.explorer.oil + amount)
                messages.append(f"✨ Oracle restored lamp oil! +{self.explorer.oil - old} oil")
            
            elif supply_type == "cure_poison":
                if self.explorer.poisoned:
                    self.explorer.poisoned = False
                    messages.append("✨ Oracle cured your poison!")
            
            elif supply_type == "identify_box":
                box_pos = supply.get("position")
                contents = supply.get("contents")
                if box_pos and contents:
                    self.explorer.known_boxes[tuple(box_pos)] = contents
                    messages.append(f"✨ Oracle revealed: Box at ({box_pos[0]},{box_pos[1]}) contains {contents.upper()}!")
        
        return messages
    
    def _render_player_grid(self) -> List[str]:
        """Render the dungeon grid with player knowledge applied."""
        grid = []
        for y in range(self.dungeon.height):
            row = ""
            for x in range(self.dungeon.width):
                cell = self.dungeon.get(x, y)
                char = cell.terrain
                
                if char == "B" and (x, y) in self.explorer.known_boxes:
                    contents = self.explorer.known_boxes[(x, y)]
                    if contents == "food":
                        char = "F"
                    elif contents == "poison":
                        char = "P"
                    elif contents == "medicine":
                        char = "+"
                    elif contents == "oil":
                        char = "O"
                
                if (x, y) == self.dungeon.start:
                    char = "S"
                elif (x, y) == self.dungeon.exit:
                    char = "E"
                
                row += char
            grid.append(row)
        return grid
    
    def get_state_dict(self) -> Dict:
        """Get full state as dictionary for API/UI."""
        visible = self.get_visible_cells()
        heard_monsters = self.can_hear_monsters()
        
        all_boxes = []
        for (x, y), cell in self.dungeon.cells.items():
            if cell.terrain == "B":
                ex, ey = self.explorer.pos
                distance = abs(x - ex) + abs(y - ey)
                all_boxes.append({
                    "pos": [x, y],
                    "contents": cell.box_contents,
                    "known_to_player": (x, y) in self.explorer.known_boxes,
                    "distance": distance,
                })
        
        all_monsters = []
        for mx, my in self.dungeon.monsters:
            cell = self.dungeon.get(mx, my)
            ex, ey = self.explorer.pos
            distance = abs(mx - ex) + abs(my - ey)
            all_monsters.append({
                "pos": [mx, my],
                "size": cell.monster_size,
                "hp": cell.monster_hp,
                "distance": distance,
            })
            
        # --- SURGICAL CHANGE: Hide answer from UI/Oracle view ---
        # riddle_ans = self.riddle_master.get_riddle_answer(cell.riddle_id)
        # riddle_ans = "???" 

        all_locked_doors = []
        for (x, y), cell in self.dungeon.cells.items():
            if cell.terrain == "L":
                ex, ey = self.explorer.pos
                distance = abs(x - ex) + abs(y - ey)
                riddle_text = self.riddle_master.get_riddle_text(cell.riddle_id)
                riddle_ans = self.riddle_master.get_riddle_answer(cell.riddle_id)
                riddle_ans = "????"
                all_locked_doors.append({
                    "pos": [x, y],
                    "question": riddle_text,
                    "answer": riddle_ans,
                    "distance": distance,
                    "known_to_player": (x, y) in self.explorer.explored or self._has_line_of_sight(ex, ey, x, y)
                })
        
        dist_to_exit = self.dungeon.distance_to_exit(self.explorer.pos)
        
        oracle_context = None
        if self.explorer.pending_oracle:
            po = self.explorer.pending_oracle
            oracle_context = {
                "question": po.question,
                "position": list(po.position),
                "about_box": list(po.about_box) if po.about_box else None,
                "about_monster": list(po.about_monster) if po.about_monster else None,
                "about_direction": po.about_direction,
            }
        
        floor_cells = sum(1 for (x, y), cell in self.dungeon.cells.items() 
                         if cell.terrain != "#")
        exploration_stats = self.explorer.exploration_stats(floor_cells)
        
        loop_warning = self.explorer.detect_loop()
        
        return {
            "explorer": {
                "pos": list(self.explorer.pos),
                "health": self.explorer.health,
                "max_health": self.explorer.max_health,
                "energy": self.explorer.energy,
                "max_energy": self.explorer.max_energy,
                "oil": self.explorer.oil,
                "max_oil": self.explorer.max_oil,
                "vision_range": self.explorer.vision_range(),
                "poisoned": self.explorer.poisoned,
                "fear_level": self.explorer.fear_level,
                "mood": self.explorer.mood,
                "turn": self.explorer.turn,
                "status": self.explorer.status.value,
                "exploration_stats": exploration_stats,
                "loop_warning": loop_warning,
                "closest_to_exit": self.explorer.closest_to_exit,
                "turns_since_progress": self.explorer.turn - self.explorer.last_progress_turn,
            },
            "dungeon": {
                "width": self.dungeon.width,
                "height": self.dungeon.height,
                "start": list(self.dungeon.start),
                "exit": list(self.dungeon.exit),
                "grid": self._render_player_grid(),
                "distance_to_exit": dist_to_exit,
            },
            "explored": [list(p) for p in self.explorer.explored],
            "notes": self.explorer.notes,
            "known_boxes": {f"{k[0]},{k[1]}": v for k, v in self.explorer.known_boxes.items()},
            "heard_monsters": [[m[0], m[1], m[2], m[3]] for m in heard_monsters],
            
            "oracle_view": {
                "all_boxes": all_boxes,
                "all_monsters": all_monsters,
                "all_locked_doors": all_locked_doors,
            },
            
            "oracle_context": oracle_context,
            
            "oracle": {
                "mana": self.oracle_mana,
                "max_mana": self.oracle_max_mana,
                "pending": self.explorer.pending_oracle is not None,
                "pending_question": self.explorer.pending_oracle.question if self.explorer.pending_oracle else None,
                "last_response": self.explorer.oracle_history[-1].response if self.explorer.oracle_history else None,
            },
            "inner_dialogue": self.explorer.inner_dialogue,
            "thinking": self.explorer.thinking,
            
            # Replaces current_plan
            "active_scroll": {
                "advice_text": self.explorer.active_scroll.advice_text,
                "advice_id": self.explorer.active_scroll.advice_id,
                "actions_taken": self.explorer.active_scroll.actions_taken,
                "author": self.explorer.active_scroll.author
            } if self.explorer.active_scroll else None,
            
            "action_history": self.explorer.action_history[-20:],
            
            "knowledge_graph": {
                "nodes": [n.to_dict() for n in self.explorer.knowledge_graph.nodes.values()] if self.explorer.knowledge_graph else [],
                "stats": self.explorer.knowledge_graph.stats() if self.explorer.knowledge_graph else {},
            },
            
            "scratchpad": self.explorer.scratchpad.to_dict() if self.explorer.scratchpad else {},
        }