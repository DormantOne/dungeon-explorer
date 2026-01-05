"""
KNOWLEDGE GRAPH MEMORY SYSTEM
=============================
A memory system where:
- AI adds facts as nodes
- Nodes have strength that decays over time
- AI can reinforce nodes to keep them alive
- Forgotten nodes (strength < threshold) are removed
- Used for reasoning and avoiding repeated mistakes

Node Types:
- BOX: Information about a box (position, contents, status)
- MONSTER: Information about a monster
- PATH: Information about routes/directions
- DANGER: Warnings and hazards
- GOAL: Objectives and plans
- OBSERVATION: General observations
- ORACLE_ADVICE: Advice received from oracles (NEW)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import time


class NodeType(Enum):
    BOX = "box"
    MONSTER = "monster"
    PATH = "path"
    DANGER = "danger"
    GOAL = "goal"
    OBSERVATION = "observation"
    ACTION = "action"  # Past actions and their outcomes
    ORACLE_ADVICE = "oracle_advice"  # NEW: Track oracle advice quality


@dataclass
class KGNode:
    """A node in the knowledge graph."""
    id: str
    node_type: NodeType
    content: str
    position: Optional[Tuple[int, int]] = None
    
    # Memory strength (0-100)
    strength: float = 100.0
    
    # Timing
    created_turn: int = 0
    last_accessed_turn: int = 0
    access_count: int = 1
    
    # Connections to other nodes
    related_to: Set[str] = field(default_factory=set)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def decay(self, current_turn: int, decay_rate: float = 5.0) -> float:
        """
        Decay the node's strength based on time since last access.
        Returns new strength.
        """
        turns_since_access = current_turn - self.last_accessed_turn
        decay = decay_rate * turns_since_access
        self.strength = max(0, self.strength - decay)
        return self.strength
    
    def reinforce(self, amount: float = 20.0, current_turn: int = 0) -> float:
        """
        Reinforce this memory (AI is using it).
        Returns new strength.
        """
        self.strength = min(100, self.strength + amount)
        self.last_accessed_turn = current_turn
        self.access_count += 1
        return self.strength
    
    def is_alive(self, threshold: float = 10.0) -> bool:
        """Check if node is still above forgetting threshold."""
        return self.strength >= threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.node_type.value,
            "content": self.content,
            "position": list(self.position) if self.position else None,
            "strength": round(self.strength, 1),
            "created_turn": self.created_turn,
            "last_accessed_turn": self.last_accessed_turn,
            "access_count": self.access_count,
            "related_to": list(self.related_to),
            "metadata": self.metadata,
        }


class KnowledgeGraph:
    """
    Knowledge Graph for AI memory.
    Nodes decay over time unless reinforced.
    """
    
    def __init__(self, decay_rate: float = 3.0, forget_threshold: float = 10.0):
        self.nodes: Dict[str, KGNode] = {}
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.node_counter = 0
        
        # Index by type for faster lookups
        self.by_type: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
        
        # Index by position
        self.by_position: Dict[Tuple[int, int], Set[str]] = {}
    
    def _generate_id(self, node_type: NodeType) -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"{node_type.value}_{self.node_counter}"
    
    def add_node(
        self,
        node_type: NodeType,
        content: str,
        position: Optional[Tuple[int, int]] = None,
        current_turn: int = 0,
        metadata: Dict = None,
        related_to: List[str] = None,
    ) -> KGNode:
        """
        Add a new knowledge node.
        Returns the created node.
        """
        node_id = self._generate_id(node_type)
        
        node = KGNode(
            id=node_id,
            node_type=node_type,
            content=content,
            position=position,
            created_turn=current_turn,
            last_accessed_turn=current_turn,
            metadata=metadata or {},
            related_to=set(related_to) if related_to else set(),
        )
        
        self.nodes[node_id] = node
        self.by_type[node_type].add(node_id)
        
        if position:
            if position not in self.by_position:
                self.by_position[position] = set()
            self.by_position[position].add(node_id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def find_by_content(self, search: str, node_type: Optional[NodeType] = None) -> List[KGNode]:
        """Find nodes containing search string."""
        search_lower = search.lower()
        results = []
        
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if search_lower in node.content.lower():
                results.append(node)
        
        return results
    
    def find_by_position(self, position: Tuple[int, int]) -> List[KGNode]:
        """Find all nodes related to a position."""
        node_ids = self.by_position.get(position, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def find_by_type(self, node_type: NodeType) -> List[KGNode]:
        """Find all nodes of a type."""
        node_ids = self.by_type.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def reinforce_node(self, node_id: str, amount: float = 20.0, current_turn: int = 0) -> bool:
        """
        Reinforce a node (AI is using this knowledge).
        Returns True if node exists and was reinforced.
        """
        node = self.nodes.get(node_id)
        if node:
            node.reinforce(amount, current_turn)
            return True
        return False
    
    def reinforce_by_content(self, search: str, amount: float = 15.0, current_turn: int = 0) -> int:
        """
        Reinforce all nodes matching content.
        Returns count of reinforced nodes.
        """
        count = 0
        for node in self.find_by_content(search):
            node.reinforce(amount, current_turn)
            count += 1
        return count
    
    def process_decay(self, current_turn: int) -> List[KGNode]:
        """
        Process decay for all nodes.
        Returns list of forgotten (removed) nodes.
        """
        forgotten = []
        to_remove = []
        
        for node_id, node in self.nodes.items():
            node.decay(current_turn, self.decay_rate)
            if not node.is_alive(self.forget_threshold):
                forgotten.append(node)
                to_remove.append(node_id)
        
        # Remove forgotten nodes
        for node_id in to_remove:
            self._remove_node(node_id)
        
        return forgotten
    
    def _remove_node(self, node_id: str):
        """Remove a node from all indices."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        # Remove from type index
        if node.node_type in self.by_type:
            self.by_type[node.node_type].discard(node_id)
        
        # Remove from position index
        if node.position and node.position in self.by_position:
            self.by_position[node.position].discard(node_id)
        
        # Remove from main dict
        del self.nodes[node_id]
    
    def get_strongest_memories(self, n: int = 10, node_type: Optional[NodeType] = None) -> List[KGNode]:
        """Get the N strongest memories, optionally filtered by type."""
        if node_type:
            nodes = self.find_by_type(node_type)
        else:
            nodes = list(self.nodes.values())
        
        return sorted(nodes, key=lambda x: x.strength, reverse=True)[:n]
    
    def get_recent_memories(self, n: int = 10, current_turn: int = 0) -> List[KGNode]:
        """Get the N most recently accessed memories."""
        nodes = list(self.nodes.values())
        return sorted(nodes, key=lambda x: x.last_accessed_turn, reverse=True)[:n]
    
    def get_oracle_advice_history(self) -> List[KGNode]:
        """Get all oracle advice memories, sorted by turn."""
        advice_nodes = self.find_by_type(NodeType.ORACLE_ADVICE)
        return sorted(advice_nodes, key=lambda x: x.created_turn, reverse=True)
    
    def render_for_ai(self, current_turn: int, max_nodes: int = 15) -> str:
        """
        Render knowledge graph as text for AI prompt.
        Shows strongest memories with strength indicators.
        """
        if not self.nodes:
            return "No memories yet."
        
        lines = []
        
        # Get strongest memories across types
        all_nodes = sorted(self.nodes.values(), key=lambda x: x.strength, reverse=True)[:max_nodes]
        
        # Group by type
        by_type: Dict[NodeType, List[KGNode]] = {}
        for node in all_nodes:
            if node.node_type not in by_type:
                by_type[node.node_type] = []
            by_type[node.node_type].append(node)
        
        # Render each type
        type_icons = {
            NodeType.BOX: "📦",
            NodeType.MONSTER: "👹",
            NodeType.PATH: "🛤️",
            NodeType.DANGER: "⚠️",
            NodeType.GOAL: "🎯",
            NodeType.OBSERVATION: "👁️",
            NodeType.ACTION: "⚡",
            NodeType.ORACLE_ADVICE: "🔮",
        }
        
        for ntype, nodes in by_type.items():
            icon = type_icons.get(ntype, "•")
            lines.append(f"\n{icon} {ntype.value.upper()}:")
            for node in nodes:
                strength_bar = "█" * int(node.strength / 10) + "░" * (10 - int(node.strength / 10))
                pos_str = f" @({node.position[0]},{node.position[1]})" if node.position else ""
                lines.append(f"  [{strength_bar}] {node.content}{pos_str}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert entire graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "decay_rate": self.decay_rate,
            "forget_threshold": self.forget_threshold,
            "node_counter": self.node_counter,
        }
    
    def stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            "total_nodes": len(self.nodes),
            "by_type": {t.value: len(ids) for t, ids in self.by_type.items()},
            "avg_strength": sum(n.strength for n in self.nodes.values()) / max(1, len(self.nodes)),
            "positions_tracked": len(self.by_position),
        }


# ============================================================================
# AI SCRATCHPAD
# ============================================================================

@dataclass
class Scratchpad:
    """
    A scratchpad for AI to do multi-step thinking.
    Persists between decisions for continuity.
    """
    
    thoughts: List[str] = field(default_factory=list)
    current_goal: str = ""
    sub_goals: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    
    # Planning
    planned_moves: List[str] = field(default_factory=list)
    backup_plan: str = ""
    
    # Learning
    lessons_learned: List[str] = field(default_factory=list)
    mistakes: List[str] = field(default_factory=list)
    
    # NEW: Oracle advice tracking
    current_advice_id: Optional[str] = None  # ID of current oracle advice being followed
    advice_start_health: int = 0
    advice_start_energy: int = 0
    advice_start_turn: int = 0
    
    def add_thought(self, thought: str):
        """Add a thought, keeping last 20."""
        self.thoughts.append(thought)
        if len(self.thoughts) > 20:
            self.thoughts = self.thoughts[-20:]
    
    def add_lesson(self, lesson: str):
        """Add a lesson learned."""
        if lesson not in self.lessons_learned:
            self.lessons_learned.append(lesson)
            if len(self.lessons_learned) > 10:
                self.lessons_learned = self.lessons_learned[-10:]
    
    def add_mistake(self, mistake: str):
        """Record a mistake to avoid repeating."""
        if mistake not in self.mistakes:
            self.mistakes.append(mistake)
            if len(self.mistakes) > 10:
                self.mistakes = self.mistakes[-10:]
    
    def set_goal(self, goal: str, sub_goals: List[str] = None):
        """Set current goal and sub-goals."""
        self.current_goal = goal
        self.sub_goals = sub_goals or []
    
    def start_advice_tracking(self, advice_id: str, health: int, energy: int, turn: int):
        """Start tracking an oracle advice session."""
        self.current_advice_id = advice_id
        self.advice_start_health = health
        self.advice_start_energy = energy
        self.advice_start_turn = turn
    
    def clear_advice_tracking(self):
        """Clear the current advice tracking."""
        self.current_advice_id = None
        self.advice_start_health = 0
        self.advice_start_energy = 0
        self.advice_start_turn = 0
    
    def render_for_ai(self) -> str:
        """Render scratchpad as text for AI prompt."""
        lines = []
        
        if self.current_goal:
            lines.append(f"🎯 CURRENT GOAL: {self.current_goal}")
            if self.sub_goals:
                for sg in self.sub_goals[:5]:
                    lines.append(f"   → {sg}")
        
        if self.planned_moves:
            lines.append(f"\n📋 PLANNED MOVES: {' → '.join(self.planned_moves[:5])}")
        
        if self.lessons_learned:
            lines.append("\n📚 LESSONS LEARNED:")
            for lesson in self.lessons_learned[-5:]:
                lines.append(f"   • {lesson}")
        
        if self.mistakes:
            lines.append("\n❌ MISTAKES TO AVOID:")
            for mistake in self.mistakes[-3:]:
                lines.append(f"   • {mistake}")
        
        if self.thoughts:
            lines.append("\n💭 RECENT THOUGHTS:")
            for thought in self.thoughts[-5:]:
                lines.append(f"   • {thought}")
        
        return "\n".join(lines) if lines else "Scratchpad empty - start planning!"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "current_goal": self.current_goal,
            "sub_goals": self.sub_goals,
            "planned_moves": self.planned_moves,
            "lessons_learned": self.lessons_learned,
            "mistakes": self.mistakes,
            "thoughts": self.thoughts[-10:],
            "current_advice_id": self.current_advice_id,
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Knowledge Graph...")
    
    kg = KnowledgeGraph(decay_rate=5.0, forget_threshold=10.0)
    
    # Add some nodes
    n1 = kg.add_node(NodeType.BOX, "Box at (5,3) contains FOOD - safe!", position=(5, 3), current_turn=1)
    n2 = kg.add_node(NodeType.MONSTER, "Large monster near exit - dangerous!", position=(10, 10), current_turn=1)
    n3 = kg.add_node(NodeType.PATH, "Go EAST then SOUTH to reach exit", current_turn=2)
    n4 = kg.add_node(NodeType.DANGER, "Avoid the darkness at (7,7) - drains oil fast", position=(7, 7), current_turn=2)
    
    print(f"\nAdded {len(kg.nodes)} nodes")
    print(kg.render_for_ai(current_turn=2))
    
    # Simulate decay
    print("\n--- After 5 turns (decay) ---")
    forgotten = kg.process_decay(current_turn=7)
    print(f"Forgotten: {len(forgotten)} nodes")
    print(kg.render_for_ai(current_turn=7))
    
    # Reinforce a memory
    print("\n--- After reinforcing BOX memory ---")
    kg.reinforce_by_content("FOOD", amount=50, current_turn=7)
    print(kg.render_for_ai(current_turn=7))
    
    # More decay
    print("\n--- After 10 more turns ---")
    forgotten = kg.process_decay(current_turn=17)
    print(f"Forgotten: {len(forgotten)} nodes")
    print(kg.render_for_ai(current_turn=17))
    
    print("\n✓ Knowledge Graph works!")
    
    # Test scratchpad
    print("\n\n--- Testing Scratchpad ---")
    sp = Scratchpad()
    sp.set_goal("Reach the exit", ["Find food first", "Avoid the large monster", "Keep oil above 20"])
    sp.add_thought("The exit is to the southeast")
    sp.add_thought("I should ask Oracle about that box")
    sp.add_lesson("Opening unknown boxes is dangerous")
    sp.add_mistake("Walked into darkness without enough oil")
    sp.planned_moves = ["MOVE_E", "MOVE_E", "MOVE_S"]
    
    print(sp.render_for_ai())
    print("\n✓ Scratchpad works!")
