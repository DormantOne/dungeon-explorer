"""
DUNGEON MAZE GENERATOR v3
=========================
Room-and-corridor dungeon with clean walls.

Terrain Types:
- '#' = Wall (impassable)
- '.' = Floor (safe)
- 'S' = Start position
- 'E' = Exit (goal)
- 'B' = Box (unknown substance - food/poison/medicine/oil)
- 'D' = Darkness (reduced vision, needs extra oil)
- 'R' = Rubble (costs extra energy to pass)
- 'M' = Monster (danger! size determined separately)
- 'L' = Locked Door (requires riddle solution)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import random

# Cardinal directions only - NO DIAGONALS
DIRECTIONS = {
    "N": (0, -1),  # North = up = y decreases
    "E": (1, 0),   # East = right = x increases
    "S": (0, 1),   # South = down = y increases
    "W": (-1, 0),  # West = left = x decreases
}


@dataclass
class DungeonCell:
    """A single cell in the dungeon."""
    x: int
    y: int
    terrain: str = "."
    
    # For boxes - what's inside (unknown to player until oracle reveals)
    box_contents: Optional[str] = None  # "food", "poison", "medicine", "oil", None
    
    # For monsters - their properties
    monster_size: Optional[str] = None  # "small", "medium", "large"
    monster_hp: int = 0
    
    # For Locked Doors ("L")
    riddle_id: Optional[int] = None
    
    def is_passable(self) -> bool:
        # L is not passable until solved (turned into .)
        return self.terrain not in ("#", "L")
    
    def is_dangerous(self) -> bool:
        return self.terrain == "M"


@dataclass 
class Dungeon:
    """The dungeon map."""
    width: int
    height: int
    cells: Dict[Tuple[int, int], DungeonCell] = field(default_factory=dict)
    start: Tuple[int, int] = (1, 1)
    exit: Tuple[int, int] = (1, 1)
    seed: int = 0
    
    # Monster tracking
    monsters: List[Tuple[int, int]] = field(default_factory=list)
    
    def get(self, x: int, y: int) -> DungeonCell:
        """Get cell at position (x, y)."""
        if (x, y) in self.cells:
            return self.cells[(x, y)]
        # Out of bounds = wall
        return DungeonCell(x, y, "#")
    
    def set_terrain(self, x: int, y: int, terrain: str):
        """Set terrain at position (x, y)."""
        if (x, y) not in self.cells:
            self.cells[(x, y)] = DungeonCell(x, y)
        self.cells[(x, y)].terrain = terrain
    
    def passable(self, x: int, y: int) -> bool:
        """Check if cell can be walked through."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.get(x, y).is_passable()
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get passable neighbors (cardinal directions only)."""
        neighbors = []
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if self.passable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """BFS to find path between two points."""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) == end:
                    return path + [(nx, ny)]
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return []  # No path found
    
    def is_solvable(self) -> bool:
        """Check if exit is reachable from start."""
        return len(self.find_path(self.start, self.exit)) > 0
    
    def distance_to_exit(self, pos: Tuple[int, int]) -> int:
        """Get shortest distance from pos to exit."""
        path = self.find_path(pos, self.exit)
        return len(path) - 1 if path else 999
    
    def render_grid(self) -> List[str]:
        """
        Render dungeon as list of strings.
        grid[y] is a row, grid[y][x] is the cell at (x, y).
        """
        grid = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                cell = self.get(x, y)
                if (x, y) == self.start:
                    row += "S"
                elif (x, y) == self.exit:
                    row += "E"
                else:
                    row += cell.terrain
            grid.append(row)
        return grid
    
    def render_string(self) -> str:
        """Render dungeon as single string."""
        return "\n".join(self.render_grid())


def _create_maze_dungeon(width: int, height: int, rng: random.Random, 
                          room_density: float = 0.15) -> List[List[str]]:
    """
    Generate a more maze-like dungeon with winding corridors and occasional rooms.
    Uses recursive backtracking to create a maze, then carves out a few rooms.
    """
    # Start with all walls
    grid = [["#" for _ in range(width)] for _ in range(height)]
    
    # Use odd coordinates for the maze (standard maze generation)
    maze_width = (width - 1) // 2
    maze_height = (height - 1) // 2
    
    # Track visited cells for maze generation
    visited = set()
    
    def carve(x: int, y: int):
        """Recursive backtracker maze carving."""
        visited.add((x, y))
        # Convert maze coords to grid coords
        gx, gy = x * 2 + 1, y * 2 + 1
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = "."
        
        # Randomize direction order
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        rng.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze_width and 0 <= ny < maze_height and (nx, ny) not in visited:
                # Carve passage between cells
                px, py = x * 2 + 1 + dx, y * 2 + 1 + dy
                if 0 <= px < width and 0 <= py < height:
                    grid[py][px] = "."
                carve(nx, ny)
    
    # Start maze from center-ish position
    start_x = maze_width // 3
    start_y = maze_height // 3
    carve(start_x, start_y)
    
    # Now add some rooms by carving out rectangular areas
    num_rooms = int(width * height * room_density / 16)  # Fewer rooms
    num_rooms = max(3, min(8, num_rooms))
    
    rooms = []
    for _ in range(num_rooms * 3):  # Try multiple times
        if len(rooms) >= num_rooms:
            break
        
        # Smaller rooms for maze-like feel
        rw = rng.randint(2, 4)
        rh = rng.randint(2, 4)
        rx = rng.randint(2, width - rw - 2)
        ry = rng.randint(2, height - rh - 2)
        
        # Check if this room would connect to existing maze
        connects = False
        for y in range(max(0, ry-1), min(height, ry + rh + 1)):
            for x in range(max(0, rx-1), min(width, rx + rw + 1)):
                if grid[y][x] == ".":
                    connects = True
                    break
            if connects:
                break
        
        if connects:
            # Carve the room
            for y in range(ry, min(height - 1, ry + rh)):
                for x in range(rx, min(width - 1, rx + rw)):
                    grid[y][x] = "."
            rooms.append((rx, ry, rw, rh))
    
    # Add extra corridor connections for loops (prevent dead-end frustration)
    dead_ends = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] == ".":
                # Count adjacent floors
                adj_floors = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)] 
                               if 0 <= x+dx < width and 0 <= y+dy < height 
                               and grid[y+dy][x+dx] == ".")
                if adj_floors == 1:
                    dead_ends.append((x, y))
    
    # Connect some dead ends to nearby passages
    for x, y in dead_ends[:len(dead_ends)//2]:
        if rng.random() < 0.4:  # 40% chance to add shortcut
            for dx, dy in [(0,2),(0,-2),(2,0),(-2,0)]:
                nx, ny = x + dx, y + dy
                mx, my = x + dx//2, y + dy//2
                if (0 < nx < width - 1 and 0 < ny < height - 1 
                    and grid[ny][nx] == "." and grid[my][mx] == "#"):
                    grid[my][mx] = "."
                    break
    
    # Widen some corridors for variety
    widen_count = (width * height) // 100
    floor_cells = [(x, y) for y in range(1, height-1) for x in range(1, width-1) if grid[y][x] == "."]
    
    for _ in range(widen_count):
        if floor_cells:
            x, y = rng.choice(floor_cells)
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                if rng.random() < 0.3:
                    nx, ny = x + dx, y + dy
                    if 0 < nx < width - 1 and 0 < ny < height - 1:
                        grid[ny][nx] = "."
    
    return grid


def _create_room_dungeon(width: int, height: int, rng: random.Random, 
                          num_rooms: int = 10, min_room: int = 2, max_room: int = 5) -> List[List[str]]:
    """
    Generate a complex dungeon with rooms, corridors, and interesting features.
    """
    # Start with all walls
    grid = [["#" for _ in range(width)] for _ in range(height)]
    
    rooms: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
    room_centers: List[Tuple[int, int]] = []
    
    # Place rooms with varied sizes
    attempts = 0
    max_attempts = 200
    
    while len(rooms) < num_rooms and attempts < max_attempts:
        attempts += 1
        
        if rng.random() < 0.3:  # 30% small alcoves
            rw = rng.randint(2, 3)
            rh = rng.randint(2, 3)
        elif rng.random() < 0.2:  # 20% large chambers
            rw = rng.randint(4, max_room)
            rh = rng.randint(4, max_room)
        else:  # 50% medium rooms
            rw = rng.randint(min_room, 4)
            rh = rng.randint(min_room, 4)
        
        rx = rng.randint(2, width - rw - 2)
        ry = rng.randint(2, height - rh - 2)
        
        # Check for overlap
        overlap = False
        for (ex, ey, ew, eh) in rooms:
            if (rx < ex + ew + 1 and rx + rw + 1 > ex and
                ry < ey + eh + 1 and ry + rh + 1 > ey):
                overlap = True
                break
        
        if not overlap:
            rooms.append((rx, ry, rw, rh))
            cx = rx + rw // 2
            cy = ry + rh // 2
            room_centers.append((cx, cy))
            
            # Carve out the room
            for y in range(ry, ry + rh):
                for x in range(rx, rx + rw):
                    grid[y][x] = "."
    
    # Guarantee rooms in quadrants
    quadrants = [
        (2, 2, width // 2 - 2, height // 2 - 2),
        (width // 2, 2, width - 4, height // 2 - 2),
        (2, height // 2, width // 2 - 2, height - 4),
        (width // 2, height // 2, width - 4, height - 4),
    ]
    
    for qi, (qx1, qy1, qx2, qy2) in enumerate(quadrants):
        has_room = any(
            qx1 <= rx + rw // 2 <= qx2 and qy1 <= ry + rh // 2 <= qy2
            for (rx, ry, rw, rh) in rooms
        )
        if not has_room:
            rx = rng.randint(qx1 + 1, max(qx1 + 2, qx2 - 4))
            ry = rng.randint(qy1 + 1, max(qy1 + 2, qy2 - 4))
            for y in range(ry, min(ry + 3, height - 2)):
                for x in range(rx, min(rx + 3, width - 2)):
                    grid[y][x] = "."
            rooms.append((rx, ry, 3, 3))
            room_centers.append((rx + 1, ry + 1))
    
    # Connect rooms with minimum spanning tree
    sorted_centers = sorted(enumerate(room_centers), key=lambda x: x[1][0] + x[1][1])
    connected = {sorted_centers[0][0]}
    
    while len(connected) < len(room_centers):
        best_dist = float('inf')
        best_pair = None
        
        for ci in connected:
            cx1, cy1 = room_centers[ci]
            for uj, (cx2, cy2) in enumerate(room_centers):
                if uj not in connected:
                    dist = abs(cx1 - cx2) + abs(cy1 - cy2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (ci, uj)
        
        if best_pair:
            i, j = best_pair
            connected.add(j)
            
            x1, y1 = room_centers[i]
            x2, y2 = room_centers[j]
            
            if rng.random() < 0.5:
                dx = 1 if x2 > x1 else -1
                x = x1
                while x != x2:
                    grid[y1][x] = "."
                    x += dx
                grid[y1][x2] = "."
                
                dy = 1 if y2 > y1 else -1
                y = y1
                while y != y2:
                    grid[y][x2] = "."
                    y += dy
                grid[y2][x2] = "."
            else:
                dy = 1 if y2 > y1 else -1
                y = y1
                while y != y2:
                    grid[y][x1] = "."
                    y += dy
                grid[y2][x1] = "."
                
                dx = 1 if x2 > x1 else -1
                x = x1
                while x != x2:
                    grid[y2][x] = "."
                    x += dx
                grid[y2][x2] = "."
    
    # Add extra corridors for loops
    num_extra_corridors = len(rooms) // 2 + 2
    for _ in range(num_extra_corridors):
        if len(room_centers) >= 2:
            i = rng.randint(0, len(room_centers) - 1)
            j = rng.randint(0, len(room_centers) - 1)
            if i != j:
                x1, y1 = room_centers[i]
                x2, y2 = room_centers[j]
                
                if abs(x1 - x2) + abs(y1 - y2) < width:
                    dx = 1 if x2 > x1 else -1
                    x = x1
                    while x != x2:
                        grid[y1][x] = "."
                        x += dx
                    
                    dy = 1 if y2 > y1 else -1
                    y = y1
                    while y != y2:
                        grid[y][x2] = "."
                        y += dy
    
    # Add dead-end corridors
    num_dead_ends = rng.randint(2, 5)
    for _ in range(num_dead_ends):
        floor_cells = [(x, y) for y in range(2, height - 2) 
                       for x in range(2, width - 2) if grid[y][x] == "."]
        if floor_cells:
            sx, sy = rng.choice(floor_cells)
            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            dx, dy = rng.choice(directions)
            length = rng.randint(2, 4)
            x, y = sx, sy
            for _ in range(length):
                nx, ny = x + dx, y + dy
                if 2 <= nx < width - 2 and 2 <= ny < height - 2:
                    grid[ny][nx] = "."
                    x, y = nx, ny
                else:
                    break
    
    return grid


def generate_dungeon(
    size: int = 15,
    seed: Optional[int] = None,
    num_boxes: int = 5,
    num_monsters: int = 3,
    num_darkness: int = 4,
    num_rubble: int = 3,
    maze_style: bool = True,
) -> Dungeon:
    """
    Generate a solvable SQUARE dungeon with rooms and items.
    """
    size = max(15, size)
    if size % 2 == 0:
        size += 1
    width = size
    height = size
    
    if seed is None:
        seed = random.randint(1, 999999999)
    rng = random.Random(seed)
    
    # Scale items with dungeon size
    area = width * height
    if num_boxes == 5:
        num_boxes = max(5, area // 50)
    if num_monsters == 3:
        num_monsters = max(3, area // 80)
    if num_darkness == 4:
        num_darkness = max(4, area // 60)
    if num_rubble == 3:
        num_rubble = max(3, area // 70)
    
    for attempt in range(50):
        if maze_style:
            grid = _create_maze_dungeon(width, height, rng, room_density=0.12)
        else:
            num_rooms = max(8, (size * size) // 20)
            grid = _create_room_dungeon(width, height, rng, num_rooms=num_rooms)
        
        dungeon = Dungeon(width=width, height=height, seed=seed + attempt)
        
        for y in range(height):
            for x in range(width):
                dungeon.cells[(x, y)] = DungeonCell(x, y, grid[y][x])
        
        floor_cells = [(x, y) for (x, y), cell in dungeon.cells.items() 
                       if cell.terrain == "."]
        
        if len(floor_cells) < 10:
            continue
        
        upper_left = [(x, y) for (x, y) in floor_cells 
                      if x < width // 2 and y < height // 2]
        if upper_left:
            dungeon.start = rng.choice(upper_left)
        else:
            dungeon.start = floor_cells[0]
        
        lower_right = [(x, y) for (x, y) in floor_cells 
                       if x > width // 2 and y > height // 2]
        if lower_right:
            lower_right.sort(key=lambda p: -(abs(p[0] - dungeon.start[0]) + abs(p[1] - dungeon.start[1])))
            dungeon.exit = lower_right[0]
        else:
            floor_cells.sort(key=lambda p: -(abs(p[0] - dungeon.start[0]) + abs(p[1] - dungeon.start[1])))
            dungeon.exit = floor_cells[0]
        
        dungeon.set_terrain(dungeon.start[0], dungeon.start[1], ".")
        dungeon.set_terrain(dungeon.exit[0], dungeon.exit[1], ".")
        
        if not dungeon.is_solvable():
            continue
        
        open_cells = [
            (x, y) for (x, y), cell in dungeon.cells.items()
            if cell.terrain == "." and (x, y) != dungeon.start and (x, y) != dungeon.exit
        ]
        rng.shuffle(open_cells)
        
        if len(open_cells) < num_boxes + num_monsters + num_darkness + num_rubble + 4:
            continue
        
        placed = 0
        
        # Boxes (Updated with 'oil')
        box_types = ["food", "poison", "medicine", "oil"]
        for i in range(min(num_boxes, len(open_cells) - placed)):
            x, y = open_cells[placed]
            dungeon.set_terrain(x, y, "B")
            cell = dungeon.get(x, y)
            cell.box_contents = rng.choice(box_types)
            placed += 1
        
        # Monsters
        monster_sizes = ["small", "medium", "large"]
        monster_hp = {"small": 10, "medium": 25, "large": 50}
        monsters_placed = 0
        for i in range(placed, len(open_cells)):
            if monsters_placed >= num_monsters:
                break
            x, y = open_cells[i]
            if abs(x - dungeon.start[0]) + abs(y - dungeon.start[1]) < 4:
                continue
            dungeon.set_terrain(x, y, "M")
            cell = dungeon.get(x, y)
            size = rng.choice(monster_sizes)
            cell.monster_size = size
            cell.monster_hp = monster_hp[size]
            dungeon.monsters.append((x, y))
            monsters_placed += 1
        placed += monsters_placed
        
        # Darkness
        for i in range(min(num_darkness, len(open_cells) - placed)):
            if placed >= len(open_cells):
                break
            x, y = open_cells[placed]
            if dungeon.get(x, y).terrain == ".":
                dungeon.set_terrain(x, y, "D")
                placed += 1
        
        # Rubble
        for i in range(min(num_rubble, len(open_cells) - placed)):
            if placed >= len(open_cells):
                break
            x, y = open_cells[placed]
            if dungeon.get(x, y).terrain == ".":
                dungeon.set_terrain(x, y, "R")
                placed += 1

        # Locked Doors ("L") - Place 4
        # We try to place them at choke points or just random locations.
        # Random for now, but ensure we pick spots that were floors.
        num_locked_doors = 4
        doors_placed = 0
        
        # Re-scan for floor cells as we modified terrain
        current_floor_cells = [
            (x, y) for (x, y), cell in dungeon.cells.items()
            if cell.terrain == "." and (x, y) != dungeon.start and (x, y) != dungeon.exit
        ]
        rng.shuffle(current_floor_cells)
        
        for i in range(min(num_locked_doors, len(current_floor_cells))):
            x, y = current_floor_cells[i]
            dungeon.set_terrain(x, y, "L")
            cell = dungeon.get(x, y)
            # Assign riddle ID (0 or 1 alternating or random)
            cell.riddle_id = i % 2 
            doors_placed += 1
        
        if dungeon.is_solvable(): # Re-check solvability isn't strictly necessary if L blocks path, we assume player can solve.
            path_len = dungeon.distance_to_exit(dungeon.start)
            print(f"[MazeGen] Generated solvable dungeon in {attempt + 1} attempt(s)")
            print(f"[MazeGen] Size: {width}x{height}, Path length: {path_len}")
            print(f"[MazeGen] Start: {dungeon.start}, Exit: {dungeon.exit}")
            print(f"[MazeGen] Boxes: {num_boxes}, Monsters: {len(dungeon.monsters)}, Doors: {doors_placed}")
            return dungeon
    
    # Fallback
    print("[MazeGen] WARNING: Using fallback dungeon generation")
    dungeon = Dungeon(width=width, height=height, seed=seed)
    
    for y in range(height):
        for x in range(width):
            dungeon.cells[(x, y)] = DungeonCell(x, y, "#")
    
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            dungeon.set_terrain(x, y, ".")
    
    dungeon.start = (3, 3)
    dungeon.exit = (width - 4, height - 4)
    
    return dungeon


if __name__ == "__main__":
    print("Testing maze generator...")
    
    dungeon = generate_dungeon(size=15, seed=42)
    
    print(f"\nDungeon {dungeon.width}x{dungeon.height}, seed={dungeon.seed}")
    print(f"Start: {dungeon.start}, Exit: {dungeon.exit}")
    print(f"Solvable: {dungeon.is_solvable()}")
    print(f"Path length: {dungeon.distance_to_exit(dungeon.start)}")
    print(f"Monsters: {len(dungeon.monsters)}")
    
    print("\nMap:")
    print(dungeon.render_string())