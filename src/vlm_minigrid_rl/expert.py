from collections import deque


def turn_balance(actions):
    return actions.count(0) - actions.count(1)


def get_shortest_path_actions(env, action_order=(0, 1, 2)):
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir

    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    if not goal_pos:
        return []

    dir_delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    start = (agent_pos[0], agent_pos[1], agent_dir)
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (x, y, d), path = queue.popleft()
        if (x, y) == goal_pos:
            return path
        for action in action_order:
            nx, ny, nd = x, y, d
            if action == 0:
                nd = (d - 1) % 4
            elif action == 1:
                nd = (d + 1) % 4
            else:
                dx, dy = dir_delta[d]
                nx = x + dx
                ny = y + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                cell = grid.get(nx, ny)
                if cell is not None and cell.type == "wall":
                    continue
            new_state = (nx, ny, nd)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [action]))
    return []
