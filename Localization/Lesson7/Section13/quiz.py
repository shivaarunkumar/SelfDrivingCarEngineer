# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
#
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left,
# up, and down motions. Note that the 'v' should be
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']


def search(grid, init, goal, cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1

    action = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    x = init[0]
    y = init[1]
    g = 0

    open = [[g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(open) == 0:
            resign = True
            return 'fail'
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]

            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i

    return action  # make sure you return the shortest path


action = search(grid, init, goal, cost)
path = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
x = goal[0]
y = goal[1]
path[x][y] = '*'
while not (x == init[0] and y == init[1]):
    prev_action = action[x][y]
    x = x - delta[prev_action][0]
    y = y - delta[prev_action][1]
    path[x][y] = delta_name[prev_action]

[print(row) for row in action]
[print(row) for row in path]
