# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1,1, 1,1, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1  # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']


def compute_value(grid, goal, cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------

    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    value = [[99 for i in range(len(row))] for row in grid]
    closed = [[0 for i in range(len(row))] for row in grid]

    x = goal[0]
    y = goal[1]
    open = []
    open.append([0, x, y])
    while open:
        open.sort()
        open.reverse()
        c_cost, c_x, c_y = open.pop()
        value[c_x][c_y] = c_cost
        closed[c_x][c_y] = 1
        for i in range(len(delta)):
            x2 = c_x + delta[i][0]
            y2 = c_y + delta[i][1]
            if 0 <= x2 and x2 < len(grid) and 0 <= y2 and y2 < len(grid[0]) and closed[x2][y2] == 0 and grid[x2][
                y2] != 1:
                open.append([c_cost + 1, x2, y2])

    return value


value = compute_value(grid, goal, cost)
for i in range(len(value)):
    print(value[i])
