"""
This script generates a random maze using Depth-First Search (DFS) algorithm.
The generated maze is saved to a CSV file.
"""
import random


def generate_maze(rows: int, cols: int) -> list[list[int]]:
    """
    Generates a maze using Depth-First Search (DFS) algorithm.
    
    The maze is represented as a 2D grid where:
                                                '1' represents walls,
                                                '0' represents open paths,
                                                'S' is the starting point,
                                                'F' is the ending point.
    
    Args:
        rows (int): Number of rows in the maze. Must be at least 3.
        cols (int): Number of columns in the maze. Must be at least 3.

    Returns:
        List[List[int]]: A 2D list representing the generated maze.
    
    Raises:
        ValueError: If rows or columns are less than 3.
    """
    if rows < 3 or cols < 3:
        raise ValueError('Розмір лабіринту має бути не менше 3x3')

    maze = [[1 for _ in range(cols)] for _ in range(rows)]
    start_row = random.randint(0, (rows - 1) // 2) * 2
    start_col = random.randint(0, (cols - 1) // 2) * 2
    maze[start_row][start_col] = 'S'

    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    stack = [(start_row, start_col)]

    while stack:
        row, col = stack.pop()
        random.shuffle(directions)
        for x, y in directions:
            new_row = row + x
            new_col = col + y

            if 0 <= new_row < rows and 0 <= new_col < cols and maze[new_row][new_col] == 1:
                maze[row + x // 2][col + y // 2] = 0
                maze[new_row][new_col] = 0
                stack.append((new_row, new_col))

    while True:
        finish_row = random.randint(0, (rows - 1) // 2) * 2
        finish_col = random.randint(0, (cols - 1) // 2) * 2
        if maze[finish_row][finish_col] == 0:
            maze[finish_row][finish_col] = 'F'
            break

    return maze


def save_maze_to_csv(maze: list[list[int]], filename: str):
    """
    Saves the generated maze to a CSV file.

    Args:
        maze (list[list[int]]): The maze matrix.
        filename (str): The name of the CSV file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for row in maze:
            file.write(','.join(map(str, row)) + '\n')


if __name__ == '__main__':
    maze_rows, maze_cols = 25, 25  # Maze dimensions (should be odd).

    maze_matrix = generate_maze(maze_rows, maze_cols)
    save_maze_to_csv(maze_matrix, 'maze.csv')
    print("Maze is saved to 'maze.csv'")
