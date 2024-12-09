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
        raise ValueError("Розмір лабіринту має бути не менше 3x3")

    def dfs(row_index: int, col_index: int) -> None:
        """
        Recursive helper function to carve paths in the maze.

        Args:
            row (int): The current row position.
            col (int): The current column position.
        """
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        for x, y in directions:
            new_row_index = row_index + x
            new_col_index = col_index + y

            if 0 <= new_row_index < rows and 0 <= new_col_index < cols:
                if maze[new_row_index][new_col_index] == 1:
                    maze[row_index + x // 2][col_index + y // 2] = 0
                    maze[new_row_index][new_col_index] = 0
                    dfs(new_row_index, new_col_index)

    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    start_row = random.randint(0, (rows - 1) // 2) * 2
    start_col = random.randint(0, (cols - 1) // 2) * 2
    maze[start_row][start_col] = 'S'

    dfs(start_row, start_col)

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


if __name__ == "__main__":
    rows, cols = 21, 21  # Maze dimensions (must be odd)
    maze = generate_maze(rows, cols)
    save_maze_to_csv(maze, "maze.csv")
    print("Maze saved to 'maze.csv'")
