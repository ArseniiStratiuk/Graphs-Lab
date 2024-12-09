"""
This module contains functions for reading a matrix from a .csv file, finding
the shortest path in the maze, and visualizing the results.

The matrix is represented as a list of rows, where each row is a list of
integers and strings. The integers represent the obstacles (1) and the
free spaces (0), while the strings represent the start ('S') and the
finish ('F') positions.
"""
STEPS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def read_file(filename: str) -> list[list[int | str]]:
    """
    Reads the matrix from a .csv file and returns a list of rows.
    Converts '1' and '0' to integers, keeps 'S' and 'F' as strings, 
    and returns 'incorrect matrix' for invalid inputs.

    Args:
        file_name (str): .csv file with the matrix.

    Returns:
        list[list[int | str]]: list of rows or 'Incorrect matrix' if invalid inputs are found.

    >>> with open("matrix1.csv", "w", encoding="utf-8") as f:
    ...    _ = f.write("0,1,0,0,0\\n0,1,0,1,0\\n0,0,0,1,F\\nS,1,1,1,0\\n0,0,0,0,0\\n")
    >>> read_file('matrix1.csv')
    [[0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 'F'], ['S', 1, 1, 1, 0], [0, 0, 0, 0, 0]]
    >>> with open("matrix2.csv", "w", encoding="utf-8") as f:
    ...    _ = f.write("0,S,0,0\\n1,1,0,F\\n0,S,0,0\\n")
    >>> read_file('matrix2.csv')
    'Incorrect maze'
    """
    matrix=[]
    found_s_f = False
    count_s_f=0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            row=line.strip().split(',')
            processed_row = []
            for part in row:
                if part == '1':
                    processed_row.append(1)
                elif part == '0':
                    processed_row.append(0)
                elif part in {'S', 'F'}:
                    count_s_f+=1
                    processed_row.append(part)
                else:
                    return 'Incorrect maze'
            if 'S' in processed_row and 'F' in processed_row:
                found_s_f = True
            matrix.append(processed_row)
    if not found_s_f and count_s_f!=2:
        return 'Incorrect maze'
    return matrix


def get_neighbors(matrix: list[list[int | str]],
                  row: int, column: int) -> list[tuple[int, int]]:
    """
    Get the neighbors of the given row and column in the matrix.

    Returns:
        list[tuple[int, int]]: The neighbors of the given row and column.

    >>> get_neighbors([[1, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], \
[1, 1, 1, 1, 1], [1, 0, 0, 0, 1]], 1, 1)
    [(2, 1), (1, 2)]
    """
    result_lst = []
    for y, x in STEPS:
        y += row
        x += column
        if is_valid(matrix, y, x) and (matrix[y][x] in [0, 'F']):
            result_lst.append((y, x))
    return result_lst


def is_valid(matrix: list[list[int | str]], row: int, column: int) -> bool:
    """
    Check if the given row and column are valid in the matrix.

    Returns:
        bool: True if the row and column are valid, False otherwise.

    >>> is_valid([[1, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], \
[1, 1, 1, 1, 1], [1, 0, 0, 0, 1]], 1, 1)
    True
    """
    return 0 <= row < len(matrix) and 0 <= column < len(matrix[0])


def find_start(matrix: list[list[int | str]]) -> tuple[int, int]:
    """
    Find the start position in the maze.

    Args:
        matrix (list[list[int | str]]): The maze represented as a matrix.

    Returns:
        tuple[int, int]: The start position.

    >>> maze = [
    ...     [ 0 , 1 , 0 , 0 , 0 ],
    ...     [ 0 , 1 , 0 , 1 , 0 ],
    ...     [ 0 , 0 , 0 , 1 ,'F'],
    ...     ['S', 1 , 1 , 1 , 0 ],
    ...     [ 0 , 0 , 0 , 0 , 0 ],
    ... ]
    >>> find_start(maze)
    (3, 0)
    """
    n = len(matrix)
    for i in range(n):
        m = len(matrix[i])
        for j in range(m):
            if matrix[i][j] == 'S':
                return (i, j)
    return None


def get_shortest_path(matrix: list[tuple[int, int]],
                      start: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Марта, Арсеній

    find_shortest_path(matrix: list[list[int]], start: tuple[int, int]) -> list[tuple[int, int]]

    Finds the shortest path in a maze represented as a grid, where:
    - 0 represents a passable cell (path),
    - 1 represents a wall (impassable cell),
    - 'S' is the starting point, and
    - 'F' is the ending point.

    :param: matrix, list[list[int]]
        A 2D list representing the maze grid. Each element is either 0, 1, 'S', 'F'.
        
    :param: start, tuple[int, int]
        A tuple representing the (row, column) indices of the starting position in the maze.

    :return: list[tuple[int, int]]
        A list of tuples representing the shortest path from 'S' to 'F'.
        Each tuple is a pair of indices (row, column) representing a step in the path.
        Returns -1 if no path exists.

    Example:
    --------
    >>> maze = [
    ...     [ 0 , 1 , 0 , 0 , 0 ],
    ...     [ 0 , 1 , 0 , 1 , 0 ],
    ...     [ 0 , 0 , 0 , 1 ,'F'],
    ...     ['S', 1 , 1 , 1 , 0 ],
    ...     [ 0 , 0 , 0 , 0 , 0 ],
    ... ]
    >>> start = (3, 0)
    >>> get_shortest_path(maze, start)
    [(3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4)]
    >>> len(get_shortest_path(maze, start)) == 8
    True
    """
    def recurtione(queue_, visited):
        node = queue_.pop()
        neighbors = get_neighbors(matrix, node[0], node[1])

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            queue_.append(neighbor)
            visited.add(neighbor)
            if matrix[neighbor[0]][neighbor[1]] == 'F':
                return [neighbor]
            return [neighbor] + recurtione(queue_, visited)

    queue = [start]
    visited = set()
    path = [start] + recurtione(queue, visited)
    return path

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
