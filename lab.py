"""
Discrete Mathematics Lab on Graph Theory, specifically on
Depth First Search (DFS) and Breadth First Search (BFS) algorithms.
"""


def read_incidence_matrix(filename: str) -> list[list]:
    """
    Read the incidence matrix from a file.
    
    Args:
        filename (str): Path to file.

    Returns:
        list[list]: The incidence matrix of a given graph.

    >>> read_incidence_matrix('input.dot')
    [[1, 1, -1, 0, -1, 0], [-1, 0, 1, 1, 0, -1], [0, -1, 0, -1, 1, 1]]
    """
    edges = []
    vertices = set()

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if '->' in line:
                parts = line.split('->')
                edges.append((int(parts[0].strip()), int(parts[1].strip(';'))))
                vertices.update([int(parts[0].strip()), int(parts[1].strip(';'))])

    vertices = sorted(vertices)
    vertices_count, edge_count = len(vertices), len(edges)

    matrix = [[0] * edge_count for _ in range(vertices_count)]

    for edge_index, (x, y) in enumerate(edges):
        if x == y:
            matrix[x][edge_index] = 2
        else:
            matrix[x][edge_index] = 1
            matrix[y][edge_index] = -1

    return matrix


def read_adjacency_matrix(filename: str) -> list[list]:
    """
    Read the adjacency matrix from a file.
    
    Args:
        filename (str): Path to file.

    Returns:
        list[list]: The adjacency matrix of a given graph.

    >>> read_adjacency_matrix('input.dot')
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    """
    edges = []
    nodes = set()

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if '->' in line:
                parts = line.split('->')
                edges.append((int(parts[0]), int(parts[1].strip(';'))))

    for edge in edges:
        nodes.update(edge)
    size = max(nodes) + 1

    matrix = [[0] * size for _ in range(size)]

    for x, y in edges:
        matrix[x][y] = 1

    return matrix


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    Read the adjacency dictionary from a file.

    Args:
        filename (str): Path to file.

    Returns:
        dict: The adjacency dictionary of a given graph.

    >>> read_adjacency_dict('input.dot')
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    dictionary = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if '->' in line:
                parts = line.split('->')
                x, y = int(parts[0]), int(parts[1].strip(';'))

                if x not in dictionary:
                    dictionary[x] = []
                dictionary[x].append(y)

                if y not in dictionary:
                    dictionary[y] = []

    return dictionary


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Traverse the graph using the iterative depth-first search algorithm.
    
    Args:
        graph (dict): The adjacency dict of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The DFS traversal of the graph.

    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    path = []
    stack = [start]
    if not graph:
        return []

    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue

        visited.add(vertex)
        path.append(vertex)

        for neighbor in reversed(graph[vertex]):
            if neighbor not in visited:
                stack.append(neighbor)

    return path


def iterative_adjacency_matrix_dfs(graph: list[list], start: int) -> list[int]:
    """
    Traverse the graph using the iterative depth-first search algorithm.

    Args:
        graph (list): The adjacency matrix of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The DFS traversal of the graph.

    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    num_vertices = len(graph)
    visited = [False] * num_vertices
    path = []
    explore = [0] * num_vertices
    search = [start]

    while search:
        current_vertex = search[-1]

        if not visited[current_vertex]:
            visited[current_vertex] = True
            path.append(current_vertex)

        next_vertex = None
        for j in range(explore[current_vertex], num_vertices):
            if graph[current_vertex][j] == 1 and not visited[j]:
                next_vertex = j
                explore[current_vertex] = j + 1
                break

        if next_vertex is None:
            search.pop()
        else:
            search.append(next_vertex)

    return path


def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Traverse the graph using the recursive depth-first search algorithm.
    
    Args:
        graph (dict): The adjacency dict of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The DFS traversal of the graph.

    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    result = []

    def dfs(v: int):
        visited.add(v)
        result.append(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)

    return result


def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    Traverse the graph using the recursive depth-first search algorithm.

    Args:
        graph (list): The adjacency matrix of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The DFS traversal of the graph.

    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = [False] * len(graph)
    path = []

    def explore(current: int):
        """
        To explore vertices connected to the current vertex.
        """

        visited[current] = True
        path.append(current)

        for next_vertex in range(len(graph)):
            if graph[current][next_vertex] == 1 and not visited[next_vertex]:
                explore(next_vertex)

    explore(start)

    return path


def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Perform a breadth-first search on the graph using an iterative approach.

    Args:
        graph (dict): The adjacency dict of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The BFS traversal of the graph.

    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = [start]
    path = []

    while queue:   
        vertex = queue.pop(0)

        if vertex not in visited:
            visited.add(vertex)
            path.append(vertex)

            queue.extend([neighbor for neighbor in graph.get(vertex, [])
                          if neighbor not in visited])

    return path


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) -> list[int]:
    """
    Traverse the graph using the iterative breadth-first search algorithm.

    Args:
        graph (list): The adjacency matrix of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The BFS traversal of the graph.

    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = [start]
    path = []

    while queue:
        vertex = queue.pop(0)

        if vertex not in visited:
            visited.add(vertex)
            path.append(vertex)

            for neighbor, connected in enumerate(graph[vertex]):
                if connected and neighbor not in visited:
                    queue.append(neighbor)

    return path


def recursive_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Traverse the graph using the recursive breadth-first search algorithm.

    Args:
        graph (dict): The adjacency dict of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The BFS traversal of the graph.

    >>> recursive_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    def bfs_helper(queue: list[int], visited: set[int]) -> list[int]:
        """
        Helper function to implement breadth-first search recursively.
        """
        if not queue:
            return []

        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)

            queue.extend([neighbor for neighbor in graph.get(vertex, [])
                          if neighbor not in visited and neighbor not in queue])

        return [vertex] + bfs_helper(queue, visited)

    return bfs_helper([start], set())


def recursive_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
    """
    Traverse the graph using the recursive breadth-first search algorithm.

    Args:
        graph (list): The adjacency matrix of a given graph.
        start (int): The start vertex of the search.

    Returns:
        list[int]: The BFS traversal of the graph.

    >>> recursive_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    def bfs_helper(queue: list[int], visited: set[int]) -> list[int]:
        """
        Helper function to implement breadth-first search recursively.
        """
        if not queue:
            return []

        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)

            queue.extend([neighbor for neighbor, connected in enumerate(graph[vertex])
                          if connected and neighbor not in visited and neighbor not in queue])

        return [vertex] + bfs_helper(queue, visited)

    return bfs_helper([start], set())


def adjacency_matrix_radius(graph: list[list]) -> int:
    """
    Calculate the radius of the graph.
    
    Args:
        graph (list): The adjacency matrix of a given graph.
    
    Returns:
        int: The radius of the graph.

    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    1
    """
    def get_distances(start: int) -> list[int]:
        """Find the distances of the vertex using BFS."""
        distances = [-1] * len(graph)
        distances[start] = 0
        queue = [start]

        while queue:
            vertex = queue.pop(0)

            for neighbor, connected in enumerate(graph[vertex]):
                if connected and distances[neighbor] == -1:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)

        return distances

    eccentricities = [max(get_distances(v)) for v in range(len(graph))]

    return min(eccentricities)


def adjacency_dict_radius(graph: dict[int, list[int]]) -> int:
    """
    Calculate the radius of the graph.
    
    Args:
        graph (dict): The adjacency dictionary of a given graph.
    
    Returns:
        int: The radius of the graph.

    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1]})
    1
    """
    def get_distances(start: int) -> list[int]:
        """Find the distances of the vertex using BFS."""
        distances = {}
        distances[start] = 0
        queue = [start]

        while queue:
            vertex = queue.pop(0)

            for neighbor in graph.get(vertex, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)

        return distances

    eccentricities = [max(get_distances(v).values()) for v in graph]

    return min(eccentricities)


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
