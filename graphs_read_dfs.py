"""
Lab 2 template
"""

def read_incidence_matrix(filename: str) -> list[list]:
    """
    Read the incidence matrix from a file
    
    :param str filename: path to file
    :returns list[list]: the incidence matrix of a given graph

    >>> read_incidence_matrix('input.dot')
    [[1, 1, -1, 0, -1, 0], [-1, 0, 1, 1, 0, -1], [0, -1, 0, -1, 1, 1]]
    """

    edges = []
    vertices = set()
    directed = False

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if 'digraph' in line:
                directed = True
            if '->' in line:
                parts = line.split('->')
                edges.append((int(parts[0].strip()), int(parts[1].strip(';'))))
                vertices.update([int(parts[0].strip()), int(parts[1].strip(';'))])

    vertices = sorted(vertices)
    vertices_count, edge_count = len(vertices), len(edges)

    matrix = [[0] * edge_count for _ in range(vertices_count)]

    for edge_index, (x, y) in enumerate(edges):
        if directed:
            if x == y:
                matrix[x][edge_index] = 2
            else:
                matrix[x][edge_index] = 1
                matrix[y][edge_index] = -1
        else:
            matrix[x][edge_index] = 1
            matrix[y][edge_index] = 1

    return matrix


def read_adjacency_matrix(filename: str) -> list[list]:
    """
    :param str filename: path to file
    :returns list[list]: the adjacency matrix of a given graph

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
        matrix[y][x] = 1

    return matrix


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict: the adjacency dict of a given graph

    >>> read_adjacency_dict('input.dot')
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """

    dictionary = {}
    directed = False

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if 'digraph' in line:
                directed = True
            if '->' in line:
                parts = line.split('->')
                x, y = int(parts[0]), int(parts[1].strip(';'))

                if x not in dictionary:
                    dictionary[x] = []
                dictionary[x].append(y)

                if y not in dictionary:
                    dictionary[y] = []
                    if not directed:
                        dictionary[y].append(x)

    return dictionary


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
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


def iterative_adjacency_matrix_dfs(graph: list[list], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
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
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
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


def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """

    visited = [False] * len(graph)
    path = []

    def explore(current):
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
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    pass


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    pass


def recursive_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> recursive_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    pass


def recursive_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> recursive_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    pass


def adjacency_matrix_radius(graph: list[list]) -> int:
    """
    :param list[list] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    2
    """
    pass


def adjacency_dict_radius(graph: dict[int: list[int]]) -> int:
    """
    :param dict graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1]})
    2
    """
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
