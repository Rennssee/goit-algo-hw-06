import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq
import random


def dfs(graph, start, end, path=None):
    if path is None:
        path = []
    path += [start]

    if start == end:
        return path

    for neighbor in graph.neighbors(start):
        if neighbor not in path:
            new_path = dfs(graph, neighbor, end, path)
            if new_path:
                return new_path

    return None


def bfs(graph, start, end):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == end:
            return path

        if node not in visited:
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

            visited.add(node)

    return None


def dijkstra(graph, start):
    distances = {node: float("infinity") for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor].get("weight", 1)
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


# Створення графа для соціальної мережі з випадковими вагами
social_network = nx.Graph()
people = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eva",
    "Frank",
    "Grace",
    "Hank",
    "Ivy",
    "Jack",
    "Kelly",
    "Leo",
    "Mia",
    "Nathan",
    "Olivia",
    "Peter",
    "Quinn",
    "Rachel",
    "Steve",
    "Tina",
    "Ulysses",
    "Violet",
    "Walter",
    "Xena",
    "Yuri",
    "Zoe",
    "Adam",
    "Bella",
    "Chris",
]
social_network.add_nodes_from(people)

relationships = [
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "Charlie"),
    ("David", "Eva"),
    ("Eva", "Frank"),
    ("Grace", "Hank"),
    ("Hank", "Ivy"),
    ("Ivy", "Jack"),
    ("Jack", "Kelly"),
    ("Kelly", "Leo"),
    ("Leo", "Mia"),
    ("Mia", "Nathan"),
    ("Nathan", "Olivia"),
    ("Olivia", "Peter"),
    ("Peter", "Quinn"),
    ("Quinn", "Rachel"),
    ("Rachel", "Steve"),
    ("Steve", "Tina"),
    ("Tina", "Ulysses"),
    ("Ulysses", "Violet"),
    ("Violet", "Walter"),
    ("Walter", "Xena"),
    ("Xena", "Yuri"),
    ("Yuri", "Zoe"),
    ("Adam", "Bella"),
    ("Bella", "Chris"),
    ("Alice", "Tina"),
    ("Frank", "Leo"),
    ("Adam", "Peter"),
    ("Mia", "Violet"),
]

for edge in relationships:
    weight = {"weight": random.randint(1, 15)}
    social_network.add_edge(edge[0], edge[1], **weight)

# Виведення основних характеристик графа
num_nodes = social_network.number_of_nodes()
num_edges = social_network.number_of_edges()
print("Кількість вершин:", num_nodes)
print("Кількість ребер:", num_edges)

print("\nСтупінь вершин:")
for person in people:
    print(f"{person}: {social_network.degree(person)}")

# Візуалізація графа з вагами, DFS та BFS шляхами
pos = nx.spring_layout(social_network)
nx.draw(
    social_network,
    pos,
    with_labels=True,
    font_weight="bold",
    node_color="lightblue",
    edge_color="gray",
)

# Відмалювання ребер з вагами
edge_labels = nx.get_edge_attributes(social_network, "weight")
nx.draw_networkx_edge_labels(social_network, pos, edge_labels=edge_labels)

# Задання початкової та кінцевої вершин для DFS та BFS
start_node = "Alice"
end_node = "Chris"

# Відмалювання DFS та BFS шляхів
dfs_path = dfs(social_network, start_node, end_node)
bfs_path = bfs(social_network, start_node, end_node)

print("\nDFS шлях:", dfs_path)
print("BFS шлях:", bfs_path)

plt.show()

# Запуск алгоритму Дейкстри для кожної вершини
shortest_paths = {}
for node in social_network.nodes:
    distances, predecessors = dijkstra(social_network, node)
    shortest_paths[node] = {"distances": distances, "predecessors": predecessors}

# Виведення результатів
for node in social_network.nodes:
    print(f"\nНайкоротші шляхи з вершини {node}:")

    for other_node in social_network.nodes:
        if node != other_node:
            path = [other_node]
            current_node = other_node
            while shortest_paths[node]["predecessors"][current_node] is not None:
                current_node = shortest_paths[node]["predecessors"][current_node]
                path.insert(0, current_node)

            distance = shortest_paths[node]["distances"][other_node]
            print(f"{node} -> {other_node}: {path}, Відстань: {distance}")
