class GraphNode:
    def __init__(self, value, cost: int):
        self.value = value
        self.cost = cost
        self.neighbors = set()


def dfs(node: GraphNode, target):
    stack = [node]
    visited = set()
    while stack:
        node = stack.pop()
        if node.value == target:
            print("Node:", node.value)
            break
        visited.add(node)
        for child in node.neighbors:
            if child not in visited and child not in stack:
                stack.append(child)
    else:
        print("Not Found")


def bfs(node: GraphNode, target):
    queue = [node]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node.value == target:
            print("Found:", node.value)
            break
        visited.add(node)
        for child in node.neighbors:
            if child not in visited and child not in queue:
                queue.append(child)
    else:
        print("Not Found")


def hill_climbing(start: GraphNode, goal: GraphNode):
    current = start
    visited = set()

    while True:
        print(current.value)

        if current is goal:
            print("Found Goal")
            return

        visited.add(current)

        best = min(
            (n for n in current.neighbors if n not in visited),
            key=lambda n: n.cost,
            default=None
        )

        if best is None or best.cost >= current.cost:
            print("local minimum")
            return

        current = best


if __name__ == '__main__':
    heuristics = {
        'A': 15,
        'B': 13,
        'C': 9,
        'D': 12,
        'E': 7,
        'F': 8,
        'G': 0
    }

    nodes = {k: GraphNode(k, v) for k, v in heuristics.items()}

    nodes['A'].neighbors = {nodes['B'], nodes['C'], nodes['D']}
    nodes['B'].neighbors = {nodes['A']}
    nodes['C'].neighbors = {nodes['E'], nodes['F']}
    nodes['D'].neighbors = {nodes['A']}
    nodes['E'].neighbors = {nodes['C'], nodes['G']}
    nodes['F'].neighbors = {nodes['C'], nodes['G']}
    nodes['G'].neighbors = set()

    print("Hill Climbing")
    hill_climbing(nodes['A'], nodes['G'])
