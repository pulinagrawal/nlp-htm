from collections import defaultdict

def topological_sort_kahn(graph):
    indegree = defaultdict(int) # Track indegrees
    queue = [] #Initialize queue

    # Calculate indegrees
    for node in graph:
        for neighbour in graph[node]:
            indegree[neighbour] += 1

    # Add nodes with 0 indegree to queue
    for node in graph:
        if indegree[node] == 0:
            queue.append(node)

    topological_order = []

    # Process until queue is empty
    while queue:

        # Remove node from queue and add to topological order
        node = queue.pop(0)
        topological_order.append(node)

        # Reduce indegree for its neighbors
        for neighbour in graph[node]:
            indegree[neighbour] -= 1

            # Add new 0 indegree nodes to queue
            if indegree[neighbour] == 0:
                queue.append(neighbour)

    if len(topological_order) != len(graph):
        print("Cycle exists")

    return topological_order