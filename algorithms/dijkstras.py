# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# dijkstra's implementation for directed graph

class DijkstrasGraph:

    def __init__(self, graph):
        self.graph = graph
        self.nodes = len(graph[0])

    def dijkstra(self):
        pass

    def print_solution(self):
        pass


# test code here
if __name__ == "__main__":
    graph = [[0, 5, 0, 6, 0, 0],
             [0, 0, 3, 1, 0, 8],
             [0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 3, 6],
             [0, 0, 0, 0, 0, 2],
             [0, 0, 0, 0, 0, 0]]

    g = DijkstrasGraph(graph=graph)
    g.dijkstra()
