import numpy as np

class AntColony:
    def __init__(self, size, density, start_node, end_node, alpha=1, beta=2, evap_rate=0.5, generations=100):
        self.size = size
        self.density = density
        self.start_node = start_node
        self.end_node = end_node
        self.alpha = alpha
        self.beta = beta
        self.evap_rate = evap_rate
        self.generations = generations
        
        self.graph = self.generate_graph(size, density)
        self.pheromones = np.ones((size, size))
        
    def generate_graph(self, size, density):
        graph = np.full((size, size), np.inf)
        for i in range(size):
            for j in range(i, size):
                if np.random.random() < density and i != j:
                    weight = np.random.randint(1, 20)
                    graph[i][j] = weight
                    graph[j][i] = weight
        return graph
        
    def choose_next_node(self, current_node, visited):
        unvisited = np.where(visited == False)[0]
        graph = self.graph[current_node, unvisited]
        pheromones = self.pheromones[current_node, unvisited]
        weights = np.power(pheromones, self.alpha) * np.power(1 / graph, self.beta)
        weights /= np.sum(weights)
        next_node = np.random.choice(unvisited, p=weights)
        return next_node
        
    def traverse(self):
        current_node = self.start_node
        path = [current_node]
        visited = np.zeros(self.size, dtype=bool)
        visited[current_node] = True
        cost = 0
        
        while current_node != self.end_node:
            next_node = self.choose_next_node(current_node, visited)
            visited[next_node] = True
            cost += self.graph[current_node][next_node]
            path.append(next_node)
            current_node = next_node
            
        return path, cost
        
    def release_pheromones(self):
        paths, costs = [], []
        for i in range(self.size):
            path, cost = self.traverse()
            paths.append(path)
            costs.append(cost)
        return paths, costs
        
    def update_pheromones(self, paths, costs):
        self.pheromones *= (1 - self.evap_rate)
        for path, cost in zip(paths, costs):
            for i in range(len(path) - 1):
                current_node, next_node = path[i], path[i+1]
                self.pheromones[current_node][next_node] += 1 / cost
        
    def run(self):
        for i in range(self.generations):
            paths, costs = self.release_pheromones()
            self.update_pheromones(paths, costs)
            best_path, best_cost = paths[np.argmin(costs)], np.min(costs)
            print(f"Generation {i+1}, Best Path: {best_path}, Cost: {best_cost}")

# Example usage
ac = AntColony(size=10, density=0.5, start_node=0, end_node=7)
ac.run()
