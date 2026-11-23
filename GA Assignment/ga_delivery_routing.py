import numpy as np

# Symmetric distance matrix (in km) for Depot A and Locations B-G
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCE_MATRIX = np.array([
    [0, 4, 8, 6, 7, 9, 5],  # A (Depot)
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
])


def calculate_route_distance(chromosome):
    """Calculate total round-trip distance starting and ending at Depot A (index 0)."""
    full_route = [0] + list(chromosome) + [0]
    total_distance = 0
    for i in range(len(full_route) - 1):
        total_distance += DISTANCE_MATRIX[full_route[i], full_route[i + 1]]
    return total_distance


def order_crossover(parent1, parent2, rng):
    """Order Crossover (OX1) preserves valid permutations with no duplicates."""
    size = len(parent1)
    start, end = sorted(rng.choice(range(size), size=2, replace=False))
    child = [-1] * size
    child[start : end + 1] = parent1[start : end + 1]

    remaining_genes = [gene for gene in parent2 if gene not in child]
    fill_idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining_genes[fill_idx]
            fill_idx += 1
    return np.array(child)


def swap_mutation(chromosome, mutation_rate, rng):
    """Swap two random delivery locations in the route with probability mutation_rate."""
    mutated = chromosome.copy()
    if rng.random() < mutation_rate:
        idx1, idx2 = rng.choice(range(len(mutated)), size=2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated


def run_delivery_ga(seed=42, pop_size=30, generations=60, mutation_rate=0.20, n_elite=2):
    """Run Genetic Algorithm to find the shortest delivery route."""
    rng = np.random.default_rng(seed)
    delivery_indices = np.array([1, 2, 3, 4, 5, 6])  # Represents B, C, D, E, F, G

    # 1. Initial population of random valid route permutations
    population = np.array([rng.permutation(delivery_indices) for _ in range(pop_size)])

    best_chromosome = None
    best_distance = float("inf")

    for _ in range(generations):
        # 2. Evaluate fitness (total distance; lower is better)
        distances = np.array([calculate_route_distance(ind) for ind in population])

        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_distance = distances[min_idx]
            best_chromosome = population[min_idx].copy()

        # 3. Elitism: preserve top routes unchanged
        elite_indices = np.argsort(distances)[:n_elite]
        elites = population[elite_indices]

        # 4. Tournament selection (size 3), Order Crossover, and Swap Mutation
        children = []
        while len(children) < pop_size - n_elite:
            t1 = rng.choice(pop_size, size=3, replace=False)
            parent1 = population[t1[np.argmin(distances[t1])]]

            t2 = rng.choice(pop_size, size=3, replace=False)
            parent2 = population[t2[np.argmin(distances[t2])]]

            child = order_crossover(parent1, parent2, rng)
            child = swap_mutation(child, mutation_rate, rng)
            children.append(child)

        population = np.vstack([elites, np.array(children)])

    best_route_names = ["A"] + [LOCATIONS[i] for i in best_chromosome] + ["A"]
    return best_route_names, best_distance


if __name__ == "__main__":
    best_route, total_km = run_delivery_ga(seed=42, pop_size=30, generations=60)
    print("=== Genetic Algorithm Delivery Route Optimisation ===")
    print("Best Route:     ", " -> ".join(best_route))
    print(f"Total Distance:  {total_km} km")