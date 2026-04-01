from typing import List
import numpy as np
class SmallWorldLattice:
    """
    Watts-Strogatz small-world network for social influence.
    
    Creates a network where nodes are connected in a ring lattice, then
    edges are randomly rewired to introduce short-cuts while maintaining
    high clustering.
    """
    
    def __init__(self, n: int, k: int, p_rewire: float, rng: np.random.Generator):
        """
        Initialize and create the small-world network.
        
        Args:
            n: Number of nodes (voters)
            k: Number of neighbors per node (must be even)
            p_rewire: Probability of rewiring each edge
            rng: Random number generator
            
        Raises:
            ValueError: If k is not even
        """
        self.n = n
        self.k = k
        self.p_rewire = p_rewire
        self.rng = rng
        
        if k % 2 != 0:
            raise ValueError("k_neighbors must be even for the small-world construction.")
        
        self.neighbors = self._build()
    
    def _build(self) -> List[np.ndarray]:
        """Build the small-world network."""
        neighbors = [set() for _ in range(self.n)]
        half = self.k // 2
        
        # Create regular ring lattice
        for i in range(self.n):
            for d in range(1, half + 1):
                j = (i + d) % self.n
                neighbors[i].add(j)
                neighbors[j].add(i)
        
        # Rewire edges with probability p_rewire
        for i in range(self.n):
            for d in range(1, half + 1):
                j = (i + d) % self.n
                if self.rng.random() < self.p_rewire:
                    possible = list(set(range(self.n)) - {i} - neighbors[i])
                    if possible:
                        new_j = int(self.rng.choice(possible))
                        neighbors[i].discard(j)
                        neighbors[j].discard(i)
                        neighbors[i].add(new_j)
                        neighbors[new_j].add(i)
        
        return [np.array(sorted(list(s)), dtype=np.int32) for s in neighbors]
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Allow indexing to get neighbors of a node."""
        return self.neighbors[idx]
    
    def __len__(self) -> int:
        """Return number of nodes."""
        return self.n
    
    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get neighbors of a specific node."""
        return self.neighbors[idx]
    
    def clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient (optional utility method)."""
        # Implementation if needed
        pass
    
    def average_path_length(self) -> float:
        """Calculate average shortest path length (optional utility method)."""
        # Implementation if needed
        pass
