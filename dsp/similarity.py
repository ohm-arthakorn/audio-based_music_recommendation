import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculates Cosine Similarity with the same length vectors

        Formula: vec_a · vec_b / (||vec_a|| * ||vec_b||)

        Args:
            vec_a: first input vector
            vec_b: second input vector
        
        Returns:
            The cosine between `vec_a` and `vec_b`
            If `vec_a` or `vec_b` has norm equal 0, returns cosine = 0
        
        Raises:
            ValueError: If `vec_a` and `vec_b` do not have the same length.
    """

    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector lengths must be equal: got {len(vec_a)} and {len(vec_b)}."
        )

    norm_vec_a = np.sqrt(sum(a ** 2 for a in vec_a))        
    norm_vec_b = np.sqrt(sum(b ** 2 for b in vec_b))     
    cosine = 0.0

    # Check 0 vector from norm of each vectors
    if norm_vec_a == 0 or norm_vec_b == 0:
        return cosine

    # Pair corresponding elements from both vectors and sum thier products.
    dot_product = sum(a*b for a,b in zip(vec_a, vec_b)) 

    cosine = dot_product/(norm_vec_a*norm_vec_b) 

    return cosine 

if __name__ == "__main__":
    print("Debug and testing.")
    print(f"Cosine Similarity no.1 : {cosine_similarity([1,0],[0,1])}")
    print(f"Cosine Similarity no.2 : {cosine_similarity([1,2,3],[2,4,6])}")
    print(f"Cosine Similarity no.3 : {cosine_similarity([1,0],[-1,0])}")
    print(f"Cosine Similarity no.4 : {cosine_similarity([1,1],[1,0])}")
    print(f"Cosine Similarity no.5 : {cosine_similarity([0,0],[1,2])}")
    