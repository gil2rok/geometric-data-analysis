# Geometric Data Analysis :doughnut:

**The work of Gilad Turok for [COMS 4995] Geometric Data Analysis with Professor Andrew Blumberg in Spring of 2023 at Columbia University :pencil2: :triangular_ruler: :bar_chart:.**

(Unfortunately, there are no good emojis for topology or manifolds. Until this crime is remedied, I will be using a doughnut (torus) as above.)

## **HW1** :round_pushpin: ##
**Implement and compare clustering algorithms:**
- $k$-means
- $k$-medians
- single-linkage clustering
- spectral clustering with graphs constructed by $k$ nearest-neighbors and the Gaussian/RBF kernel.

## **HW2** :round_pushpin: ##
**Implement and compare manifold dimensionality reduction algorithms:**
- IsoMap
- Multi-Dimensional Scaling
- Locally Linear Embedding
- Laplacian Eigenmaps

**Prove multiple properties of manifolds:**
- enumerate a space that is not a manifold
- show that for any point $m \in \mathcal{M}$ on a manifold, a chart $\theta$ can be constructed such that $\theta(m)=0$
- explicate coordinate patches and transition functions for the circle using charts defined by the angle and projections to the axes
- prove that solving the MDS matrix factorization problem $A=U \Lambda U^T$ for $m << n$ points approximates the true solution.

**Helpful Resources:**
- [Github tutorial on manifold learning](https://github.com/drewwilimitis/Manifold-Learning)
- [paper summarizing manifold learning algorithms](https://www.cs.columbia.edu/~verma/classes/ml/ref/lec8_cayton_manifolds.pdf).

## **HW3:** :round_pushpin: ##
**Prove multiple properties of tree metrics $d_T$, Hausdorff metrics $d_H$ and Gromov-Hausdorff metrics $d_{GH}$:**

- Metric tree spaces $\mathcal{T}$ have nonpositive curvature
- The Hausdorff $d_H$ and Gromov-Hausdorff metrics $d_{GH}$ are formal metrics
- The Gromov-Hausdorff metric $d_{GH}$ is bounded by the diameter of a space $diam(X)$
- The Gromov Hausdorff metric $d_{GH}$ is bounded by $\epsilon$ for spaces $X,Y$ where $Y \subset X$ is an $\epsilon$-net

**Implement algorithms to:**    
- **Estimate the intrinsic dimension** by approximating the tangent plane for Gaussians and hypercubes
- Emperically prove the **Johnson-Lindenstrauss Lemma** with numerical experiments
- Find the **centroid in Wasserstein space** to "average" distrubutions in $\mathbb{R}^2$ of MNIST digits.
- On tooth and bone datasets, use an **approximation of the Gromov-Hausdorff metric $d_{GH}$ to cluster** the data