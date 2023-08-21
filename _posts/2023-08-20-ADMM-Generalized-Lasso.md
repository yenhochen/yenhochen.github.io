---
layout: blogpost
title: ADMM for Generalized Lasso Inference 
date: 2023-08-21 00:53:00-0400
description: L1 penalized inference is central to the field of compressed sensing. I derive an ADMM solver for the generalized Lasso problem and discuss three important special cases; standard lasso, variable fusion, and fused lasso. Further, I provide python code and demonstrate it on a simulated dataset. 
categories: 
giscus_comments: false
related_posts: false
scripts:
    # - posts/2023-07-01-transposedConv/transposedConv_tool.js 
style: 
# posts/2023-07-01-transposedConv/transposedConv_tool.css

---

---

Sparsity plays an important role in machine learning, statistics, and signal processing. In the basic set up, we are given a noisy observation $$y\in\mathbb{R}^m$$ and a dictionary $$D \in \mathbb{R}^{m\times n}$$. The goal is to estimate the weights $$w \in \mathbb{R}^n$$ that produce the observation under the assumption that most of its entries have a value of zero. A popular approach to obtaining sparse solutions is through the class of $$\ell_1$$ penalized regression problems. Mathematically, these are written as

$$
\DeclareMathOperator*{\argmin}{arg\,min}
$$

$$
\begin{equation}
\label{eq:general-lasso}
w^* = \argmin_{w} \frac{1}{2} \| Dw - y \|_2^2 + \lambda \| F w \|_1
\end{equation}
$$


where $$F \in \mathbb{R}^{k \times n}$$ encodes $$K$$ distinct $$\ell_1$$-norm penalties, and  $$\lambda \geq 0$$ is a lagrange multiplier that determines the strength of those penalty terms. In statistics, this is also known as the generalized Lasso problem [1]. The main challenge with solving equation $$\ref{eq:general-lasso}$$ lies in the non-differentiability of the $$\ell_1$$-norm. Therefore, we cannot obtain an analytical solution and must resort to iterative solvers. 

<br>

<!-- The main aim of this post is to carefully work through the derivation of an efficient iterative solution for L1 penalized objectives. While existing texts offer the solution, they often omit many important details in the explanation and assume substantial background knowledge. In contrast, I provide a derivation that is accessible to those with only a basic knowledge in matrix calculus and linear algebra. -->

---


<!-- This is achieved by combining the benefits of decomposability from dual ascent with the fast convergence properties of the method of multipliers. -->





### Deriving an ADMM algorithm

An overview of the ADMM framework can be found 
[here]({% post_url 2023-08-19-Alternating Directions Method of Multipliers%}) 
. We can rewrite the generalized Lasso into ADMM form by introducing a variable $$z$$ and a constraint which transforms equation $$\ref{eq:general-lasso}$$ into the following problem

$$
\begin{equation}
\min_w \frac{1}{2} \|Dw-y \|_2^2 + \lambda \| z \|_1 \quad
\textrm{ subject to }
F w = z
\end{equation}
$$

The corresponding scaled Augmented Lagragian is given by

$$
\begin{equation}
\label{eq:L}
L(w, z, u) = \frac{1}{2} \|Dw-y \|_2^2 + \lambda \| z \|_1 + \frac{\rho}{2} \|F w-z +u \|_2^2 - \frac{\rho}{2} \|u \|_2^2
\end{equation}
$$


Note that introducing $$z$$ doesn't change the optimal solution since the additional penalty terms in $$L$$ are minimized when $$w-z=0$$. However, we benefit from separating the $$\ell_1$$ penalty from the reconstruction term. ADMM proceeds with block minimization under $$L$$. This consists of an $$w$$-minimization step, a  $$z$$-minimization step, and a gradient ascent update step on the dual variables $$u$$. 
<!-- However, a major benefit is that the ADMM form enables separability between the smooth and non-smooth components in the objective.  -->
<!-- As a result, we can proceed .  -->




---

##### Updating $$w$$

The standard ADMM update for $$w$$ is given by

$$
w_{k+1} = \argmin_w \hspace{0.2em} L(w, z_{k}, u_k) 
$$

To obtain the RHS, we set the partial derivative of equation $$\ref{eq:L}$$ wrt. $$w$$ to zero.

$$
\nabla_w L = D^\top(Dw-y) + \rho F^\top (F w-z+u) = 0
$$


Now rearrange to collect terms with $$w$$ on the LHS and all other terms on the RHS.


$$
D^\top Dw  + \rho F^\top F w = D^\top y + \rho F^\top z - \rho F^\top u
$$

Factorize and solve for $$w$$.


$$
w^* = (D^\top D  + \rho F^\top F)^{-1}(D^\top y +\rho F^\top (z_k - u_k))
$$

$$w^*$$ minimizes the augmented Lagrangian given our previous estimates of $$z_k$$ and $$u_k$$ and becomes the closed-form update rule for $$w_{k+1}$$.

<!-- $$
\begin{alignat*}{2}
\Rightarrow \quad &D^\top Dw  + \rho F^\top F w = D^\top y + \rho F^\top z - \rho F^\top u\\
% &(D^\top D  + \rho F^\top F) w = D^\top y +\rho F^\top (z - u)\\
\Rightarrow \quad & w = (D^\top D  + \rho F^\top F)^{-1}(D^\top y +\rho F^\top (z_k - u_k))\\
\end{alignat*}
$$ -->


---

##### Updating $$z$$

Next, we find the ADMM update for $$z$$

$$
z_{k+1} = \argmin_z \hspace{0.2em} L(w_{k+1}, z, u_k) 
$$

Similarly, we set the partial derivative of $$L$$ wrt. $$z$$ to zero

$$
\begin{equation}
\label{eq:L_z}
\nabla_z L = \frac{\partial}{\partial z} (\lambda \|z \|_1) - \rho (Fw-z+u) = 0
\end{equation}
$$

Immediately, we encounter the non-differentiable $$\ell_1$$ term. We can proceed by observing two facts.  1) The overall $$\ell_1$$ penalty can be decomposed as a sum of $$\ell_1$$ separate penalties placed on each component of $$z$$,  i.e.
$$\|z\|_1 = \sum_{i=1}^N |z_i|$$. Therefore, we can deal with each component independently. And
2) For each component, there exists only a single discontinuity centered around zero, and that the derivative exists for all other values. Therefore, we can split this problem into two cases $$z=0$$ and $$z\neq 0$$.



<br> 


For $$z \neq 0$$, the derivative is equal to the $${\rm sign} (z)$$ which can be substituted into equation $$\ref{eq:L_z}$$ to get

$$
\lambda {\rm sign}(z) - \rho (Fw-z+u) = 0
$$

Collecting all terms with $$z$$ on the LHS and all others on the RHS, we get

$$
z + \frac{\lambda}{\rho} {\rm sign}(z)= Fw + u 
$$

When $$z < 0$$, then $$Fw + u < -\frac{\lambda}{\rho}$$ and when $$z > 0$$, then $$Fw + u > \frac{\lambda}{\rho}$$. This observation can be compactly written as
$$|Fw + u| > \frac{\lambda}{\rho}$$
and $${\rm sign}(z) = {\rm sign}(Fw+u)$$ to get the update

$$
z^* = Fw + u - \frac{\lambda}{\rho} {\rm sign}(Fw + u)
$$

<br>

When $$z = 0$$, we must deal with the discontinuity using subgradients, which is given by $$\partial \| z\|_1 =[-1,1]$$. The optimality condition is zero exists within the subgradient. Substituting this into equation $$\ref{eq:L_z}$$, 

$$
0 \in \lambda[-1,1] - \rho (Fw-z+u) 
$$

which can be rearranged to get the form

$$
Fw+u  \in \left[-\frac{\lambda}{\rho},\frac{\lambda}{\rho}\right] 
$$

In words, $$z=0$$ is an optimal solution only when $$Fw+u$$ is within the region $$\left[-\frac{\lambda}{\rho},\frac{\lambda}{\rho}\right]$$. Combining the results above for $$z \neq 0$$ with $$z = 0$$, we get the following updates

$$
z_{k+1} = 
\begin{cases}
   0 &, \quad|Fw_{k+1}+u_k| \leq \frac{\lambda}{\rho} \\
   Fw_{k+1} + u_k - \frac{\lambda}{\rho} {\rm sign}(w_{k+1}) &, \quad |Fw_{k+1}+u_k| > \frac{\lambda}{\rho}
\end{cases}
$$

More compactly this can be written using the soft-threshold operator $$\mathcal{S}$$. The update equation for $$z$$ is given by

$$
\begin{align*}
z_{k+1} &= {\rm sign}(Fw_{k+1} + u_k) \max  \left( \left| Fw_{k+1} + u_k \right| - \frac{\lambda}{\rho}, 0 \right) \\
&= \mathcal{S}_{\lambda/\rho} \left(Fw_{k+1} + u_k  \right)
\end{align*}
$$



 <!-- Therefore, there are two different cases we encounter when computing the gradient of the -->

---

##### Updating $$u$$

Given the updated $$w_{k+1}$$ and $$z_{k+1}$$, we perform a gradient ascent update to the dual variables. This is simply given by the running sum of residuals

$$
u_{k+1} = u_k + F w_{k+1} - z_{k+1}
$$



---


### Special Cases of the Generalized Lasso

For different choices of $$F$$, we recover several well-studied problems as special cases. 

<br><br>

###### Standard Lasso 

When $$F$$ is an identity matrix $$I\in\mathbb{R}^{n \times n}$$, we recover the standard Lasso problem [2]

$$
\min_w \frac{1}{2} \|Dw - y \|_2^2 + \lambda \| w \|_1
$$

This model is commonly used in compressed sensing where we are interested in sparse signal recovery, and can be used as a convex relaxation for $$\ell_0$$-norm variable selection problem .

<br><br>

###### Variable Fusion

When $$F \in \mathbb{R}^{(n-1) \times n}$$ is a first order difference matrix, 

$$
F_{ij} = 
\begin{cases}
1 &,\quad i = j -1\\
-1 &,\quad i = j \\
0 &, \quad {\rm  otherwise}
\end{cases}
$$

we obtain the variable fusion model
[3] which corresponds to the problem

$$
\min_w \frac{1}{2} \|Dw - y \|_2^2 + \lambda \sum_{i=2}^n \| w_{k} -  w_{k-1}\|_1
$$

This model is used when we expect the ordering of the weights to have smooth structure.

<br><br>


###### Fused Lasso

When $$F \in \mathbb{R}^{(2n-1)\times n}$$ combines both the penalty from the standard lasso and the variable fusion, 

$$
F = 

\begin{bmatrix}
F_{\textrm{Lasso}} \\
F_{\textrm{Fusion}}
\end{bmatrix}


$$

we obtain the Fused Lasso model [4] which combines both a sparsity and smoothness penalty.


$$
\min_w \frac{1}{2} \|Dw - y \|_2^2 + \lambda_1 \| w \|_1 + \lambda_2 \sum_{i=2}^n \| w_{k} -  w_{k-1}\|_1 
$$


### Demo

I illustrate the impact of the various penalties mentioned earlier on the inference results. Noisy observations are generated from a random dictionary using smooth and sparse weights. For a given set of hyperparameters, we obtain the following ADMM solutions


<!-- <div class="home-img-container">
  <img src="assets/img/ADMM-lasso-results.png" width="300px" height="300px" id="ADMM"> 
</div> -->

<!-- <div class="col-sm mt-3 mt-md-0"> -->
<!-- </div> -->
<br> 


{% include figure.html path="assets/img/ADMM-lasso-results.png" class="admm-figure" width="75%" %}



The standard lasso problem encourages shrinkage towards zero of each weight independently. While it accurately captures certain regions with active support, adjacent weights can have large variance.  On the other hand, the variable fusion model prioritizes smooth patterns and effectively identifies distinct areas of smoothness. However, since its penalty does not promote zero shrinkage, it incorrectly identifies that the entire support is active. The fused Lasso model excels in accurately identifying both smoothness and the active support set. Importantly, note that solutions across all models exhibit a bias towards lower magnitudes than the actual weights. This is because $$\ell_1$$ inference provides a biased estimate of the weight vector.

<!-- This is a direct result of the soft-thresholding operation. -->

<!-- The variable fusion model encourages smooth structure and correctly finds different regions of smoothness. However, since it's penalty does not encourage shrinkage towards zero, it incorrectly identifies active coefficients (i.e. indices 1-12, 8-36 and 45-50). The fused Lasso model most accurately identifies smoothness and the set of active coefficients. Finally, notice that the optimal solution for all models are biased to have lower magnitude than the true weights.  -->




<br><br>


```python
import numpy as np
import numpy.random as npr

def soft_threshold(x, thresh):
    return np.sign(x)* np.max([np.abs(x)-thresh, np.zeros(len(x))], axis=0)

def ADMM_generalized_lasso(y, D, F, rho, lam, n_iters=100):

    # y: (m,) array. observation 

    # D: (m, n) array. dictionary

    # F: (k, n) array. constraint matrix

    # rho: augmented lagrange multiplier

    # lam: lagrange multiplier


    n = len(D.T)

    # random initialization

    w = npr.randn(n) 
    u = npr.randn(len(F)) 
    z = npr.randn(len(F)) 
    
    FtF = F.T @ F

    for i in range(n_iters):
        w = np.linalg.lstsq(D.T @ D + rho * FtF, D.T @ y + rho * F.T @ (z-u), rcond=None)[0]
        z = soft_threshold(F @ w + u, lam/rho)
        u = u + F @ w - z
        
    return w

from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt

n, m = 50, 100
n_nonzero_coefs = 10

# generate dictionary

_, D, _ = make_sparse_coded_signal(
    n_samples=1,
    n_components=n,
    n_features=m,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=1,
)
D = D.T



# generate structured coefficients

np.random.seed(1)
w_true = np.zeros(n)
for i in range(5):
    ix = np.random.choice(n)
    length = np.random.randint(5,10)
    w_true[ix:ix+length] = npr.randn()
    
    
# generate noisy observations 

y_true = D @ w_true
y = y_true + npr.randn(m)*0.2


# define hyperparameters

rho = 0.3 # augmentation multiplier

lam = 0.5 # general multiplier for L1

lam2 = 0.3 # multipler for sparsity in Fused Lasso




# construct F matrices

F_Lasso = np.eye(n) # Lasso solution

F_fusion = (np.diag(np.ones(n),k=1)[:-1,:-1] + np.diag(np.ones(n)*-1,k=0))[:-1] # Fusion Penalty solution

F_fusedLasso = np.concatenate([np.diag(np.ones(n)/lam*lam2), F_fusion]) # Fused Lasso solution


# compute ADMM solution

w_lasso = ADMM_generalized_lasso(y, D, F_Lasso, rho, lam)
w_fusion = ADMM_generalized_lasso(y, D, F_fusion, rho, lam)
w_fusedLasso = ADMM_generalized_lasso(y, D, F_fusedLasso, rho, lam)


    
# Plot

plt.figure(figsize=(6,3))
plt.plot(w_lasso, label="Lasso")
plt.plot(w_fusion, label="Fusion")
plt.plot(w_fusedLasso, label="Fused Lasso")
plt.plot(w_true, 'k--', label="true")
plt.ylabel("Weight")
plt.xlabel("index")
plt.legend()
plt.show()
    

```

---

1. Ryan J Tibshirani. "The solution path of the generalized lasso". 2011.
2. Robert Tibshirani. "Regression shrinkage and selection via the lasso". Journal of the Royal Statistical Society Series B: Statistical Methodology, 1996.
3. S Land, and J Friedman. "Variable fusion: a new method of adaptive signal regression". Technical Report, Department of Statistics, Stanford University, 1996.
4. Robert Tibshirani, Michael Saunders, Saharon Rosset, Ji Zhu, and Keith Knight. "Sparsity and smoothness via the fused lasso". Journal of the Royal Statistical Society Series B: Statistical Methodology, 2005.
{: .font-size-blog-ref}

