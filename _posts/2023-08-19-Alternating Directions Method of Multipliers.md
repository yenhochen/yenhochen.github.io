---
layout: blogpost
title: Alternating Directions Method of Multipliers
date: 2023-08-19 15:53:00-0400
description: Complex optimization problems can often be approached by splitting them up into tractable subproblems. I describe an approach where local solutions are coordinated to find a global solution.
categories: 
giscus_comments: false
related_posts: false
scripts:
    # - posts/2023-07-01-transposedConv/transposedConv_tool.js 
style: 
# posts/2023-07-01-transposedConv/transposedConv_tool.css

---

---

Alternating Directions Method of Multipliers (ADMM) {% cite boyd2011distributed -f admm-generalized-lasso %} is an effective class of iterative optimization algorithms that are particularly well-suited towards separable and non-smooth objectives. The core idea is to break down a complex optimization problem into simple tractable subproblems. Most generally, ADMM targets objectives of the form

$$
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{equation}
\min_x f(x) + g(z) \quad
\textrm{ subject to }
Ax + Bz = c
\end{equation}
$$

for variables $$x \in \mathbb{R}^n$$ and $$z \in \mathbb{R}^p$$. Equality constraints are encoded through $$A \in \mathbb{R}^{k \times n}$$, $$B \in \mathbb{R}^{k \times p}$$, and $$c\in\mathbb{R}^k$$.
 <!-- $$z$$ is sometimes referred to as a slack variable since it allows deviatation from the  -->
For a convex optimization problem, one strategy for separating the objective is to define $$f(x)$$ to contain all the smooth terms and $$g(z)$$ to contain all the non-smooth terms. This decomposition allows us to split minimization over $$x$$ and $$z$$ into two steps, where each step often consists of a simple closed form update. 

<br>
ADMM then proceeds by forming the Augmented Lagrangian $$L$$ which has the general form

$$
L(x,z,y) = f(x) + g(z) + y^\top (Ax+Bz-c) + \frac{\rho}{2} \| Ax+Bz-c \|_2^2
$$

where $$\rho > 0$$. The above is the *unscaled* Augmented Lagrangian. However, it is often more convenient to work in its *scaled* form. Let $$r =  Ax+Bz-c$$. The last two terms can be combined as

$$
\begin{alignat*}{2}
y^\top r + \frac{\rho}{2} \| r \|_2^2 &= y^\top r + \frac{\rho}{2} r^\top r \\
&=  -2{ \left( -\frac{1}{2} y \right)}^\top r + \frac{\rho}{2} r^\top r \\
&= \frac{\rho}{2} \left(r+\frac{2}{\rho} \frac{1}{2} y \right)^\top \left(r+\frac{2}{\rho} \frac{1}{2} y \right) - \frac{2}{\rho} \left(\frac{1}{2} y \right)^\top \left(\frac{1}{2} y \right) \\
  &= \frac{\rho}{2} \| r+\frac{1}{\rho}  y  \|_2^2 
 - \frac{1}{2\rho}  \| y \|_2^2  \\
   &= \frac{\rho}{2} \| r+  u  \|_2^2 
 - \frac{\rho}{2}  \|  u \|_2^2  \\
\end{alignat*}
$$

where $$u=\frac{1}{\rho} y$$ is the scaled dual variable. Lines 3 comes directly from completing the square (Proposition 1 from {% cite rosenberg2017completing -f admm-generalized-lasso%}). All together, the scaled form of the augmented Lagrangian is given by

$$
\begin{equation}
L(x,z,u) = f(x) + g(z) + \frac{\rho}{2} \| Ax+Bz-c +  u  \|_2^2 
 - \frac{\rho}{2}  \|  u \|_2^2
 \end{equation}
$$


<br>

From here, we perform block minimization under the scaled $$L$$. This consists of iterating between a primal variable minimization step, a slack variable minimization step, and a gradient ascent update on the dual scaled variables. 


$$
\begin{alignat}{3}
1.& \quad x_{k+1} \quad &= \quad &\argmin_x \hspace{0.5em} L (x, z_k, u_k) \\
2.& \quad z_{k+1} \quad &= \quad &\argmin_z \hspace{0.5em} L (x_{k+1}, z, u_k)\\
3.& \quad u_{k+1} \quad &= \quad &u_k + Ax+Bz-c \\
\end{alignat} 
$$


<!-- [Discussion on the step-size for $$u$$. Primal Dual Feasability. step-size of 1 is a good choice!] -->


---

{% bibliography --cited_in_order --file admm-generalized-lasso -T bib-blog %}
{: .font-size-blog-ref}



