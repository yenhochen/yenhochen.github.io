---
layout: blogpost
title: Interactive Transposed Convolution Tool
date: 2023-07-01 15:53:00-0400
description: I present an interactive tool that visualizes how altering parameters values of the transposed convlution affects the resulting output. Further, I discuss the connection between transposed convolution and standard convolution, and provide Pytorch Code that demonstrates their relationship.

# Additionally, I present an interactive tool that visualizes how altering parameters values affects the resulting output.


# Upsampling is a important operation I provide a tool that explores different parameters of the transposed convolution which upsamples data. 
categories: All 
giscus_comments: false
related_posts: false
scripts:
    - posts/2023-07-01-transposedConv/transposedConv_tool.js 
style: posts/2023-07-01-transposedConv/transposedConv_tool.css

---

---

Many computer vision tasks rely on convolutional and pooling layers to extract meaningful features from images. These operations often reduce the dimsionality of the features which can be computationally practical when training very deep models (e.g. less parameters to train) and enables learning important spatial hierarchies.
<!-- by increasing the size of the receptive field for downstream layers.  -->
While defining an operation that summarizes information can be accomplished through familiar functions such as summing, averaging, or taking the max or min, the converse operation of expanding the dimensionality of image features is not as intuitive. 

<br>

Below, I present an interactive tool where one can explore how each parameter of the transposed convolution, denoted as $$\star \star$$, affects the output. Hovering over each entry of the output shows its corresponding dot product from the input. When exploring, we see that the parameters behave much differently from standard convolution.
<!-- exploration reveals that these parameters exhibit intriguing behaviors distinct from those in standard convolution -->
For instance, increasing the padding surpringly results in a smaller output, and larger kernels correspond to more padding around the input. There's even an extra padding parameter not usually found in regular convolution.

<!-- Very quickly, it becomes clear that the parameters behave very differently from what one would expect from standard convolution. -->

<br>
 <!-- discussion closely follows the example provided by [^fn], but is more pedantic -->

<br>


{% include posts/2023-07-01-transposedConv/transposedConv_tool.html %}


<div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.4.2/chroma.min.js" charset="utf-8"></script>
    <script src="https://d3js.org/d3.v6.min.js" charset="utf-8"></script>
    <!-- https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js -->
    <div class="d3post1"></div>
    {% for js in page.scripts %}
        <script type="text/javascript">
            {% include {{ js }} %}
        </script>
    {% endfor %}
</div>


<br>



My goal with this post is to give a better sense of where the transposed convolution comes from and how it can be related to operations that we are familiar with. The discussion below closely follows the example in {% cite dumoulin2016guide --file transposed-conv %}, but includes insights from other resources to hopefully give a more complete picture.


---



### Convolution is a Linear Operation

Most often, the convolution, denoted as $$\star$$, is depicted as a kernel $$W$$ sliding over the input data $$X$$, computing dot products at each position to produce a lower-dimensional output $$Y$$. Equivalently, the convolution can be written as simple matrix-vector multiplication 

$$Y =X \star W  = \widetilde{W} \widetilde{x}$$


where $$\widetilde{W}$$ and $$\widetilde{x}$$ are augmentations of the original kernel and input data. For example, consider a convolution between a $$3\times 3$$ kernel over a  $$4 \times 4$$ input.

$$
X \star W = 


\begin{bmatrix}
   x_{00} & x_{01} & x_{02} & x_{03}\\
   x_{10} & x_{11} & x_{12} & x_{13}\\
   x_{20} & x_{21} & x_{22} & x_{23}\\
   x_{30} & x_{31} & x_{32} & x_{33}
\end{bmatrix}

\star

\begin{bmatrix}
   w_{00} & w_{01} & w_{02}\\
   w_{10} & w_{11} & w_{12}\\
   w_{20} & w_{21} & w_{22}
\end{bmatrix}



$$

To obtain the matrix-vector representation, we row-order flatten the input data into a vector $$\widetilde{x}$$ and rewrite the kernel as a sparse matrix using the elements of $$\widetilde{W}$$. The exact structure of the sparsity depends on the parameters of the convolution. If we assume that the stride is 1, dilation is 1, and padding is 0, then we get the following augmentated representation.


$$

\begin{equation*}
\begin{split}

\widetilde{W} \widetilde{x} = 

\begin{bmatrix}
   w_{00} & w_{01} & w_{02} & 0 & w_{10} & w_{11} & w_{12} & 0 & w_{20} & w_{21} & w_{22} & 0 & 0 & 0 & 0 & 0 \\
   0 & w_{00} & w_{01} & w_{02} & 0 & w_{10} & w_{11} & w_{12} & 0 &w_{20} & w_{21} & w_{22} & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & w_{00} & w_{01} & w_{02} & 0 & w_{10} & w_{11} & w_{12} & 0 & w_{20} & w_{21} & w_{22} & 0\\
   0 & 0 & 0 & 0 & 0 & w_{00} & w_{01} & w_{02} & 0 & w_{10} & w_{11} & w_{12} & 0 &w_{20} & w_{21} & w_{22} \\
\end{bmatrix}

\begin{bmatrix}
   x_{00} \\ 
   x_{01} \\ 
   x_{02} \\ 
   x_{03}\\
   x_{10} \\ 
   x_{11} \\ 
   x_{12} \\ 
   x_{13}\\
   x_{20} \\ 
   x_{21} \\ 
   x_{22} \\ 
   x_{23}\\
   x_{30} \\ 
   x_{31} \\ 
   x_{32} \\ 
   x_{33}
\end{bmatrix}

\end{split}
\end{equation*}

$$




<br> 

--- 

### Forward and Backward Pass of Linear Operators

The main insight from the previous section is that convolution is simply a linear operation with a very specific type of structure in the weight matrix. Therefore, we can analyze it like a linear layer. Let's take another look at the convolution $$y=\widetilde{W}\widetilde{x}$$, paying closer attention to how the dimensionality changes. In our example, $$M=16$$ and $$N=4$$. 

<br>

In the forward pass, a high dimensional vector $$\widetilde{x} \in \mathbb{R}^{M \times 1}$$ is mapped onto a low dimensional vector $$y \in \mathbb{R}^{N \times 1}$$ by the augmented kernel $$\widetilde{W} \in \mathbb{R}^{N \times M}$$.

<br>

In the backward pass, we compute the partial derivative of each matrix with respect to some loss $$l$$. The derivative with respect to the data shows how information is propogated through the convolution, and is given by

$$
\begin{equation*}
\frac{\partial l}{\partial \tilde{x}} = \frac{\partial y}{\partial \tilde{x}} \frac{\partial l}{\partial y} = \widetilde{W}^\top \frac{\partial l}{\partial y}
\end{equation*}
$$

A careful derivation of this result can be found in {% cite johnson-notes-backprop --file transposed-conv %}. Here, a low-dimensional vector $$\frac{\partial l}{\partial y} \in \mathbb{R}^{N\times 1}$$ is mapped to a high dimensional vector $$\frac{\partial l}{\partial \tilde{x}} \in \mathbb{R}^{M\times 1}$$ using $$\widetilde{W}^\top$$, or the transpose of the augmented kernel. 


<br>


Notice that while $$\widetilde{W}$$ reduces the dimensionality in the forward pass, its transpose $$\widetilde{W}^\top$$ expands the dimensionality in the backward pass. Importantly, both the forward and backward operations are defined by the same kernel. 


<br> <br> <br> <br> 


---

### From Convolution to Transposed Convolution and Back

If we swap the forward and backward passes of the standard convolution, then we get the Transposed Convolution, denoted as $$\star \star$$. By using the transpose of the augmented kernel as the forward operation, we achieve a dimensionality expansion in the outputs. It follows that on the backwards pass, the gradients of the loss undergo a dimensionality reduction, since $$(\widetilde{W}^\top)^\top = \widetilde{W}$$.

<br>

Let's see an example. Consider a low-dimensional input $$X \in \mathbb{R}^{2 \times 2}$$  and a kernel $$W\in\mathbb{R}^{3\times 3}$$. The corresponding row-order flattened data vector is given by $$\widetilde{x} \in \mathbb{R}^{4\times 1}$$. Its transposed convolution has the following matrix-vector form

$$
\begin{equation}
\label{eq:convT-matrix}
Y =  X \star \star \hspace{0.3em} W = \widetilde{W}^\top \tilde{x} = 

\begin{bmatrix}
    w_{00} &  0 & 0 & 0 \\
    w_{01} & w_{00} & 0 & 0 \\
    w_{02} & w_{01} & 0 & 0 \\
    0 & w_{02} &  0 &0 \\
    w_{10} & 0 &  w_{00} & 0 \\
    w_{11} & w_{10} & w_{01} &  w_{00} \\
    w_{12} & w_{11} & w_{02} & w_{01} \\
    0 & w_{12} & 0 & w_{02} \\
    w_{20} & 0 & w_{10} & 0 \\
    w_{21} & w_{20} & w_{11} & w_{10} \\
    w_{22} & w_{21} & w_{12} & w_{11} \\
    0 &  w_{22} & 0 & w_{12} \\
    0 & 0 & w_{20} &  0 \\
    0 & 0 & w_{21} &  w_{20} \\
    0 & 0 & w_{22} &  w_{21} \\
    0 & 0 & 0 &       w_{22} \\            
\end{bmatrix}

\begin{bmatrix}
   x_{00} \\ 
   x_{01} \\ 
   x_{10} \\ 
   x_{11} \\
\end{bmatrix}
\end{equation}
$$

Notice that the 4 dimensional input has been mapped to a 16 dimensional output. Interestingly, we can also show that any transposed convolution has a corresponding standard convolution form. This arises because the backward pass, being a linear operation, can be viewed as a convolution itself. Consider the following convolution

$$
\begin{equation}
\label{eq:convT_conv_form}
X' \star W' =
\begin{bmatrix}
   0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & x_{00} & x_{01} & 0 & 0\\ 
   0 & 0 & x_{10} & x_{11} & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}

\star


\begin{bmatrix}
   w_{22} & w_{21} & w_{20}\\
   w_{12} & w_{11} & w_{10}\\
   w_{02} & w_{01} & w_{00}
\end{bmatrix}
\end{equation}
$$

where $$X'$$ is the data matrix with padding equal to the kernel size minus 1 and $$W'$$ is the flip of the original kernel. Taking a stride of 1 and dilation of 1, it is easy to see that every inner product in equation $$\ref{eq:convT_conv_form}$$ corresponds to a single row times vector operation in equation $$\ref{eq:convT-matrix}$$. In fact, both forms are equivalent up to a reshape. 

$$
Y =  X \star \star \hspace{0.3em} W  = X' \star W'   = {\rm reshape}(\widetilde{W}^\top \tilde{x})
$$


<!-- We can reshape this output into its correspond $$4\times 4$$ matrix form. -->

<!-- $$
\begin{bmatrix}
    w_{00} x_{00} &  w_{01} x_{00} + w_{00} x_{01} &  w_{02} x_{00} + w_{01} x_{01}& w_{02} x_{01} \\
    w_{01} x_{00} + w_{00} x_{10}& 
    \begin{split}
    w_{11}x_{00} &+ w_{10}x_{01} \\
    + w_{01}&x_{10} + w_{00}x_{11}  
    \end{split}&

    \begin{split}
    w_{12} x_{00} &+ w_{11}x_{01} \\ + w_{02}&x_{10}+ w_{01}x_{11}
    \end{split}&
    w_{12}x_{01} + w_{02}x_{11} \\
    
    
    w_{20}x_{00} + w_{10} x_{10}& 
    \begin{split}
    w_{21}x_{00} &+ w_{20}x_{01} \\
    + w_{11}&x_{10} + w_{10} x_{11}
    \end{split}& 


    \begin{split}
    w_{22}x_{00} &+ w_{21}x_{01} \\
    + w_{12}&x_{10} + w_{11} x_{11}
    \end{split}&
    
    
    \\  
\end{bmatrix}
$$ -->



<br> <br> <br> <br> 

---


### Parameters for Pytorch Implementation

Up to now, we've observed how transposed convolutions achieve upsampling by exchanging the forward and backward passes of standard convolutions. Furthermore, the transposed convolution also has a corresponding standard convolution form. In general, for kernel size $$k$$, stride $$s$$, dilation $$d$$, padding $$p$$, and output padding $$o$$, the convolutional form is recovered by

<br>


1. padding within the data by inserting $$s - 1$$ zeros between rows and columns

2. padding the outer edges of the data with $$p' = d(k-1)-p + o$$ zeros.

3. Flipping the kernel $$W$$ to get $$W'$$

4. Convolve the padded data with the flipped kernel $$X' \star W'$$ with stride $$s' = 1$$.

<br>

Given the above procedure, we can understand how each of the parameters affect the transposed convolution output.

<br>

- $$s$$ only effects the amount of padding within the data elements.
- $$d$$ increases the outer padding size by a factor of $$k-1$$
- $$p$$ directly decreases the outer padding size on the input data.
- $$o$$ directly increases the outer padding size on the input data.


<br><br><br><br>

---


### Python Code

<!-- Below I present working code that shows the equivalence between the Pytorch implementation of transposed convolution and its corresponding convolution form. There is close agreement up to small numerical errors. -->

Here is code that demonstrates that transposed convolution and its standard convolution form are equivalent. It initializes random input data, performs a forward pass through a transposed convolution layer and the corresponding standard convolution, and then checks agreement between the outputs up to small numerical errors.




<br><br>


```python
# Import functions

import torch
import torch.nn as nn
import torch.nn.functional as F

# define zero padding in between matrix entries.

def pad_within(x, stride=2):
    if stride != 1:
        w = x.new_zeros(stride, stride)
        w[0, 0] = 1
        out = F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))
        out = out[:,:,:-stride+1,:-stride+1]
    else:
        out = x
    return out

# Set the parameters

w = 5
h = 10
kernel = 5
stride = 5
padding = 4
outpad = 2
dilation = 3

# Set the kernel

kernel_weights = torch.rand((1, 1, kernel, kernel))

# generate some data

img = torch.randn(h,w).float().reshape(1,1,h,w)

# initiate pytorch function with generated kernel

convT = nn.ConvTranspose2d(1, 1, kernel, stride, padding, outpad, bias=False, dilation=dilation)
convT.weight.data = kernel_weights

# compute output size

w_out = (w-1)*stride -2*padding+dilation*(kernel-1)+outpad+1
h_out = (h-1)*stride -2*padding+dilation*(kernel-1)+outpad+1

# pad or crop edges and within

p = dilation * (kernel - 1) - padding + outpad
if p >=0:
    img_ = pad_within(img, stride)
    img_ = torch.nn.functional.pad(img_, (p, p, p, p))
else:
    img_ = pad_within(img, stride)[0, 0, -p:p, -p:p][None,None,:]


# convolve padded image and transposed kernel

kernel_transposed = kernel_weights.flip(2,3)
uf = nn.Unfold(kernel, dilation=dilation, padding=0, stride=1)
conv2d = (uf(img_)*kernel_transposed.reshape(1,-1,1)).sum(1)
my_convT = conv2d.reshape(h_out+outpad, w_out+outpad)[outpad:,outpad:]

# check agreement with torch output

print(torch.isclose(torch_convT, my_convT, atol=1e-7).prod() == 1)
```




<!-- https://leimao.github.io/blog/Transposed-Convolution-As-Convolution/ -->
<!-- https://www.coursera.org/lecture/convolutional-neural-networks/transpose-convolutions-kyoqR -->

<!-- https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11 -->


<!-- https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md -->

<br><br><br><br><br><br>

---

**References**



{% bibliography --cited --file transposed-conv -T bib-blog %}
{: .font-size-blog-ref}

<br><br><br><br><br><br>

---

<br><br><br>


<!-- [^1]: {% reference dumoulin2016guide -f transposed-conv %} -->