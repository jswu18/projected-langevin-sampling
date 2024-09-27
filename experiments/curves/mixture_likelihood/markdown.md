# Mixture Likelihood

In this toy example we assume that the data $Y$ is generated as 
$$
Y(x) = f(x) + Z \cdot c + \sigma \epsilon
$$
where $c \in \mathbb{R}$, $Z \sim B(\alpha)$, $\epsilon \sim \mathcal{N}(0,1)$ and $\sigma > 0$. 

Essentially, there is a latent variable $Z$ which is unbobserved that cases an offset $c$. 

This leads to a mixture likelihood for $Y$ given as
$$
p(y | f(x) )= \alpha \mathcal{N}(y | f(x)+c, \sigma^2) + (1-\alpha) \mathcal{N}(y | f(x), \sigma^2)
$$
Consequently, with the notation from the paper, we obtain the cost
$$
c(y,f(x)) = - \log p(y|f(x)) = - \log  \Big( \alpha \mathcal{N}(y | f(x)+c, \sigma^2) + (1-\alpha) \mathcal{N}(y | f(x), \sigma^2) \Big).
$$
and the partial derrivative with respect to second component is
$$
\partial_2 c(y, f(x)) = - \frac{1}{ \sqrt{2\pi} \sigma^3 p(y|f(x))} \Big( 
    \alpha ( y - f(x)-c) \exp(- \frac{(y- f(x)-c)^2}{2 \sigma^2} )
    + (1-\alpha) (y- f(x)) \exp(- \frac{(y-f(x))^2}{2 \sigma^2}) .

\Big)
$$

