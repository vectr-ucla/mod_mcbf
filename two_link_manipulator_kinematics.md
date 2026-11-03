# Kinematics of the Planar Two-Link Manipulator

Angles are absolute, as in the paper: $\theta_1$ and $\theta_2$ are both measured counter-clockwise from the horizontal. $l_1$ and $l_2$ are the link lengths, and the state $\mathbf x=(\theta_1,\theta_2)$ collects the joint angles. Adopt the shorthand $c_i=\cos\theta_i$, $s_i=\sin\theta_i$ for $i=1,2$.

## 1. End-effector position

Joint 1 is at the origin and joint 2 is at $l_1[c_1,\ s_1]^\top$. The end effector lies a further $l_2$ along the direction $\theta_2$:

$$
\mathbf y(\mathbf x) = \begin{bmatrix} l_1\cos\theta_1 + l_2\cos\theta_2 \\
l_1\sin\theta_1 + l_2\sin\theta_2 \end{bmatrix}.
$$

## 2. End-effector Jacobian

$$
\mathbf J(\mathbf x) = \frac{\partial\mathbf y}{\partial\mathbf x}
= \begin{bmatrix}
\dfrac{\partial\mathbf y}{\partial\theta_1} & \dfrac{\partial\mathbf y}{\partial\theta_2}
\end{bmatrix}
= \begin{bmatrix}
-l_1\sin\theta_1 & -l_2\sin\theta_2 \\
l_1\cos\theta_1 & l_2\cos\theta_2
\end{bmatrix},
\qquad \dot{\mathbf y} = \mathbf J(\mathbf x)\thinspace\dot{\mathbf x} .
$$

## 3. $\mathbf J\mathbf J^\top$

$$
\mathbf J\mathbf J^\top =
\begin{bmatrix} -l_1 s_1 & -l_2 s_2 \\
l_1 c_1 & l_2 c_2 \end{bmatrix}
\begin{bmatrix} -l_1 s_1 & l_1 c_1 \\
-l_2 s_2 & l_2 c_2 \end{bmatrix}
= \begin{bmatrix}
l_1^2 s_1^2 + l_2^2 s_2^2 & -\left(l_1^2 s_1 c_1 + l_2^2 s_2 c_2\right) \\
-\left(l_1^2 s_1 c_1 + l_2^2 s_2 c_2\right) & l_1^2 c_1^2 + l_2^2 c_2^2
\end{bmatrix}.
$$


## 4. The singularity-avoidance barrier

Proposition 1 and Theorem 1 below are those of the paper *Optimal Decay Matrix Control Barrier Functions: Soundness and Completeness for Matrix Safety Constraints* by Gessow, Wu, Ames, and Lopez ([arXiv:2610.18675](https://arxiv.org/abs/2610.18675)); Proposition 1 concerns scalar optimal decay (SOD) and Theorem 1 matrix optimal decay (MOD). With $\bar\sigma>0$ the required lower bound on the smallest singular value of $\mathbf J$,

$$
\mathbf M = \mathbf J\mathbf J^\top,\qquad
\mathbf H = \mathbf M-\bar\sigma^2 \mathbf I\succeq\mathbf 0 \iff \sigma_{\min}(\mathbf J)\ge\bar\sigma,\qquad
\dot{\mathbf H} = \dot\theta_1\thinspace\partial_{\theta_1}\mathbf M + \dot\theta_2\thinspace\partial_{\theta_2}\mathbf M .
$$

Explicitly, in the Cartesian coordinates of $\mathbf y$,

$$
\mathbf H=\begin{bmatrix}
l_1^2 s_1^2 + l_2^2 s_2^2-\bar\sigma^2 & -\left(l_1^2 s_1 c_1 + l_2^2 s_2 c_2\right) \\
-\left(l_1^2 s_1 c_1 + l_2^2 s_2 c_2\right) & l_1^2 c_1^2 + l_2^2 c_2^2-\bar\sigma^2
\end{bmatrix}.
$$

The eigenvalues of $\mathbf H$ follow from its trace and determinant. From the entries above, $\mathrm{tr}\thinspace \mathbf H=l_1^2+l_2^2-2\bar\sigma^2$ and $\det \mathbf H=(l_1^2-\bar\sigma^2)(l_2^2-\bar\sigma^2)-l_1^2l_2^2\cos^2(\theta_2-\theta_1)$, so the eigenvalues of $\mathbf H$ are

$$
\lambda_{\min}(\mathbf H),\ \lambda_{\max}(\mathbf H)=\frac{l_1^2+l_2^2-2\bar\sigma^2\mp\sqrt{(l_1^2-l_2^2)^2+4l_1^2l_2^2\cos^2(\theta_2-\theta_1)}}{2},
$$

and $\mathbf H\succeq\mathbf 0$ iff $\lambda_{\min}(\mathbf H)\ge0$. Both depend on $\mathbf x$ only through the angle $\theta_2-\theta_1$ at joint 2: advancing $\theta_1$ and $\theta_2$ by the same angle rotates $\mathbf y$, $\mathbf J$, $\mathbf M$, and $\mathbf H$ rigidly, which turns the eigenvectors of $\mathbf H$ but leaves its eigenvalues unchanged.

The boundary $\partial\mathcal C$ of the safe set $\mathcal C$ is where $\lambda_{\min}(\mathbf H)=0$, i.e., $\det \mathbf H=0$ with $\mathrm{tr}\thinspace \mathbf H\ge0$:

$$
(l_1^2-\bar\sigma^2)(l_2^2-\bar\sigma^2)=l_1^2l_2^2\cos^2(\theta_2-\theta_1).
$$

Assume $\bar\sigma<\min(l_1,l_2)$, so that $\mathcal C$ has an interior: $\lambda_{\min}(\mathbf H)$ reaches its maximum $\min(l_1^2,l_2^2)-\bar\sigma^2$ at $\theta_2-\theta_1=\pm\pi/2$. Then $\sin2(\theta_2-\theta_1)\neq0$ on $\partial\mathcal C$, and there the eigenvalues of $\mathbf H$ are $0$ and

$$
\lambda=\mathrm{tr}\thinspace \mathbf H=l_1^2+l_2^2-2\bar\sigma^2>0,
$$

because the trace is the sum of the eigenvalues. The paper writes $\lambda$ for this eigenvalue; the zero eigenvalue is simple.

In double angles, $2s_ic_i=\sin2\theta_i$ and $c_i^2-s_i^2=\cos2\theta_i$, and with $l_1^2+l_2^2-2\bar\sigma^2=\lambda$, the entries of $\mathbf H$ on $\partial\mathcal C$ read

$$
\mathbf H=\frac12\begin{bmatrix}
\lambda-\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right) & -\left(l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\right)\\
-\left(l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\right) & \lambda+\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right)
\end{bmatrix},
$$

and $\det \mathbf H=0$ reads

$$
\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right)^2+\left(l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\right)^2=\lambda^2,
\qquad\text{i.e.,}\qquad
l_1^4+l_2^4+2l_1^2l_2^2\cos2(\theta_2-\theta_1)=\lambda^2 .
$$

Let $\mathbf v_0$ be the unit kernel vector, $\mathbf H\mathbf v_0=\mathbf 0$, and let $\mathbf v_1$ be $\mathbf v_0$ turned a quarter turn counter-clockwise, so that $\mathbf V=[\mathbf v_0\ \mathbf v_1]$ is a rotation, $\mathbf V^\top \mathbf V=\mathbf I$, $\det \mathbf V=1$. The first row of $\mathbf H\mathbf v_0=\mathbf 0$ gives $\mathbf v_0$ up to scale, $\mathbf v_0\propto\left(l_1^2s_1c_1+l_2^2s_2c_2,\ l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\right)$, and $\det \mathbf H=0$, which says that the square of the off-diagonal entry of $\mathbf H$ is the product of its diagonal entries, gives the scale: the squared length of this vector is $\left(l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\right)\mathrm{tr}\thinspace \mathbf H$. Hence

$$
\mathbf v_0=\frac{1}{\sqrt{\lambda\left(l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\right)}}
\begin{bmatrix}l_1^2s_1c_1+l_2^2s_2c_2\\
l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\end{bmatrix},\qquad
\mathbf v_1=\frac{1}{\sqrt{\lambda\left(l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\right)}}
\begin{bmatrix}-\left(l_1^2s_1^2+l_2^2s_2^2-\bar\sigma^2\right)\\
l_1^2s_1c_1+l_2^2s_2c_2\end{bmatrix}.
$$

If $l_1^2s_1^2+l_2^2s_2^2=\bar\sigma^2$, then $\det \mathbf H=0$ forces $l_1^2s_1c_1+l_2^2s_2c_2=0$ too, $\mathbf H=\mathrm{diag}(0,\lambda)$ already, and $\mathbf v_0=(1,0)$, $\mathbf v_1=(0,1)$. Multiplying out $\mathbf H\mathbf v_1$ and using $\det \mathbf H=0$ and $\mathrm{tr}\thinspace \mathbf H=\lambda$ gives $\mathbf H\mathbf v_1=\lambda \mathbf v_1$. So

$$
\mathbf V^\top \mathbf H\mathbf V=\begin{bmatrix}0&0\\
0&\lambda\end{bmatrix},\qquad\text{equivalently}\qquad
\mathbf H=\lambda\thinspace \mathbf v_1\mathbf v_1^\top,\qquad \mathbf v_0\mathbf v_0^\top=\mathbf I-\frac{\mathbf H}{\lambda} .
$$

The last form gives the products of the entries of $\mathbf v_0$ directly from the double-angle form of $\mathbf H$:

$$
\mathbf v_0\mathbf v_0^\top=\frac{1}{2\lambda}\begin{bmatrix}
\lambda+\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right) & l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\\
l_1^2\sin2\theta_1+l_2^2\sin2\theta_2 & \lambda-\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right)
\end{bmatrix}.
$$

Differentiating $\mathbf H$ along $\dot{\mathbf x}$ gives $\dot{\mathbf H}$. Since $\bar\sigma^2\mathbf I$ is constant, $\partial_{\theta_i}\mathbf H=\partial_{\theta_i}\mathbf M$, and

$$
\partial_{\theta_i}\mathbf M=l_i^2\begin{bmatrix}2s_ic_i& s_i^2-c_i^2\\
s_i^2-c_i^2&-2s_ic_i\end{bmatrix},\qquad
\dot{\mathbf H}=\begin{bmatrix}
2\left(l_1^2\dot\theta_1 s_1c_1+l_2^2\dot\theta_2 s_2c_2\right) & l_1^2\dot\theta_1\left(s_1^2-c_1^2\right)+l_2^2\dot\theta_2\left(s_2^2-c_2^2\right)\\
l_1^2\dot\theta_1\left(s_1^2-c_1^2\right)+l_2^2\dot\theta_2\left(s_2^2-c_2^2\right) & -2\left(l_1^2\dot\theta_1 s_1c_1+l_2^2\dot\theta_2 s_2c_2\right)
\end{bmatrix},
$$

so $\mathrm{tr}\thinspace\dot{\mathbf H}=0$. In double angles,

$$
\dot{\mathbf H}=\begin{bmatrix}
l_1^2\dot\theta_1\sin2\theta_1+l_2^2\dot\theta_2\sin2\theta_2 & -\left(l_1^2\dot\theta_1\cos2\theta_1+l_2^2\dot\theta_2\cos2\theta_2\right)\\
-\left(l_1^2\dot\theta_1\cos2\theta_1+l_2^2\dot\theta_2\cos2\theta_2\right) & -\left(l_1^2\dot\theta_1\sin2\theta_1+l_2^2\dot\theta_2\sin2\theta_2\right)
\end{bmatrix}.
$$

The matrices $\mathbf M$, $\mathbf H$, and $\dot{\mathbf H}$ above are written in the Cartesian coordinates of $\mathbf y$ and are not diagonal. Proposition 1 is stated after the change of basis by $\mathbf V$:

$$
\mathbf V^\top\dot{\mathbf H}\mathbf V=\begin{bmatrix}\mathbf Z_{00}&\mathbf Z_{01}\\
\mathbf Z_{01}&\mathbf Z_{11}\end{bmatrix},\qquad
\mathbf Z_{00}=\mathbf v_0^\top\dot{\mathbf H}\mathbf v_0,\quad \mathbf Z_{01}=\mathbf v_0^\top\dot{\mathbf H}\mathbf v_1,\quad \mathbf Z_{11}=\mathbf v_1^\top\dot{\mathbf H}\mathbf v_1=-\mathbf Z_{00},
$$

so the blocks of Proposition 1 are the entries of $\mathbf V^\top\dot{\mathbf H}\mathbf V$ and not of $\dot{\mathbf H}$; $\mathbf Z_{11}=-\mathbf Z_{00}$ because $\mathrm{tr}\thinspace\dot{\mathbf H}=0$. The paper's decay function $\alpha$ acts on $\mathbf H$ through its eigenvalues, as $\boldsymbol\alpha(\mathbf H)$, with $\alpha(0)=0$ and $\alpha(\lambda)>0$, so $\mathbf V^\top\boldsymbol\alpha(\mathbf H)\mathbf V=\mathrm{diag}(0,\alpha(\lambda))$. With $\mathbf Z_{00}^{+}=1/\mathbf Z_{00}$ for $\mathbf Z_{00}\neq0$ and $\mathbf Z_{00}^{+}=0$ for $\mathbf Z_{00}=0$, Proposition 1 says: the SOD condition, some $\omega\ge0$ with $\dot{\mathbf H}+\omega\thinspace\boldsymbol\alpha(\mathbf H)\succeq\mathbf 0$, holds if and only if

$$
\text{(i)}\ \mathbf Z_{00}\ge0,\qquad \text{(ii)}\ \mathbf Z_{00}=0\ \Rightarrow\ \mathbf Z_{01}=0 .
$$

Because the diagonal entries of $\dot{\mathbf H}$ are opposite and $\mathbf v_1$ is the quarter turn of $\mathbf v_0$, multiplying out the two quadratic forms leaves only the combinations of entries of $\mathbf v_0$ collected in $\mathbf v_0\mathbf v_0^\top$:

$$
\mathbf Z_{00}=\left(\dot{\mathbf H}\right)_{11}\left[\left(\mathbf v_0\mathbf v_0^\top\right)_{11}-\left(\mathbf v_0\mathbf v_0^\top\right)_{22}\right]+2\left(\dot{\mathbf H}\right)_{12}\left(\mathbf v_0\mathbf v_0^\top\right)_{12},\qquad
\mathbf Z_{01}=\left(\dot{\mathbf H}\right)_{12}\left[\left(\mathbf v_0\mathbf v_0^\top\right)_{11}-\left(\mathbf v_0\mathbf v_0^\top\right)_{22}\right]-2\left(\dot{\mathbf H}\right)_{11}\left(\mathbf v_0\mathbf v_0^\top\right)_{12}.
$$

Inserting the double-angle forms of $\mathbf v_0\mathbf v_0^\top$ and $\dot{\mathbf H}$,

$$
\begin{aligned}
\mathbf Z_{00}&=\frac{\left(l_1^2\dot\theta_1\sin2\theta_1+l_2^2\dot\theta_2\sin2\theta_2\right)\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right)-\left(l_1^2\dot\theta_1\cos2\theta_1+l_2^2\dot\theta_2\cos2\theta_2\right)\left(l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\right)}{\lambda},\\
\mathbf Z_{01}&=-\thinspace\frac{\left(l_1^2\dot\theta_1\sin2\theta_1+l_2^2\dot\theta_2\sin2\theta_2\right)\left(l_1^2\sin2\theta_1+l_2^2\sin2\theta_2\right)+\left(l_1^2\dot\theta_1\cos2\theta_1+l_2^2\dot\theta_2\cos2\theta_2\right)\left(l_1^2\cos2\theta_1+l_2^2\cos2\theta_2\right)}{\lambda},
\end{aligned}
$$

and multiplying out, the $l_1^4$ and $l_2^4$ terms cancel in $\mathbf Z_{00}$ while the cross terms combine by the angle-difference formulas:

$$
\mathbf Z_{00}=(\dot\theta_2-\dot\theta_1)\thinspace\frac{l_1^2l_2^2\sin2(\theta_2-\theta_1)}{\lambda},\qquad
\mathbf Z_{01}=-\thinspace\frac{l_1^4\dot\theta_1+l_2^4\dot\theta_2+l_1^2l_2^2(\dot\theta_1+\dot\theta_2)\cos2(\theta_2-\theta_1)}{\lambda}.
$$

At any configuration $\mathbf H=\lambda_{\min}(\mathbf H)\thinspace\mathbf v_0\mathbf v_0^\top+\lambda_{\max}(\mathbf H)\thinspace\mathbf v_1\mathbf v_1^\top$. Since $\mathbf v_0$, $\mathbf v_1$ stay orthonormal, $\mathbf{\dot v_0}=(\mathbf v_1^\top\mathbf{\dot v_0})\thinspace \mathbf v_1$ and $\mathbf{\dot v_1}=-(\mathbf v_1^\top\mathbf{\dot v_0})\thinspace \mathbf v_0$, and differentiating the decomposition gives $\mathbf v_0^\top\dot{\mathbf H}\mathbf v_0=\frac{d}{dt}\lambda_{\min}(\mathbf H)$ and $\mathbf v_0^\top\dot{\mathbf H}\mathbf v_1=-\left(\lambda_{\max}(\mathbf H)-\lambda_{\min}(\mathbf H)\right)\mathbf v_1^\top\mathbf{\dot v_0}$. On $\partial\mathcal C$,

$$
\mathbf Z_{00}=\frac{d}{dt}\lambda_{\min}(\mathbf H),\qquad
\mathbf Z_{01}=-\lambda\thinspace \mathbf v_1^\top\mathbf{\dot v_0} .
$$

The block $\mathbf Z_{00}$ is the rate of change of the margin and depends on the joint 2 rate $\dot\theta_2-\dot\theta_1$ alone; $\mathbf Z_{01}$ is $\lambda$ times the rate at which the kernel direction $\mathbf v_0$ turns. Off $\partial\mathcal C$ the two formulas for the blocks hold with $\lambda$ replaced by $\lambda_{\max}(\mathbf H)-\lambda_{\min}(\mathbf H)$.

The paper's example takes

$$
\mathbf H=\mathrm{diag}(0,\lambda),\qquad
\dot{\mathbf H}=\begin{bmatrix}0&b\\
b&c\end{bmatrix},
$$

with $\mathbf H$ and $\dot{\mathbf H}$ already written in the eigenbasis; in the notation here those are $\mathbf V^\top \mathbf H\mathbf V$ and $\mathbf V^\top\dot{\mathbf H}\mathbf V$. The first form holds at every $\mathbf x\in\partial\mathcal C$. The second holds iff $\mathbf Z_{00}=0$, and since $\sin2(\theta_2-\theta_1)\neq0$ on $\partial\mathcal C$ this is

$$
\dot\theta_1=\dot\theta_2 ,
$$

a rigid rotation of the whole arm about joint 1, which turns $\mathbf v_0$ with the arm, $\mathbf v_1^\top\mathbf{\dot v_0}=\dot\theta_1$. Then

$$
b=\mathbf Z_{01}=-\lambda\thinspace \mathbf v_1^\top\mathbf{\dot v_0}=-\lambda\thinspace\dot\theta_1,\qquad c=\mathbf Z_{11}=-\mathbf Z_{00}=0 ,
$$

the first also directly from the explicit $\mathbf Z_{01}$, since $l_1^4+l_2^4+2l_1^2l_2^2\cos2(\theta_2-\theta_1)=\lambda^2$ on $\partial\mathcal C$.

Condition (i) is $\mathbf Z_{00}\ge0$, i.e., $(\dot\theta_2-\dot\theta_1)\sin2(\theta_2-\theta_1)\ge0$: the margin must not decrease. Condition (ii) only bites when $\mathbf Z_{00}=0$, i.e., $\dot\theta_1=\dot\theta_2$, and then requires $b=-\lambda\dot\theta_1=0$. Hence, at any $\mathbf x\in\partial\mathcal C$,

$$
\text{SOD certifies }\dot{\mathbf x}\iff(\dot\theta_2-\dot\theta_1)\sin2(\theta_2-\theta_1)>0\ \text{ or }\ \dot{\mathbf x}=\mathbf 0,\qquad
\text{MOD (Theorem 1) certifies }\dot{\mathbf x}\iff(\dot\theta_2-\dot\theta_1)\sin2(\theta_2-\theta_1)\ge0,
$$

and the gap is exactly $b\neq0$:

$$
\dot\theta_1=\dot\theta_2\neq0 ,
$$

the whole arm rotating about joint 1 with $\theta_2-\theta_1$ frozen: the eigenvalues $0$ and $\lambda$ of $\mathbf H$ are unchanged while its kernel direction turns. The SOD constraint $\dot{\mathbf H}+\omega\thinspace\boldsymbol\alpha(\mathbf H)\succeq\mathbf 0$ is unchanged by the congruence with $\mathbf V$, and

$$
\mathbf V^\top(\dot{\mathbf H}+\omega\thinspace\boldsymbol\alpha(\mathbf H))\mathbf V=\begin{bmatrix}0&b\\
b&c+\omega\thinspace\alpha(\lambda)\end{bmatrix}
$$

has determinant $-b^2=-\lambda^2\dot\theta_1^2<0$ for every $\omega\ge0$.

## 5. The boundary

The safe set $\mathcal C$ is where $\lambda_{\min}(\mathbf H)\ge0$. With $\mathrm{tr}\thinspace \mathbf H>0$ under the assumption $\bar\sigma<\min(l_1,l_2)$, that is $\det \mathbf H\ge0$:

$$
\cos^2(\theta_2-\theta_1)\le\frac{(l_1^2-\bar\sigma^2)(l_2^2-\bar\sigma^2)}{l_1^2l_2^2}.
$$

It depends on $\mathbf x$ only through $\theta_2-\theta_1$. In the $(\theta_1,\theta_2)$ plane it is two bands, one around $\theta_2-\theta_1=\pi/2$ and one around $\theta_2-\theta_1=-\pi/2$, and $\partial\mathcal C$ is the four lines $\theta_2-\theta_1=\text{const}$ at their edges. Along a boundary line $\theta_1$ and $\theta_2$ advance together: the whole arm rotates about joint 1, $\mathbf J$ is multiplied on the left by the rotation through the same angle, and $\mathbf M$ and $\mathbf H$ are conjugated by it, so $\sigma_{\min}(\mathbf J)$, the eigenvalues of $\mathbf H$, and the margin are unchanged while $\mathbf v_0$ and $\mathbf v_1$ turn. Across a boundary line only $\theta_2-\theta_1$ changes.

At $\mathbf x\in\partial\mathcal C$ a velocity $\dot{\mathbf x}$ therefore has two parts: the joint 2 rate $\dot\theta_2-\dot\theta_1$, transverse to the boundary line, and the rate at which both joints advance together, along it. The blocks of Proposition 1 split accordingly. Using $l_1^4+l_2^4+2l_1^2l_2^2\cos2(\theta_2-\theta_1)=\lambda^2$ in the explicit $\mathbf Z_{01}$,

$$
\mathbf Z_{00}=(\dot\theta_2-\dot\theta_1)\thinspace\frac{l_1^2l_2^2\sin2(\theta_2-\theta_1)}{\lambda},\qquad
\mathbf Z_{01}=-\lambda\thinspace\dot\theta_1-(\dot\theta_2-\dot\theta_1)\thinspace\frac{l_2^2\left(l_2^2+l_1^2\cos2(\theta_2-\theta_1)\right)}{\lambda}.
$$

The block $\mathbf Z_{00}$, the rate of the margin, sees the joint 2 rate only. The block $\mathbf Z_{01}$, the coupling between the kernel and the range of $\mathbf H$, sees both, and along the boundary line it is $-\lambda\dot\theta_1$, the turning of the kernel.

## 6. When SOD and MOD differ

At $\mathbf x\in\partial\mathcal C$, MOD, by Theorem 1, certifies $\dot{\mathbf x}$ iff $\mathbf Z_{00}\ge0$: $\theta_2-\theta_1$ may move toward $\pm\pi/2$ or hold, and the whole arm may rotate about joint 1 at any rate. SOD certifies $\dot{\mathbf x}$ iff $\mathbf Z_{00}>0$, or $\dot{\mathbf x}=\mathbf 0$. The two differ exactly on

$$
\mathbf Z_{00}=0,\qquad \mathbf Z_{01}\neq0 ,
$$

which for the arm is $\dot\theta_1=\dot\theta_2\neq0$: motion along the boundary line. This motion keeps $\sigma_{\min}(\mathbf J)=\bar\sigma$, so it is safe, and it is the only motion that stays on $\partial\mathcal C$; the SOD condition rejects it for every $\omega$. In the eigenbasis the reason is visible: $\omega \mathbf V^\top\boldsymbol\alpha(\mathbf H)\mathbf V=\mathrm{diag}(0,\omega\thinspace\alpha(\lambda))$ adds to the $(\mathbf v_1,\mathbf v_1)$ entry only, while the entry $b=-\lambda\dot\theta_1$ coupling $\mathbf v_0$ to $\mathbf v_1$ can only be cancelled by an off-diagonal multiplier. That is what Theorem 1 supplies. Its condition is $\dot{\mathbf H}+\boldsymbol\Omega^\top\boldsymbol\alpha(\mathbf H)+\boldsymbol\alpha(\mathbf H)\boldsymbol\Omega\succeq\mathbf 0$ with a matrix $\boldsymbol\Omega$, and in the eigenbasis the blocks $\boldsymbol\Omega_{10}=-\mathbf Z_{01}/\alpha(\lambda)$ and $\boldsymbol\Omega_{11}$ large enough give

$$
\mathbf V^\top\left(\dot{\mathbf H}+\boldsymbol\Omega^\top\boldsymbol\alpha(\mathbf H)+\boldsymbol\alpha(\mathbf H)\thinspace\boldsymbol\Omega\right)\mathbf V=\begin{bmatrix}\mathbf Z_{00}&0\\
0&-\mathbf Z_{00}+2\thinspace\alpha(\lambda)\thinspace\boldsymbol\Omega_{11}\end{bmatrix}\succeq\mathbf 0 ,
$$

with $\boldsymbol\Omega_{10}=\lambda\dot\theta_1/\alpha(\lambda)$ along the boundary, a bounded multiplier.

The difference is not confined to exactly tangential motion. For $\mathbf Z_{00}>0$ the SOD constraint

$$
\mathbf V^\top(\dot{\mathbf H}+\omega\thinspace\boldsymbol\alpha(\mathbf H))\mathbf V=\begin{bmatrix}\mathbf Z_{00}&\mathbf Z_{01}\\
\mathbf Z_{01}&-\mathbf Z_{00}+\omega\thinspace\alpha(\lambda)\end{bmatrix}\succeq\mathbf 0
$$

holds iff its determinant is nonnegative,

$$
\omega\ \ge\ \frac{\mathbf Z_{00}^2+\mathbf Z_{01}^2}{\alpha(\lambda)\thinspace \mathbf Z_{00}} ,
$$

so at a fixed rotation rate the scalar decay needed grows without bound as $\dot\theta_2-\dot\theta_1$ goes to zero, while the matrix multiplier above stays bounded.

## 7. The rotation limit of the standard MCBF

The standard MCBF condition is the SOD condition of Section 4 with $\omega$ fixed at $1$:

$$
\dot{\mathbf H}+\boldsymbol\alpha(\mathbf H)\succeq\mathbf 0 .
$$

Consider the rotation of the whole arm about joint 1, $\dot\theta_1=\dot\theta_2$, at any configuration. It turns $\mathbf y$, $\mathbf J$, $\mathbf M$, and $\mathbf H$ rigidly, so the eigenvalues of $\mathbf H$ are unchanged and $\mathbf v_0$ and $\mathbf v_1$ turn with the arm, $\mathbf v_1^\top\mathbf{\dot v_0}=\dot\theta_1$. By Section 4, $\mathbf Z_{00}=\mathbf Z_{11}=0$, and $\mathbf Z_{01}=-\left(\lambda_{\max}(\mathbf H)-\lambda_{\min}(\mathbf H)\right)\dot\theta_1$, so

$$
\mathbf Z_{01}^2=\dot\theta_1^2\left((l_1^2-l_2^2)^2+4l_1^2l_2^2\cos^2(\theta_2-\theta_1)\right).
$$

Since $\boldsymbol\alpha(\mathbf H)$ is diagonal in the eigenbasis,

$$
\mathbf V^\top\left(\dot{\mathbf H}+\boldsymbol\alpha(\mathbf H)\right)\mathbf V=\begin{bmatrix}\mathbf v_0^\top\boldsymbol\alpha(\mathbf H)\mathbf v_0&\mathbf Z_{01}\\
\mathbf Z_{01}&\mathbf v_1^\top\boldsymbol\alpha(\mathbf H)\mathbf v_1\end{bmatrix}.
$$

On $\mathcal C$ its diagonal entries are nonnegative with product $\det\boldsymbol\alpha(\mathbf H)$, and a symmetric $2\times2$ matrix with nonnegative diagonal is positive semidefinite if and only if its determinant is nonnegative. The standard MCBF therefore certifies the rotation at $\mathbf x\in\mathcal C$ if and only if

$$
\dot\theta_1^2\left((l_1^2-l_2^2)^2+4l_1^2l_2^2\cos^2(\theta_2-\theta_1)\right)\le\det\boldsymbol\alpha(\mathbf H).
$$

This caps the rate of a motion that leaves $\sigma_{\min}(\mathbf J)$ and the margin unchanged. The cap is zero on $\partial\mathcal C$, where $\det\boldsymbol\alpha(\mathbf H)=0$. It is absent only where $\lambda_{\min}(\mathbf H)=\lambda_{\max}(\mathbf H)$, that is, for $l_1=l_2$ at $\theta_2-\theta_1=\pm\pi/2$.

For $\boldsymbol\alpha(\mathbf H)=\mathbf H$, $\det\boldsymbol\alpha(\mathbf H)=\det \mathbf H=(l_1^2-\bar\sigma^2)(l_2^2-\bar\sigma^2)-l_1^2l_2^2\cos^2(\theta_2-\theta_1)$. The ratio of $\det \mathbf H$ to the bracket is strictly decreasing in $\cos^2(\theta_2-\theta_1)$, since the numerator of its derivative is $-l_1^2l_2^2\left(\mathrm{tr}\thinspace \mathbf H\right)^2$. The ratio is therefore largest at $\theta_2-\theta_1=\pm\pi/2$. Every rotation the standard MCBF certifies in $\mathcal C$ satisfies

$$
|\dot\theta_1|\le\bar\nu=\frac{\sqrt{(l_1^2-\bar\sigma^2)(l_2^2-\bar\sigma^2)}}{|l_1^2-l_2^2|} ,
$$

a finite bound for $l_1\neq l_2$.

## 8. The nominal controller

The arm is velocity controlled: the joint rates $\dot{\mathbf x}$ are the command. Each safety filter receives a nominal joint rate and returns the joint rate closest to it, in the sum of squares of the joint rates, among those its condition certifies, with a penalty on how far its decay variable, $\omega$ or $\boldsymbol\Omega$, moves from its nominal value.

The nominal joint rate makes the end effector $\mathbf y$ track a target point $\mathbf y_d$ that moves in the plane. It feeds the target's velocity forward and corrects the position error $\mathbf y_d-\mathbf y$ with a gain $\kappa>0$:

$$
\mathbf J(\mathbf x)\thinspace\dot{\mathbf x}=\dot{\mathbf y}_d+\kappa\left(\mathbf y_d-\mathbf y\right).
$$

Since $\det \mathbf J=l_1l_2\sin(\theta_2-\theta_1)$, this has a unique solution wherever $\theta_2-\theta_1$ is not a multiple of $\pi$, in particular on all of $\mathcal C$. With $\dot{\mathbf y}=\mathbf J\dot{\mathbf x}$ from Section 2, the unfiltered error obeys

$$
\frac{d}{dt}\left(\mathbf y_d-\mathbf y\right)=-\kappa\left(\mathbf y_d-\mathbf y\right),
$$

so it shrinks as $e^{-\kappa t}$ along the straight line from $\mathbf y$ to $\mathbf y_d$.