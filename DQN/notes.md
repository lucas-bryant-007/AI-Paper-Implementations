# DQN Implementation Notes

In the paper, input was 84x84 grayscale pixels with 4 frames stacked together

## Algorithm 1 (Deep Q-learning with Experience Replay)
$$
\begin{aligned}
&\textbf{Initialize replay memory } \mathcal{D} \text{ to capacity } \mathcal{N}.\\
&\textbf{Initialize action-value function } Q \text{ with random weights } \theta.\\
&\textbf{Initialize target action-value function } \hat{Q} \text{ with weights } \theta^- \leftarrow \theta.\\[4pt]
&\textbf{for } \text{episode} = 1 \textbf{ to } \mathbb{M} \textbf{ do}\\
&\quad \text{Initialize sequence } s_1 = \{x_1\} \text{ and preprocessed sequence } \phi_1 = \phi(s_1).\\
&\quad \textbf{for } t = 1 \textbf{ to } T \textbf{ do}\\
&\qquad \text{With probability } \varepsilon \text{ select a random action } a_t,\\
&\qquad \text{otherwise select } a_t = \arg\max_a Q(\phi(s_t), a;\theta).\\
&\qquad \text{Execute action } a_t \text{ in emulator and observe reward } r_t \text{ and image } x_{t+1}.\\
&\qquad \text{Set } s_{t+1} = (s_t, a_t, x_{t+1}) \text{ and preprocess } \phi_{t+1} = \phi(s_{t+1}).\\
&\qquad \text{Store transition } (\phi_t, a_t, r_t, \phi_{t+1}) \text{ in } \mathcal{D}.\\
&\qquad \text{Sample random minibatch of transitions } (\phi_j, a_j, r_j, \phi_{j+1}) \text{ from } \mathcal{D}.\\
&\qquad y_j =
\begin{cases}
r_j, & \text{if the episode terminates at step } j+1,\\
r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a';\theta^-), & \text{otherwise.}
\end{cases}\\
&\qquad \text{Perform a gradient descent step on } \bigl(y_j - Q(\phi_j, a_j;\theta)\bigr)^2 \text{ w.r.t. } \theta.\\
&\qquad \text{Every } C \text{ steps reset } \hat{Q} \leftarrow Q \ \ (\theta^- \leftarrow \theta).\\
&\quad \textbf{end for}\\
&\textbf{end for}
\end{aligned}
$$
