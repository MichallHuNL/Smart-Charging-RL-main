import torch as th
import torch.nn.functional as F



class COMALearner:
    """ COMA learner class for multi-agent reinforcement learning. """

    def __init__(self, models, critic, params={}):
        self.models = models  # List of agent models
        self.critic = critic  # Centralized critic
        self.n_actions = params.get('n_actions', 5)
        self.gamma = params.get('gamma', 0.99)
        self.optimizers = [th.optim.Adam(model.parameters(), lr=params.get('lr', 5E-4)) for model in models]
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=params.get('critic_lr', 5E-4))
        self.criterion = th.nn.MSELoss()
        self.grad_norm_clip = params.get('grad_norm_clip', 10)

    def critic_values(self, states, actions):
        """ Returns the Q-values from the critic network. """
        return self.critic(states, actions)

    def compute_advantage(self, batch, agent):
        """ Computes the advantage for each agent's action. """
        actions = th.cat([batch[a]['actions'] for a in batch.keys()], dim=1)
        states = th.cat([batch[a]['states'] for a in batch.keys()], dim=1)
        q_values = self.critic_values(states, actions)

        baseline = th.zeros_like(q_values)
        for a in range(self.n_actions):
            counterfactual_actions = actions.clone()
            counterfactual_actions[:, agent] = a  # Set the agent's action to each possible action
            with th.no_grad():
                q_counterfactual = self.critic_values(states, counterfactual_actions).squeeze(dim=-1)

            # Ensure q_counterfactual has the same shape as baseline
            q_counterfactual = q_counterfactual.unsqueeze(-1)

            baseline += batch[agent]['probabilities'][:, a].unsqueeze(-1) * q_counterfactual

        advantage = q_values - baseline
        return advantage

    def train_critic(self, batch):
        """ Trains the critic network. """
        actions = th.cat([batch[a]['actions'] for a in batch.keys()], dim=1)
        rewards = th.cat([batch[a]['rewards'] for a in batch.keys()], dim=1)
        next_states = th.cat([batch[a]['next_states'] for a in batch.keys()], dim=1)
        dones = th.cat([batch[a]['dones'] for a in batch.keys()], dim=1)
        states = th.cat([batch[a]['states'] for a in batch.keys()], dim=1)

        with th.no_grad():
            target_q_values = rewards + self.gamma * (~dones * self.critic_values(next_states, actions))

        q_values = self.critic_values(states, actions).expand(target_q_values.shape)
        critic_loss = self.criterion(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, batch):
        """ Performs one policy gradient update step using COMA. """
        # Train critic
        critic_loss = self.train_critic(batch)

        # Compute policy gradient loss for each agent
        policy_losses = []
        for agent, model in enumerate(self.models):
            model.train(True)
            advantage = self.compute_advantage(batch, agent)
            probabilities = F.softmax(model(batch[agent]['states'])[:, :self.n_actions], dim=-1)
            log_probs = th.log(probabilities).gather(dim=-1, index=batch[agent]['actions'])
            policy_loss = -(log_probs * advantage.detach()).mean()
            policy_losses.append(policy_loss)

            # Backpropagate policy loss
            self.optimizers[agent].zero_grad()
            policy_loss.backward(retain_graph=True)
            th.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
            self.optimizers[agent].step()

        return policy_losses, critic_loss