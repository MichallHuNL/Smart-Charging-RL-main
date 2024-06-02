import torch as th
import torch.nn.functional as F


class COMALearner:
    """ COMA learner class for multi-agent reinforcement learning. """

    def __init__(self, model, critic, idx, params={}):
        self.model = model
        self.critic = critic
        self.all_parameters = list(model.parameters()) + list(critic.parameters())
        self.gamma = params.get('gamma', 0.99)
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=params.get('critic_lr', 5E-4))
        self.criterion = th.nn.MSELoss()
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.idx = idx

    def critic_values(self, states, actions):
        """ Returns the Q-values from the critic network. """
        return self.critic(th.cat([states, actions], dim=-1))

    def compute_advantage(self, batch):
        """ Computes the advantage for each agent's action. """
        states = batch['states']
        actions = batch['actions']
        q_values = self.critic_values(states, actions)

        with th.no_grad():
            # Compute the counterfactual baseline
            baseline = th.zeros_like(q_values)
            for a in range(actions.size(-1)):
                counterfactual_actions = actions.clone()
                counterfactual_actions[..., a] = th.tensor(0.0)  # marginalize over action a
                baseline[..., a] = self.critic_values(states, counterfactual_actions)

        advantage = q_values - baseline.mean(dim=-1, keepdim=True)
        return advantage

    def train_critic(self, batch):
        """ Trains the critic network. """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        with th.no_grad():
            target_q_values = rewards + self.gamma * (~dones * self.critic_values(next_states, actions))

        q_values = self.critic_values(states, actions)
        critic_loss = self.criterion(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, batch):
        """ Performs one policy gradient update step using COMA. """
        self.model.train(True)

        # Train critic
        critic_loss = self.train_critic(batch)

        # Compute advantage
        advantage = self.compute_advantage(batch)

        # Compute policy gradient loss
        log_probs = th.log(self.model(batch['states'][self.idx])).gather(dim=-1, index=batch['actions'][self.idx])
        policy_loss = -(log_probs * advantage.detach()).mean()

        # Backpropagate policy loss
        self.optimizer.zero_grad()
        policy_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return policy_loss.item(), critic_loss
