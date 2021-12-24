from bear.exploration.base import BaseExplorer
import numpy as np
import torch

class PongExplorer(BaseExplorer):
    """
    action space:
        [0,1,2,3,4,5]
    an example of action logits:
        [[0.9, 0.1],[0.8,0.2]]
    """
    def explore(self, act_logits):
        """combined reverse and explore """
        # probs = torch.FloatTensor(act_logits)
        # sample_actions = torch.distributions.Categorical(probs).sample()
        # return sample_actions.cpu().numpy()
        return np.asarray(list(map(lambda logits: np.random.choice([0, 1, 2, 3, 4, 5], p=logits), act_logits)))

    def reverse_action(self, act_logits):
        return np.argmax(act_logits, axis=1)
