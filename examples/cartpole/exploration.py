from bear.exploration.base import BaseExplorer
import numpy as np


class CartPoleExplorer(BaseExplorer):
    """
    action space:
        [0,1]
    an example of action logits:
        [[0.9, 0.1],[0.8,0.2]]
    """
    def explore(self, act_logits):
        """combined reverse and explore """
        return np.asarray(list(map(lambda logits: np.random.choice([0, 1], p=logits), act_logits)))

    def reverse_action(self, act_logits):
        return np.argmax(act_logits, axis=1)
