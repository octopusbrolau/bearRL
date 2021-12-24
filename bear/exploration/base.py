from abc import ABC, abstractmethod


class BaseExplorer(ABC):
    """
    action post-processing between policy outputs and env inputs
    """
    @abstractmethod
    def explore(self, act_logits):
        pass

    @abstractmethod
    def reverse_action(self, act_logits):
        """
        policy choose_action()->act_logits->reverse_action()->action that be executed by the env
        :param act_logits:
        :return:
        """
        pass

