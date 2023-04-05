from typing import List

from eval import rouge_single


class SmartReplyEnv:
    def __init__(
        self,
        threshold: float = 0.15,
        edit_reply: bool = True,

    ) -> None:

        super().__init__()
        self.threshold = threshold
        self.edit_reply = edit_reply

    def _step(self, action: List[str], target_reply: str):

        score = rouge_single(" ".join(target_reply), action)
        rewards = [0 for _ in action]

        # Step 1, reading the suggested replies 
        for idx, item in enumerate(action):
            score = rouge_single(target_reply, [action[idx]])
            if score > self.threshold:
                rewards[idx] = 1
                reply = item if not self.edit_reply else target_reply
                return reply, rewards
                
        # If no reply selected just type out response manually
        return target_reply, rewards       

    def step(self, action: List[str], target_reply: str):
        """
        action: list of text replies

        output should be structured as state, 
        """
        reply, rewards = self._step(action, target_reply)

        return reply, rewards
