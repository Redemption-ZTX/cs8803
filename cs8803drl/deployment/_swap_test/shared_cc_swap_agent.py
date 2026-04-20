from cs8803drl.deployment import trained_shared_cc_agent as _base


class SharedCCSwapAgent(_base.SharedCCAgent):
    def act(self, observation):
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            return super().act(observation)
        swapped_obs = {
            player_ids[0]: observation[player_ids[1]],
            player_ids[1]: observation[player_ids[0]],
        }
        swapped_action = super().act(swapped_obs)
        return {
            player_ids[0]: swapped_action[player_ids[1]],
            player_ids[1]: swapped_action[player_ids[0]],
        }


Agent = SharedCCSwapAgent

