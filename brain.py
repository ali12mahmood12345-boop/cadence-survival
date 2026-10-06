import numpy as np

from cadence import Brain, LearnerConfig, ActorCriticConfig


ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "WAIT",
]


class SurvivalBrain:

    def __init__(self):

        learning = LearnerConfig(
            beta=0.1,
            temperature=0.2,
            tolerance=3e-3,
            free_steps=512,
            nudged_steps=12,
            eta=0.003,
            momentum=0.9,
            normalize=0.99,
            normalize_floor=1e-4,
        )

        reward_config = ActorCriticConfig(
            gamma=0.97,
            lam=0.9,
            eta=0.001,
            eta_critic=0.3,
            momentum=0.9,
            normalize=0.99,
        )

        self.brain = Brain.compose(
            5,
            len(ACTIONS),
            seed=42,
            learning=learning,
            reward=reward_config,
        )

        self.first_step = True
        self.last_action = "WAIT"
        self.decisions = 0
        self.outcomes = 0

    def think(self, perception, reward=None, done=False):

        x = np.asarray(
            [perception],
            dtype=np.float64
        )

        try:

            # First decision of the entire life:
            # there is no previous action to reward.
            if self.first_step:

                action_index = int(
                    self.brain.step(x)[0]
                )

                self.first_step = False

            else:

                # IMPORTANT:
                # Feed Cadence the REAL reward.
                # Do not convert it to only +1 / -1.
                actual_reward = (
                    0.0
                    if reward is None
                    else float(np.sign(reward))
                )

                action_index = int(
                    self.brain.step(
                        x,
                        reward=np.asarray(
                            [actual_reward],
                            dtype=np.float64
                        ),
                        done=np.asarray(
                            [bool(done)]
                        )
                    )[0]
                )

                self.outcomes += 1

            self.last_action = ACTIONS[action_index]
            self.decisions += 1

        except RuntimeError:

            self.last_action = "WAIT"

        return self.last_action

    def diagnostics(self):

        report = self.brain.last_learning or {}

        def number(name):
            value = report.get(name, 0.0)

            try:
                return round(
                    float(np.asarray(value).mean()),
                    4
                )
            except (TypeError, ValueError):
                return 0.0

        return {
            "decisions": self.decisions,
            "outcomes": self.outcomes,
            "td_error": number("td_error"),
            "delta": number("delta"),
            "dopamine": number("dopamine"),
            "trace": number("trace"),
            "saturation": number("saturation"),
            "capped": number("capped"),
        }


if __name__ == "__main__":

    print("Creating corrected Cadence brain...")

    brain = SurvivalBrain()

    test = [
        0.5,
        -0.5,
        0.2,
        -0.2,
        1.0
    ]

    action = brain.think(test)

    print()
    print("CADENCE BRAIN ONLINE")
    print("Decision:", action)
    print("Diagnostics:", brain.diagnostics())