import random
import os
import time


class SurvivalWorld:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent = [self.size // 2, self.size // 2]
        self.health = 100
        self.energy = 100
        self.score = 0
        self.steps = 0
        self.alive = True

        self.food = self.random_empty_position()
        self.danger = self.random_empty_position()

    def random_empty_position(self):
        while True:
            pos = [
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ]

            if pos != self.agent:
                return pos

    def move(self, action):
        if not self.alive:
            return

        x, y = self.agent

        if action == "UP":
            y -= 1
        elif action == "DOWN":
            y += 1
        elif action == "LEFT":
            x -= 1
        elif action == "RIGHT":
            x += 1
        elif action == "WAIT":
            pass

        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        self.agent = [x, y]

        self.energy -= 1
        self.steps += 1

        if self.agent == self.food:
            self.energy = min(100, self.energy + 30)
            self.score += 10
            self.food = self.random_empty_position()

        if self.agent == self.danger:
            self.health -= 35
            self.score -= 10
            self.danger = self.random_empty_position()

        if self.energy <= 0:
            self.health -= 5

        if self.health <= 0:
            self.alive = False

    def draw(self):
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 45)
        print("          CADENCE // SURVIVAL")
        print("=" * 45)
        print()

        for y in range(self.size):
            row = ""

            for x in range(self.size):
                pos = [x, y]

                if pos == self.agent:
                    row += " A "
                elif pos == self.food:
                    row += " F "
                elif pos == self.danger:
                    row += " X "
                else:
                    row += " . "

            print(row)

        print()
        print(f"HEALTH : {self.health}")
        print(f"ENERGY : {self.energy}")
        print(f"SCORE  : {self.score}")
        print(f"STEPS  : {self.steps}")
        print()
        print("A = AGENT   F = FOOD   X = DANGER")


if __name__ == "__main__":
    world = SurvivalWorld()

    actions = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]

    while world.alive:
        world.draw()

        action = random.choice(actions)

        print()
        print("ACTION:", action)

        world.move(action)

        time.sleep(0.15)

    world.draw()

    print()
    print("AGENT DIED")
    print("FINAL SCORE:", world.score)