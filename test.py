import numpy as np
from direct.showbase.ShowBase import ShowBase

class DanceApp(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()
        self.camera.setPos(0, -25, 6)

        # Load model
        self.actor = self.loader.loadModel("models/man.glb")
        self.actor.reparentTo(self.render)
        self.actor.setScale(1, 1, 1)
        self.actor.setPos(0, 0, 0)

        self.time = 0
        self.taskMgr.add(self.update, "update")

    def set_bone(self, name, h=0, p=0, r=0):
        bone = self.actor.find("**/" + name)
        if not bone.isEmpty():
            bone.setHpr(h, p, r)

    def update(self, task):
        dt = globalClock.getDt()
        self.time += dt
        t = self.time

        # -----------------------------
        # 🔥 DANCE LOGIC
        # -----------------------------

        # Hip sway (core movement)
        self.set_bone("hip", 0, 0, 15 * np.sin(t * 2))

        # Chest twist
        self.set_bone("chest", 0, 10 * np.sin(t * 2), 0)

        # Head bounce
        self.set_bone("head", 0, 0, 10 * np.sin(t * 3))

        # -----------------------------
        # Arms swing
        # -----------------------------
        left_arm = 50 * np.sin(2 * t)
        right_arm = -50 * np.sin(2 * t)

        self.set_bone("upperarmL", 0, left_arm, 0)
        self.set_bone("upperarmR", 0, right_arm, 0)

        # Elbows
        self.set_bone("lowerarmL", 0, 30 * np.sin(2 * t + 1), 0)
        self.set_bone("lowerarmR", 0, -30 * np.sin(2 * t + 1), 0)

        # Hands (small motion)
        self.set_bone("handL", 0, 0, 15 * np.sin(3 * t))
        self.set_bone("handR", 0, 0, -15 * np.sin(3 * t))

        # -----------------------------
        # Legs stepping
        # -----------------------------
        left_leg = 40 * np.sin(2 * t)
        right_leg = -40 * np.sin(2 * t)

        self.set_bone("upperlegL", 0, left_leg, 0)
        self.set_bone("upperlegR", 0, right_leg, 0)

        # Knees bending
        self.set_bone("lowerlegL", 0, abs(30 * np.sin(2 * t)), 0)
        self.set_bone("lowerlegR", 0, abs(30 * np.sin(2 * t)), 0)

        # Feet tap
        self.set_bone("footL", 0, 0, 20 * np.sin(2 * t))
        self.set_bone("footR", 0, 0, -20 * np.sin(2 * t))

        return task.cont


app = DanceApp()
app.run()