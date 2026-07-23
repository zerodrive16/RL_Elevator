"""
Aufzugssteuerung als MDP - Schritt 1: Umgebung/Umwelt-Simulation
==================================================================

Bezug zur Vorlesung (W1_03):
- Zustand s_t = (f, d, w_1..w_4, z_1..z_4)
    f  : aktuelles Stockwerk (0..3, entspricht Etage 1..4)
    d  : Tuerstatus (offen / geschlossen)
    w_i: Anzahl wartender Personen in Etage i, hier vereinfacht auf {0,1}
    z_i: Anzahl Personen in der Kabine mit Ziel i, ebenfalls {0,1}
- Aktionen A = {H, R, O, S}, zustandsabhaengig eingeschraenkt (A(s))
- Uebergangsfunktion s_{t+1} = f(s_t, a_t, w_t) mit Zufallsvariable w_t
  (Bernoulli(p) je Etage, analog zum Lagerhaltungs-Beispiel W1_03)
- Kostenfunktion c(s,a) = sum(w_i) + sum(z_i)

Dies ist bewusst NUR die Umgebung (kein Lernalgorithmus, keine Strategie) -
entspricht der ersten "Baustelle" aus W1_03 ("Umwelt bzw. Umgebung,
Zustandsuebergaenge"), so wie im Feedback gefordert.
"""

import random


class ElevatorEnv:
    def __init__(self, n_floors=4, p=0.01, capacity=5, seed=None):
        self.n_floors = n_floors
        self.p = p
        self.capacity = capacity  # bindet in dieser Stufe (max. 1 Pers./Ziel) nie, siehe Bericht
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Startzustand: Kabine im Erdgeschoss, Tuer zu, niemand wartet/faehrt mit."""
        self.floor = 0
        self.door_open = False
        self.waiting = [0] * self.n_floors   # w_i
        self.targets = [0] * self.n_floors   # z_i
        return self.get_state()

    def get_state(self):
        return (self.floor, self.door_open, tuple(self.waiting), tuple(self.targets))

    def available_actions(self):
        """Entspricht A(s) aus der MDP-Modellierung."""
        if self.door_open:
            return ["S"]
        actions = ["O"]
        if self.floor < self.n_floors - 1:
            actions.append("H")
        if self.floor > 0:
            actions.append("R")
        return actions

    def step(self, action):
        """Fuehrt eine Aktion aus und liefert (Folgezustand, Kosten) zurueck."""
        if action not in self.available_actions():
            raise ValueError(f"Aktion {action} in Zustand {self.get_state()} nicht verfuegbar")

        cost = self._cost()  # Kosten des aktuellen Zustands/Aktion (vor dem Uebergang)

        if action == "H":
            self.floor += 1
            self.door_open = False
        elif action == "R":
            self.floor -= 1
            self.door_open = False
        elif action == "S":
            self.door_open = False
        elif action == "O":
            self.door_open = True
            f = self.floor
            self.targets[f] = 0          # Zielerreichende steigen aus
            if self.waiting[f] == 1:     # wartende Person steigt ein (max. 1, da w_i in {0,1})
                self.waiting[f] = 0
                ziel_kandidaten = [i for i in range(self.n_floors) if i != f]
                ziel = self.rng.choice(ziel_kandidaten)
                self.targets[ziel] = 1    # neues Fahrtziel gedrueckt

        # stochastischer Ankunftsprozess w_t: pro Etage unabhaengig, Deckel bei 1
        for i in range(self.n_floors):
            if self.waiting[i] == 0 and self.rng.random() < self.p:
                self.waiting[i] = 1

        return self.get_state(), cost

    def _cost(self):
        return sum(self.waiting) + sum(self.targets)


def demo(steps=100, p=0.01, seed=42):
    """Simuliert die Umgebung mit einer zufaelligen (noch nicht gelernten) Aktionswahl,
    um zu zeigen, dass die Umwelt-Dynamik unabhaengig von einer Strategie funktioniert."""
    env = ElevatorEnv(p=p, seed=seed)
    rng = random.Random(seed)
    state = env.get_state()
    print(f"t=0  Zustand={state}")

    total_cost = 0
    for t in range(1, steps + 1):
        action = rng.choice(env.available_actions())
        state, cost = env.step(action)
        total_cost += cost
        print(f"t={t:<3} Aktion={action}  Zustand={state}  Kosten={cost}")

    print(f"\nGesamtkosten ueber {steps} Schritte: {total_cost}")


if __name__ == "__main__":
    demo()

