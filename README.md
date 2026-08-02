# RL Elevator — Projektvorschlag 12

Aufzugssteuerung als MDP, gelöst per Value Iteration in mehreren Phasen
(Random-Baseline → volle Information → realistische Kapazität → reduzierte
Beobachtung). Details, formale MDP-Definitionen, Ergebnisse und Diskussion:
siehe **[`PHASES.md`](PHASES.md)**.

## Projektstruktur

```
Phases/
  common/             Geteilte, Environment-unabhaengige Infrastruktur (Auswertung)
    evaluation.py
  environments/        Umgebungen (ElevatorEnv / ElevatorEnvV2), je eine Datei pro Variante
    phase1_2.py         ElevatorEnv - Phase 1 (Random-Baseline) + Phase 2 (VI, volle Info/Orakel)
    phase2b.py           ElevatorEnvV2 - Phase 2b (realistische Kapazitaet/Mehrfachwarten), auch von Phase 3 genutzt
  value_iteration/     Value-Iteration-Klassen, EINE gemeinsame Basisklasse + 3 Subklassen
    base.py              ValueIterationBase - Bellman-Loop, CSV-Export, Simulation, Tabellen-Policy (siehe unten)
    phase1_2.py           Phase12ValueIteration
    phase2b.py             Phase2bValueIteration
    phase3.py               Phase3ValueIteration (reduzierte/Boolean-Beobachtung)
Results/
  CSV/                Alle Lauf-Ergebnisse, Policy-Tabellen und Konvergenz-Logs (CSV)
  Images/             Generierte Plots (matplotlib), z.B. aus main.py
Scripts/              img.png, img_1.png - Vorlesungsfolien-Screenshots (Quelle fuer PHASES.md)
main.py               Teil (c): p/K-Sweep ueber die Phasen (aktuell nur Phase 2b aktiv, siehe unten)
```

Jeder Ordner unter `Phases/` ist ein Python-Package (`__init__.py`), Imports
sind absolut (z.B. `from Phases.value_iteration.phase2b import
Phase2bValueIteration`). Jede Datei mit Cross-Package-Imports setzt den
Projekt-Root selbst auf `sys.path` (siehe Kopf der jeweiligen Datei) —
Skripte funktionieren deshalb sowohl direkt (`python
Phases/value_iteration/phase2b.py`) als auch als Modul (`python -m
Phases.value_iteration.phase2b`), **solange sie aus dem Projekt-Root heraus
aufgerufen werden** (wichtig für die relativen `Results/...`-Pfade).

## Die Value-Iteration-Basisklasse

Die drei Phasen unterscheiden sich nur in ihrer Zustandsdarstellung und der
Uebergangsdynamik der Aktion `O` (Ein-/Aussteigen) - der Bellman-Update-Loop,
der Policy-CSV-Export, die Simulation einer fertigen Strategie auf einer
echten Umgebung sowie die Erzeugung einer `policy_fn(env)` fuer
`evaluate()` sind dagegen strukturell identisch. Das steckt einmal in
`Phases/value_iteration/base.py::ValueIterationBase`; jede Phase leitet
davon ab und ueberschreibt nur `enumerate_states()`, `cost()`,
`_boarding_distribution()` (nur der Aktion-O-Fall), `arrival_distribution()`,
`start_state()` und `state_from_env()`. Sogar innerhalb des O-Falls sind
H/R/S (Bewegung/Tuer schliessen) bereits in der Basisklasse implementiert, da
sie in allen drei Phasen identisch sind.

Verwendung (siehe auch die `__main__`-Bloecke der einzelnen Dateien):

```python
from Phases.value_iteration.phase2b import Phase2bValueIteration
from Phases.environments.phase2b import ElevatorEnvV2

vi = Phase2bValueIteration(n_floors=4, p=0.1, gamma=0.9, capacity=5, max_wait_per_floor=3)
V, policy = vi.run()                                   # Bellman-Optimalitaetsgleichung loesen
vi.export_policy_csv(path="Results/CSV/policy.csv")     # V*/pi* fuer alle Zustaende exportieren
vi.simulate(ElevatorEnvV2(n_floors=4, p=0.1, seed=42))   # Trajektorie unter pi* ausdrucken

policy_fn = vi.make_table_policy()   # fuer Phases.common.evaluation.evaluate()
```

## Welche Datei für welches Experiment

| Datei | Experiment | Befehl |
|---|---|---|
| `Phases/environments/phase1_2.py` | Phase 1: Umgebung + Random-Demo | `python Phases/environments/phase1_2.py` |
| `Phases/value_iteration/phase1_2.py` | Phase 1 + 2: Random vs. Value Iteration (volle Info/Orakel) | `python Phases/value_iteration/phase1_2.py` |
| `Phases/environments/phase2b.py` | Phase 2b: Umgebung (echte Kapazität/Mehrfachwarten) + Demo | `python Phases/environments/phase2b.py` |
| `Phases/value_iteration/phase2b.py` | Phase 2b: Random vs. Value Iteration (realistische Kapazität) | `python Phases/value_iteration/phase2b.py` |
| `Phases/value_iteration/phase3.py` | Phase 3: Value Iteration mit Boolean-Beobachtung, ausgewertet auf der Phase-2b-Umgebung | `python Phases/value_iteration/phase3.py` |
| `main.py` | Teil (c): p-Sweep (`P_VALUES`) für Phase 2b bei festem `K` (`K_VALUES[0]`), inkl. matplotlib-Plot nach `Results/Images/`. `run_phase1()`/`run_phase2()`/`run_phase3()` sind bereits vorhanden, aktuell aber nicht in `__main__` eingebunden. | `uv run python main.py` |

Parameter (`p`, `K` = `max_wait_per_floor`, `gamma`, ...) stehen jeweils im
`__main__`-Block der einzelnen Dateien bzw. am Anfang von `main.py`.

**Laufzeit-Hinweis:** Phase 1/2 und Phase 3 sind schnell (Sekunden). Phase 2b
braucht bei `K=2` ca. 2-3 Minuten, bei `K=3` ca. 6-9 Minuten (großer
Zustandsraum) - `main.py` läuft das für jeden Wert in `P_VALUES` einmal durch,
entsprechend länger (ggf. im Hintergrund laufen lassen). matplotlib ist als
Projekt-Dependency eingetragen (`uv add matplotlib`) - Skripte, die es nutzen,
daher mit `uv run python ...` statt nacktem `python ...` aufrufen.

## Wo die Ergebnisse landen

- **`Results/CSV/runs.csv`** — zentrale Ergebnistabelle, eine Zeile pro
  Phase/Parameter-Kombination (`label`, `p`, `K`, `avg_cost_per_step`,
  `avg_trip_duration`, `n_completed_trips_total`, ...). Wird von jedem Skript
  per Upsert ergänzt (`Phases/common/evaluation.py::append_result`) —
  bestehende Kombinationen werden ersetzt, andere bleiben erhalten, nichts
  geht verloren.
- **`Results/CSV/policy_*.csv`** — vollständige Strategie-Tabelle (`V*`,
  `pi*` für jeden Zustand) der jeweiligen Value Iteration
  (`ValueIterationBase.export_policy_csv()`), für Phase 2b/3 mit `p` (und
  bei Phase 2b zusätzlich `K`) im Dateinamen.
- **`Results/CSV/log_*.csv`** — Konvergenz-Historie (`iteration`, `delta`,
  `V_start`) einer Value Iteration, geschrieben von `main.py` via
  `write_convergence_log()` (`vi.run(return_history=True)`).
- **`Results/Images/*.png`** — generierte Plots (matplotlib), z.B.
  `phase2b_p_variation_K{K}.png` aus `main.py::plot_phase2b_results()`
  (Kosten/Schritt und Fahrtdauer, Random vs. Value Iteration, über `p`).

## Sonstige Dateien

- `Phases/common/evaluation.py` — gemeinsame, Environment-unabhängige
  Auswertungslogik (`evaluate()`, `run_episode()`, `append_result()`,
  `write_convergence_log()`, `make_random_policy()`), von allen Phasen genutzt.
- `Phases/value_iteration/base.py` — `ValueIterationBase`, siehe oben.
- `PHASES.md` — vollständige Dokumentation: formale MDP-Definitionen,
  Ergebnisse, Diskussion (u.a. Systemüberlastung bei hoher Last, Wahl von `K`).
- `Scripts/img.png`, `Scripts/img_1.png` — Vorlesungsfolien-Screenshots
  (Quelle für die Umgebungszustand/Agentenzustand/Beobachtbarkeit-Definitionen
  in `PHASES.md`).
