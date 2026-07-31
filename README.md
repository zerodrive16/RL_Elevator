# RL Elevator — Projektvorschlag 12

Aufzugssteuerung als MDP, gelöst per Value Iteration in mehreren Phasen
(Random-Baseline → volle Information → realistische Kapazität → reduzierte
Beobachtung). Details, formale MDP-Definitionen, Ergebnisse und Diskussion:
siehe **[`PHASES.md`](PHASES.md)**.

## Projektstruktur

```
phase1_2/   Phase 1 (Random-Baseline) + Phase 2 (Value Iteration, volle Info/Orakel)
phase2b/    Phase 2b (Value Iteration, realistische Kapazität/Mehrfachwarten)
phase3/     Phase 3 (Value Iteration mit Boolean-Beobachtung)
common/     Geteilte, Environment-unabhängige Infrastruktur (Auswertung)
results/    Alle Lauf-Ergebnisse (CSV) - siehe unten
p_variation.py   Teil (c): p/K-Sweep über alle Phasen (bleibt im Projekt-Root)
```

Jeder Ordner ist ein Python-Package (`__init__.py`), Imports zwischen den
Ordnern sind absolut (z.B. `from phase1_2.value_iteration import ...`). Jede
Datei mit Cross-Package-Imports setzt den Projekt-Root selbst auf `sys.path`
(siehe Kopf der jeweiligen Datei) — Skripte funktionieren deshalb sowohl
direkt (`python phase2b/value_iteration_2.py`) als auch als Modul
(`python -m phase2b.value_iteration_2`), **solange sie aus dem Projekt-Root
heraus aufgerufen werden** (wichtig für die relativen `results/...`-Pfade).

## Welche Datei für welches Experiment

| Datei | Experiment | Befehl |
|---|---|---|
| `phase1_2/elevator_environment.py` | Phase 1: Umgebung + Random-Demo | `python phase1_2/elevator_environment.py` |
| `phase1_2/value_iteration.py` | Phase 1 + 2: Random vs. Value Iteration (volle Info/Orakel) | `python phase1_2/value_iteration.py` |
| `phase2b/elevator_environment_2.py` | Phase 2b: Umgebung (echte Kapazität/Mehrfachwarten) + Demo | `python phase2b/elevator_environment_2.py` |
| `phase2b/value_iteration_2.py` | Phase 2b: Random vs. Value Iteration (realistische Kapazität) | `python phase2b/value_iteration_2.py` |
| `phase3/value_iteration_3.py` | Phase 3: Value Iteration mit Boolean-Beobachtung, ausgewertet auf der Phase-2b-Umgebung | `python phase3/value_iteration_3.py` |
| `p_variation.py` | Teil (c): alle Phasen für mehrere `p`- und `K`-Werte auf einmal (`P_VALUES`/`K_VALUES` am Dateianfang) | `python p_variation.py` |

Parameter (`p`, `K` = `max_wait_per_floor`, `gamma`, ...) stehen jeweils im
`__main__`-Block der einzelnen Dateien bzw. am Anfang von `p_variation.py`.

**Laufzeit-Hinweis:** Phase 1/2 und Phase 3 sind schnell (Sekunden). Phase 2b
braucht bei `K=3` ca. 6-9 Minuten (großer Zustandsraum), `p_variation.py` mit
mehreren `p`/`K`-Kombinationen entsprechend länger (ggf. im Hintergrund laufen
lassen).

## Wo die Ergebnisse landen

- **`results/runs.csv`** — zentrale Ergebnistabelle, eine Zeile pro
  Phase/Parameter-Kombination (`label`, `p`, `K`, `avg_cost_per_step`,
  `avg_trip_duration`, `n_completed_trips_total`, ...). Wird von jedem Skript
  per Upsert ergänzt (`common/evaluation.py::append_result`) — bestehende
  Kombinationen werden ersetzt, andere bleiben erhalten, nichts geht verloren.
- **`results/policy_*.csv`** — vollständige Strategie-Tabelle (`V*`, `pi*`
  für jeden Zustand) der jeweiligen Value Iteration, für Phase 2b/3 mit `p`
  (und bei Phase 2b zusätzlich `K`) im Dateinamen.

## Sonstige Dateien

- `common/evaluation.py` — gemeinsame, Environment-unabhängige Auswertungslogik
  (`evaluate()`, `run_episode()`, `append_result()`), von allen Phasen genutzt.
- `PHASES.md` — vollständige Dokumentation: formale MDP-Definitionen,
  Ergebnisse, Diskussion (u.a. Systemüberlastung bei hoher Last, Wahl von `K`).
- `img.png`, `img_1.png` — Vorlesungsfolien-Screenshots (Quelle für die
  Umgebungszustand/Agentenzustand/Beobachtbarkeit-Definitionen in `PHASES.md`).