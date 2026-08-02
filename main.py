"""
Teil (c) der Aufgabenstellung: Ankunftswahrscheinlichkeit p (und den
Warteschlangen-Deckel K fuer Phase 2b/3) variieren und alle bisherigen
Phasen (1, 2, 2b, 3) fuer jede Kombination wiederholen.

Phase 1/2 haengt NICHT von K ab (nur von p) - wird deshalb pro p nur EINMAL
gerechnet, nicht pro (p, K)-Kombination (das waere unnoetig doppelte Arbeit).
Phase 2b/3 laufen fuer jede Kombination aus P_VALUES x K_VALUES.

Alle Ergebnisse werden an Results/CSV/runs.csv angehaengt (append_result() ist
ein Upsert - ein erneuter Lauf mit gleichem p/K ersetzt nur die jeweilige
Zeile, alle anderen p/K-Kombinationen bleiben unberuehrt erhalten).
"""

import os
import random

import matplotlib
matplotlib.use("Agg")  # nur PNG-Export, kein Display/GUI-Backend noetig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Phases.common.evaluation import evaluate, append_result, make_random_policy, write_convergence_log
from Phases.environments.phase1_2 import ElevatorEnv
from Phases.environments.phase2b import ElevatorEnvV2
from Phases.value_iteration.phase1_2 import Phase12ValueIteration
from Phases.value_iteration.phase2b import Phase2bValueIteration
from Phases.value_iteration.phase3 import Phase3ValueIteration

N_FLOORS = 4
GAMMA = 0.9
CAPACITY = 5
EVAL_STEPS = 100
EVAL_EPISODES = 20
P_VALUES = [0.1, 0.2, 0.3, 0.5]
K_VALUES = [3]
# K=3 ist der eigentliche Referenzwert (siehe PHASES.md "Naechste Schritte") -
# K=2 lief bereits als Vergleichslauf (Ergebnisse/Plots mit "_K2" bleiben
# erhalten, siehe Results/), hier aber bewusst nicht mehr mit drin, um die
# Laufzeit zu reduzieren. Alle drei Plots werden JE K in eine eigene Datei
# geschrieben (K im Dateinamen), ueberschreiben sich also nicht - kommt K2
# spaeter wieder dazu, reicht K_VALUES = [2, 3]. Phase 1/2/3 sind unten
# weiterhin deaktiviert, es laeuft nur Phase 2b - run_phase1()/run_phase2()/
# run_phase3() bleiben unveraendert bestehen, um spaeter wieder mit
# eingebunden zu werden.


def make_v2_random_policy(seed):
    rng = random.Random(seed)
    return lambda env: rng.choice(env.available_actions())


def run_phase1(p):
    """NUR Phase 1 (Random-Baseline). Haengt nur von p ab.
    Kein V (keine Value Iteration - reine Zufallsstrategie hat keine Wertfunktion)."""
    print(f"\n===== Phase 1, p={p} =====")
    env1_factory = lambda seed: ElevatorEnv(n_floors=N_FLOORS, p=p, seed=seed)
    result = evaluate(
        label="phase1_random", policy_fn_factory=make_random_policy,
        env_factory=env1_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=N_FLOORS, p=p,
    )
    result["V"] = ""  # keine Value Iteration fuer die Zufallsstrategie
    print(f"  fertig: avg_cost_per_step={result['avg_cost_per_step']}")
    append_result(result)
    return result


def run_phase2(p):
    """NUR Phase 2 (VI, volle Info/Orakel). Haengt nur von p ab."""
    print(f"\n===== Phase 2, p={p} =====")
    vi = Phase12ValueIteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, theta=1e-4)
    V1, policy1, history1 = vi.run(verbose=False, return_history=True)
    write_convergence_log(f"Results/CSV/log_phase2_vi_full_info_p{p:.2f}.csv", history1)
    env1_factory = lambda seed: ElevatorEnv(n_floors=N_FLOORS, p=p, seed=seed)
    result = evaluate(
        label="phase2_vi_full_info", policy_fn_factory=lambda seed: vi.make_table_policy(),
        env_factory=env1_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=N_FLOORS, p=p,
    )
    start1 = vi.start_state()
    result["V"] = round(V1[start1], 4)
    print(f"  fertig: V*={V1[start1]:.2f}")
    append_result(result)
    return result


def run_phase2b(p, K):
    """NUR Phase 2b (Random + VI, realistische Kapazitaet). Gibt zusaetzlich
    die Konvergenz-Historie der VI zurueck (fuer plot_convergence())."""
    print(f"\n===== Phase 2b, p={p}, K={K} =====")
    vi = Phase2bValueIteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, capacity=CAPACITY,
                                max_wait_per_floor=K, theta=1e-4)
    V2, policy2, history2 = vi.run(verbose=False, return_history=True)
    write_convergence_log(f"Results/CSV/log_phase2b_vi_realistic_p{p:.2f}_K{K}.csv", history2)
    env2_factory = lambda seed: ElevatorEnvV2(n_floors=N_FLOORS, p=p, capacity=CAPACITY,
                                               max_wait_per_floor=K, seed=seed)
    start2 = vi.start_state()
    results = {
        "phase2b_random": evaluate(
            label="phase2b_random", policy_fn_factory=make_v2_random_policy,
            env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
        ),
        "phase2b_vi_realistic": evaluate(
            label="phase2b_vi_realistic", policy_fn_factory=lambda seed: vi.make_table_policy(),
            env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
            n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
        ),
    }
    results["phase2b_random"]["V"] = ""  # keine Value Iteration fuer die Zufallsstrategie
    results["phase2b_vi_realistic"]["V"] = round(V2[start2], 4)
    print(f"  fertig: V*={V2[start2]:.2f}")

    for r in results.values():
        append_result(r)
    return results, history2


def run_phase3(p, K):
    """NUR Phase 3 (Boolean-Beobachtung), ausgewertet auf ElevatorEnvV2 mit
    gegebenem K (unabhaengig davon, ob Phase 2b fuer dasselbe K/p schon lief -
    die Naeherungs-Politik selbst haengt gar nicht von K ab, siehe PHASES.md).
    K beeinflusst hier NUR die reale Auswertungsumgebung (evaluate()), nicht
    die Value Iteration selbst (Phase3ValueIteration nimmt gar kein K entgegen).
    Gibt zusaetzlich die Konvergenz-Historie zurueck (fuer plot_convergence())."""
    print(f"\n===== Phase 3, p={p}, K={K} =====")
    env2_factory = lambda seed: ElevatorEnvV2(n_floors=N_FLOORS, p=p, capacity=CAPACITY,
                                               max_wait_per_floor=K, seed=seed)
    vi = Phase3ValueIteration(n_floors=N_FLOORS, p=p, gamma=GAMMA, theta=1e-4)
    V3, policy3, history3 = vi.run(verbose=False, return_history=True)
    write_convergence_log(f"Results/CSV/log_phase3_boolean_obs_p{p:.2f}_K{K}.csv", history3)
    result = evaluate(
        label="phase3_boolean_obs", policy_fn_factory=lambda seed: vi.make_table_policy(),
        env_factory=env2_factory, gamma=GAMMA, steps=EVAL_STEPS, n_episodes=EVAL_EPISODES,
        n_floors=N_FLOORS, p=p, capacity=CAPACITY, max_wait_per_floor=K,
    )
    start3 = vi.start_state()
    # V hier = Naeherungs-Wertfunktion (c_approx-basiert), NICHT mit V aus Phase 1/2/2b vergleichbar - siehe PHASES.md
    result["V"] = round(V3[start3], 4)
    print(f"  fertig: V*_approx={V3[start3]:.2f} (Naeherungs-Kosten, nicht mit V* aus Phase 1/2/2b vergleichbar)")
    append_result(result)
    return result, history3


def plot_phase2b_results(results_by_p, K, path=None):
    """Plottet avg_cost_per_step und avg_trip_duration (Random vs. VI) ueber p
    fuer Phase 2b bei festem K und speichert das Ergebnis als PNG in
    Results/Images/. avg_trip_duration ist None, wenn eine Politik bei diesem
    p komplett kollabiert (0 abgeschlossene Fahrten, siehe PHASES.md
    "Systemueberlastung") - wird als NaN geplottet, matplotlib zeigt das als
    Luecke in der Linie statt eines falschen Werts."""
    ps = sorted(results_by_p.keys())

    def series(label, metric):
        return [
            results_by_p[p][label][metric] if results_by_p[p][label][metric] is not None else float("nan")
            for p in ps
        ]

    fig, (ax_cost, ax_duration) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_cost.plot(ps, series("phase2b_random", "avg_cost_per_step"), marker="o", label="Random")
    ax_cost.plot(ps, series("phase2b_vi_realistic", "avg_cost_per_step"), marker="o", label="Value Iteration")
    ax_cost.set_xlabel("p (Ankunftswahrscheinlichkeit)")
    ax_cost.set_ylabel("avg_cost_per_step")
    ax_cost.set_title(f"Phase 2b: Kosten/Schritt (K={K})")
    ax_cost.legend()
    ax_cost.grid(True, alpha=0.3)

    ax_duration.plot(ps, series("phase2b_random", "avg_trip_duration"), marker="o", label="Random")
    ax_duration.plot(ps, series("phase2b_vi_realistic", "avg_trip_duration"), marker="o", label="Value Iteration")
    ax_duration.set_xlabel("p (Ankunftswahrscheinlichkeit)")
    ax_duration.set_ylabel("avg_trip_duration (Schritte)")
    ax_duration.set_title(f"Phase 2b: Fahrtdauer (K={K})")
    ax_duration.legend()
    ax_duration.grid(True, alpha=0.3)

    fig.tight_layout()
    if path is None:
        path = f"Results/Images/phase2b_p_variation_K{K}.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert nach: {path}")


def plot_convergence(histories_by_p, results_by_p, K, policy_label, title, y_label="V", path=None):
    """Konvergenz der Wertfunktion V (Value Iteration) fuer ALLE p in EINEM
    Plot, je p eine eigene Farbe: X-Achse = Iteration der Value Iteration (die
    "Zeitschritte" des Bellman-Updates, NICHT Simulationsschritte), Y-Achse =
    V(Startzustand) nach dieser Iteration (history-Eintraege aus
    vi.run(return_history=True), siehe
    Phases/value_iteration/base.py::ValueIterationBase.run()).

    Je p zusaetzlich eingezeichnet (gleiche Farbe, andere Linienart): das
    final konvergierte V* (gestrichelt) und avg_discounted_cost aus der
    Simulation (gepunktet) - der Unterschied zwischen beiden ist KEINE
    Ungenauigkeit der Value Iteration (die rechnet exakt), sondern eine
    Gegenprobe: V* ist exakt berechnet, avg_discounted_cost ist ein
    simulierter Schaetzwert ueber EVAL_EPISODES endliche Episoden - beide
    sollten nah beieinander liegen (siehe PHASES.md "Zwei getrennte Schritte:
    Lernen vs. Messen").

    policy_label: Schluessel in results_by_p[p], um an das evaluate()-Ergebnis
    heranzukommen (z.B. "phase2b_vi_realistic", das run_phase2b() als eines
    von mehreren Ergebnissen liefert). None, falls results_by_p[p] bereits
    direkt das evaluate()-Ergebnis ist (Phase 3, siehe run_phase3() - dort
    gibt es nur EINE Politik statt Random+VI)."""
    ps = sorted(histories_by_p.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(9, 6))
    diffs = []

    for i, p in enumerate(ps):
        color = colors[i % len(colors)]
        history = histories_by_p[p]
        iterations = [h[0] for h in history]
        v_values = [h[2] for h in history]
        v_star = v_values[-1]
        result = results_by_p[p][policy_label] if policy_label is not None else results_by_p[p]
        measured = result["avg_discounted_cost"]

        ax.plot(iterations, v_values, color=color, label=f"p={p}")
        ax.axhline(v_star, color=color, linestyle="--", alpha=0.6)
        if measured is not None:
            ax.axhline(measured, color=color, linestyle=":", alpha=0.6)
            diffs.append(f"p={p}: V*={v_star:.2f}, gemessen={measured:.2f} (Diff={measured - v_star:+.2f})")
        else:
            diffs.append(f"p={p}: V*={v_star:.2f}, gemessen=n/a")

    ax.set_xlabel("Iteration (Bellman-Update)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # zwei getrennte Legenden: Farbe = p (welche Kurve), Linienart = Bedeutung
    # (Konvergenzkurve/V*/gemessen) - sonst waeren es bei 4 p-Werten x 3
    # Linien pro p zu viele Eintraege fuer eine einzelne Legende
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], label=f"p={p}") for i, p in enumerate(ps)]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label=f"{y_label}(Startzustand) je Iteration"),
        Line2D([0], [0], color="black", linestyle="--", label=f"{y_label}* (konvergiert)"),
        Line2D([0], [0], color="black", linestyle=":", label="gemessen (avg_discounted_cost)"),
    ]
    legend_p = ax.legend(handles=color_handles, loc="lower right", title="p", fontsize=9)
    ax.add_artist(legend_p)
    ax.legend(handles=style_handles, loc="center right", fontsize=9)

    # Differenz V* vs. gemessen zusaetzlich als Text ablesbar (nicht nur visuell
    # als Luecke zwischen gestrichelter/gepunkteter Linie)
    ax.text(
        0.02, 0.98, "\n".join(diffs), transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    if path is None:
        path = f"Results/Images/convergence_K{K}.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert nach: {path}")


def plot_trip_completion(results_by_p, K, series, title_prefix, path=None):
    """Zwei Panels ueber p:
    links: avg_trip_duration - Dauer NUR der angekommenen (abgeschlossenen)
    Fahrten.
    rechts: completion_rate = n_completed_trips_total / n_total_arrivals -
    wie viele der insgesamt aufgetretenen (gefahrenen) Personen im Verhaeltnis
    tatsaechlich angekommen sind. Eine niedrige Fahrtdauer allein kann
    truegen, wenn nur wenige Personen ueberhaupt bedient werden (siehe
    PHASES.md "Die Erledigungsquote zeigt den Unterschied deutlicher als die
    reinen Kosten") - deshalb beide Kennzahlen nebeneinander.

    series: Liste von (Anzeigename, policy_key)-Paaren - policy_key ist der
    Schluessel in results_by_p[p], um an das jeweilige evaluate()-Ergebnis zu
    kommen (z.B. "phase2b_vi_realistic", Phase 2b hat Random+VI also 2
    Eintraege). policy_key ist None, falls results_by_p[p] bereits direkt das
    Ergebnis ist (Phase 3, siehe run_phase3() - dort gibt es nur EINE Politik,
    also nur 1 Eintrag in series)."""
    ps = sorted(results_by_p.keys())

    def series_values(policy_key, metric):
        vals = []
        for p in ps:
            result = results_by_p[p][policy_key] if policy_key is not None else results_by_p[p]
            v = result[metric]
            vals.append(v if v is not None else float("nan"))
        return vals

    fig, (ax_duration, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, policy_key in series:
        ax_duration.plot(ps, series_values(policy_key, "avg_trip_duration"), marker="o", label=label)
    ax_duration.set_xlabel("p (Ankunftswahrscheinlichkeit)")
    ax_duration.set_ylabel("avg_trip_duration (Schritte)")
    ax_duration.set_title(f"{title_prefix}: Fahrtdauer angekommener Fahrten (K={K})")
    ax_duration.legend()
    ax_duration.grid(True, alpha=0.3)

    for label, policy_key in series:
        ax_ratio.plot(ps, series_values(policy_key, "completion_rate"), marker="o", label=label)
    ax_ratio.set_xlabel("p (Ankunftswahrscheinlichkeit)")
    ax_ratio.set_ylabel("completion_rate (angekommen / gesamt gefahren)")
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_title(f"{title_prefix}: Erledigungsquote: angekommen im Verhaeltnis zu allen (K={K})")
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)

    fig.tight_layout()
    if path is None:
        path = f"Results/Images/trip_completion_K{K}.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert nach: {path}")


if __name__ == "__main__":
    # Phase 2b (K_VALUES) ist bereits vollstaendig gelaufen (siehe
    # Results/Images/phase2b_*_K2.png und _K3.png) - hier erstmal deaktiviert,
    # damit ein Lauf nicht jedes Mal wieder 30+ Minuten fuer eine bereits
    # vorhandene Kombination braucht. Einfach wieder einruecken/aktivieren,
    # um Phase 2b erneut (z.B. fuer ein neues K) laufen zu lassen.
    # for K in K_VALUES:
    #     all_results = {}
    #     all_histories = {}
    #     for p in P_VALUES:
    #         all_results[p], all_histories[p] = run_phase2b(p, K)
    #
    #     print(f"\n\n===== Zusammenfassung: avg_cost_per_step (Phase 2b, K={K}) =====")
    #     header = f"{'p':<6}{'phase2b_random':>16}{'phase2b_vi_realistic':>22}"
    #     print(header)
    #     for p in P_VALUES:
    #         row = f"{p:<6}"
    #         row += f"{all_results[p]['phase2b_random']['avg_cost_per_step']:>16}"
    #         row += f"{all_results[p]['phase2b_vi_realistic']['avg_cost_per_step']:>22}"
    #         print(row)
    #
    #     # Dateinamen enthalten K (siehe path=None-Default in den drei plot_*-
    #     # Funktionen) - jeder K-Wert bekommt also automatisch eigene PNGs,
    #     # nichts wird zwischen K=2 und K=3 ueberschrieben.
    #     plot_phase2b_results(all_results, K)
    #     plot_convergence(
    #         all_histories, all_results, K, "phase2b_vi_realistic",
    #         title=f"Phase 2b: Konvergenz der Wertfunktion V (K={K})",
    #         path=f"Results/Images/phase2b_convergence_K{K}.png",
    #     )
    #     plot_trip_completion(
    #         all_results, K,
    #         series=[("Random", "phase2b_random"), ("Value Iteration", "phase2b_vi_realistic")],
    #         title_prefix="Phase 2b",
    #         path=f"Results/Images/phase2b_trip_completion_K{K}.png",
    #     )

    # Phase 3 (Boolean-Beobachtung): dieselbe Konvergenz-/Fahrtdauer-Analyse wie
    # fuer Phase 2b, jetzt fuer Phase 3. Die Value Iteration selbst ist
    # K-unabhaengig (Phase3ValueIteration nimmt gar kein K entgegen, siehe
    # run_phase3()) - K beeinflusst hier NUR die reale Auswertungsumgebung
    # (ElevatorEnvV2). run_phase3() hat nur EINE Politik (kein Random-Vergleich
    # wie bei Phase 2b), daher policy_key=None in series/plot_convergence.
    K_VALUES_PHASE3 = [2, 3]
    for K3 in K_VALUES_PHASE3:
        all_results_p3 = {}
        all_histories_p3 = {}
        for p in P_VALUES:
            all_results_p3[p], all_histories_p3[p] = run_phase3(p, K3)

        plot_convergence(
            all_histories_p3, all_results_p3, K3, None,
            title=f"Phase 3: Konvergenz der Naeherungs-Wertfunktion V_approx (gemessen auf K={K3})",
            y_label="V_approx",
            path=f"Results/Images/phase3_convergence_K{K3}.png",
        )
        plot_trip_completion(
            all_results_p3, K3,
            series=[("Value Iteration (Boolean-Obs)", None)],
            title_prefix="Phase 3",
            path=f"Results/Images/phase3_trip_completion_K{K3}.png",
        )
