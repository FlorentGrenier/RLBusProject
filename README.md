# bus-graph-rl

Prototype de reinforcement learning ou un bus evolue sur un graphe urbain.

Le projet contient aujourd'hui :
- un environnement OSM jouable sans notebook
- un agent Q-learning tabulaire pour une premiere boucle d'apprentissage
- une CLI simple pour lancer un entrainement

Le projet ne contient pas encore :
- une modelisation realiste du trafic ou des passagers
- une pipeline d'experiences complete
- une version finalisee de l'environnement grille

## Installation

Prerequis :
- Python 3.10 ou plus
- `uv`

```bash
pip install uv
uv sync
uv pip install -e .
```

## Lancer un entrainement

```bash
uv run python -m bus_graph_rl.cli.train_qlearning --episodes 50 --area Toulouse
```

## Environnement OSM

L'environnement `OSMBusEnv` charge un graphe OSM via `osmnx`, marque des arrets de bus aleatoires, puis genere une mission simple :
- aller au point de pickup
- effectuer le pickup
- rejoindre le point de dropoff
- effectuer le dropoff

L'espace d'actions fonctionne comme suit :
- action `0` : tenter de servir l'arret courant
- actions `1..N` : se deplacer vers un voisin sortant du noeud courant

L'observation expose :
- `passenger_on`
- `passenger_off`
- `current_node_is_stop`
- `distance_to_target`
- `action_mask`

## Structure

- `src/bus_graph_rl/envs/` : environnements RL
- `src/bus_graph_rl/agents/` : agents d'apprentissage
- `src/bus_graph_rl/graph/` : chargement des graphes
- `src/bus_graph_rl/cli/` : point d'entree entrainement
- `tests/` : smoke tests et tests unitaires de base
- `notebooks/` : travail exploratoire initial

## Etat du repo

Ce repo est une base de travail serieuse, mais encore experimentale.

Stable aujourd'hui :
- packaging Python simple
- chargement de graphe OSM
- boucle d'entrainement minimale
- tests unitaires de base hors reseau

Encore en chantier :
- meilleur shaping de reward
- observations plus riches pour apprendre sur graphe
- comparaison avec des methodes plus solides que le Q-learning tabulaire
- `GridBusEnv`, encore incomplet

## Tests

```bash
uv run pytest
```

## Prochaines evolutions utiles

- ajouter un rendu ou des traces de trajectoire pour debugguer les episodes
- persister des metriques d'entrainement
- introduire une vraie selection d'actions basee sur la structure locale du graphe
- preparer une transition vers un agent DQN, PPO, ou un modele de graphe
