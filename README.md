# bus-graph-rl

Projet d’apprentissage par renforcement (RL) : un **bus** qui explore un **graphe urbain** (réseau routier) , avec une première implémentation basée sur OSM.

## Démarrage rapide

### Installation & exécution (avec `uv`)

Ce projet utilise **uv** pour gérer l’environnement Python et les dépendances.

### Prérequis

* Python ≥ 3.10
* `uv` installé

```bash
pip install uv
```

### Initialiser l’environnement

À la racine du repo :

```bash
uv sync
```

Cette commande :

* crée automatiquement le virtualenv (`.venv/`)
* installe toutes les dépendances définies dans `pyproject.toml`
* utilise `uv.lock` si le fichier esr présent (reproductibilité)

### Installer le projet en mode editable

Indispensable pour que les imports fonctionnent correctement :

```bash
uv pip install -e .
```

### Lancer un entraînement (sans notebook)

```bash
uv run python -m bus_graph_rl.cli.train_qlearning \
  --episodes 50 \
  --area Toulouse
```

### Activer manuellement le virtualenv (optionnel)

Utilisation `uv run` (recommandé) ou activer le venv :

Linux / macOS :

```bash
source .venv/bin/activate
```

Windows (PowerShell) :

```powershell
.venv\Scripts\Activate.ps1
```

### Utilisation avec Jupyter Notebook

Pour utiliser cette env dans Jupyter :

```bash
uv pip install ipykernel
python -m ipykernel install --user --name bus-graph-rl
```

Puis sélectionner le kernel **bus-graph-rl** dans Jupyter.

### Vérification rapide

```bash
uv run python - << 'EOF'
from bus_graph_rl.envs.osm_bus_env import OSMBusEnv
print("Environment OK")
EOF
```

## Structure du projet

* `src/bus_graph_rl/` : code principal (envs, agents, training)
* `notebooks/` : notebooks exploratoires (avec les notebooks originaux)
* TODO `configs/` : configurations YAML (env, agent, training)
* TODO `runs/` : logs et checkpoints
* TODO `data/` : données brutes et pré‑traitées


## Notes importantes

Ce repo est volontairement **évolutif** :

* Q-learning tabulaire pour démarrer
* transition possible vers PPO / GNN (Avec PyTorch Geometric)
* reward shaping et observation space encore très expérimentaux
