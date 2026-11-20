# GTA 6 Trophy Rarity Forecaster

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250).

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Arifpratama19/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Arifpratama19/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

`main.py` (Python 3.13, standard library only) reads a case `/input/<id>.in` and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each of the 55 trophies' earn rate at the target to `/output/<id>.out`. It models each trophy's earners and the players-tracked count from the case history, and forecasts honestly with a spread that grows with the horizon.
