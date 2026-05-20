# MiniGrid Navigation NanoVLM (SFT + GRPO)

## Описание проекта

В этом проекте адаптируется vision-language модель NanoVLM для управления агентом в среде MiniGrid EmptyEnv. Агент получает частичное RGB-наблюдение и должен дойти до зелёной клетки цели, выбирая одно из трёх действий: `left`, `right`, `forward`.

Обучение проводится в два этапа:

1. **SFT (Supervised Fine-Tuning)** на экспертных траекториях.
2. **GRPO (Group Relative Policy Optimization)** для дообучения через RL.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![MiniGrid](https://img.shields.io/badge/MiniGrid-RL%20Environment-green)
![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-orange?logo=weightsandbiases)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-GRPO-purple)

## Оглавление

1. [Генерация экспертных траекторий](#генерация-экспертных-траекторий)
2. [Используемая модель](#используемая-модель)
3. [SFT обучение](#sft-обучение)
4. [GRPO обучение](#grpo-обучение)
5. [Результаты](#результаты)
6. [Логирование](#логирование)
7. [Запуск проекта](#запуск-проекта)
8. [Структура проекта](#структура-проекта)
9. [Возможные улучшения](#возможные-улучшения)
10. [Итог](#итог)

## Генерация экспертных траекторий

Для создания экспертных траекторий используется BFS (Breadth-First Search), который находит кратчайший путь до цели. Состояние в BFS включает не только позицию агента, но и направление взгляда: `(agent_x, agent_y, agent_dir)`. Это важно, потому что действия `left` и `right` меняют ориентацию агента, а не позицию.

Экспертный датасет содержит пары:

- RGB-наблюдение агента;
- текстовый prompt;
- экспертное следующее действие;
- `episode_id`, `step`, `env_size`, позицию и направление агента.

В текущей версии генератор сравнивает left-first и right-first shortest paths и выбирает вариант, который уменьшает накопленный дисбаланс между `left` и `right`. Это убирает искусственный перекос, который появлялся из-за фиксированного порядка обхода BFS.

Датасеты:

| Environment | Path | Episodes | Rows | Action distribution |
|---|---|---:|---:|---|
| 8x8 | `datasets/dataset_8x8` | 1000 | 5280 | `forward=3914`, `left=686`, `right=680` |
| 16x16 | `datasets/dataset_16x16` | 1000 | 10530 | `forward=9105`, `left=726`, `right=699` |

## Используемая модель

В проекте используется **NanoVLM v0.1**:

🔗 [https://github.com/huggingface/nanoVLM/releases/tag/v0.1](https://github.com/huggingface/nanoVLM/releases/tag/v0.1)

## SFT обучение

SFT-бейзлайн обучает NanoVLM предсказывать следующее действие эксперта по текущему наблюдению. Формат входа, где `{prompt}` заменяется текстом ниже:

```text
User: <image>
{prompt}
Assistant:
```

Текст промпта:

```text
You are a robot in a 2D grid world. You see a 7x7 partial RGB view in front of you.
Your mission: get to the green goal square as quickly as possible.
Choose the next action: forward, left or right.
```

Целевой ответ:

```text
 left
 right
 forward
```

Prompt tokens в SFT loss маскируются, поэтому модель обучается на assistant action, а не на воспроизведении всего prompt.

Train/validation split выполняется на уровне эпизодов, а не отдельных строк. Дополнительно split подбирается так, чтобы proportions действий в validation были близки к полному датасету.

Почему validation accuracy не является основной метрикой:

- действие `forward` доминирует в датасете;
- majority baseline может иметь высокую offline accuracy;
- качество политики лучше измерять в среде через success rate, average reward, timeouts и длину успешной траектории.

## GRPO обучение

После SFT модель дообучалась через GRPO-style RL loop. Политика инициализируется из SFT adapter, затем запускаются группы rollout-ов в MiniGrid. Для каждой группы считается group-relative advantage, после чего выполняется clipped update с KL-штрафом к reference SFT policy.

Текущая реализация:

- использует прямой вывод действия (`left`, `right`, `forward`);
- сохраняет checkpoints каждые 25 GRPO episodes;
- использует SFT policy как reference model;
- считает приближённый sampled KL для выбранного действия.

## Результаты

### Основной эксперимент: 8x8

На `MiniGrid-Empty-8x8-v0` SFT очень хорошо решает задачу, а GRPO даёт дополнительное улучшение. Финальная оценка проводилась на 250 эпизодах.

Команда:

```powershell
python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8 --grpo-adapter-path checkpoints/grpo_adapter --episodes 250
```

Dataset baseline:

- train rows: `4735`, validation rows: `545`;
- majority action: `forward`;
- majority validation accuracy: `0.7413`;
- train action distribution: `left=615`, `right=610`, `forward=3510`;
- validation action distribution: `left=71`, `right=70`, `forward=404`.

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Action distribution |
|---|---:|---:|---:|---:|---|
| Majority-forward | 5.2% | 0.052 | 2.2 | 237/250 | L:0.0% / R:0.0% / F:100.0% |
| SFT | 91.2% | 0.892 | 6.1 | 22/250 | L:17.0% / R:15.5% / F:67.5% |
| GRPO | 95.2% | 0.932 | 6.1 | 12/250 | L:17.3% / R:14.9% / F:67.8% |
| Expert BFS | 100.0% | 0.982 | 5.2 | 0/250 | L:13.4% / R:13.4% / F:73.2% |

Вывод: для 8x8 пайплайн `expert trajectories -> SFT -> GRPO -> environment evaluation` работает корректно. GRPO уменьшает число timeout-ов с `22/250` до `12/250` и повышает success rate на `+4.0` процентных пункта относительно SFT.

### Stress-test: 16x16

На `MiniGrid-Empty-16x16-v0` задача заметно сложнее. Финальная оценка также проводилась на 250 эпизодах.

Команда:

```powershell
python scripts/test_models.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16 --grpo-adapter-path checkpoints/grpo_adapter_16x16 --episodes 250 --max-steps 40 --val-split 0.01
```

Dataset baseline:

- train rows: `10427`, validation rows: `103`;
- majority action: `forward`;
- majority validation accuracy: `0.8641`;
- train action distribution: `left=719`, `right=692`, `forward=9016`;
- validation action distribution: `left=7`, `right=7`, `forward=89`.

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Action distribution |
|---|---:|---:|---:|---:|---|
| Majority-forward | 3.2% | 0.032 | 4.1 | 242/250 | L:0.0% / R:0.0% / F:100.0% |
| SFT | 43.2% | 0.424 | 20.7 | 142/250 | L:12.0% / R:4.3% / F:83.8% |
| GRPO | 58.4% | 0.573 | 22.0 | 104/250 | L:13.3% / R:3.5% / F:83.2% |
| Expert BFS | 100.0% | 0.991 | 10.6 | 0/250 | L:6.8% / R:6.9% / F:86.3% |

Вывод:  В среде 16x16 GRPO повышает success rate на `+15.2` процентных пункта относительно SFT, но сохраняет большое число timeout-ов (`104/250`) и остаётся значительно ниже expert BFS.

## Логирование

Во время обучения используется Weights & Biases (`wandb`):

- логируются train loss, validation accuracy, mean return и success rate;
- сохраняются кривые обучения;
- checkpoints сохраняются локально в `checkpoints/`.

Если `wandb` недоступен или не нужен, используйте флаг:

```powershell
--no-wandb
```

## Запуск проекта

### Установка

1. Клонируйте репозиторий.
2. Установите зависимости:

```powershell
pip install -r requirements.txt
```

3. Скачайте или клонируйте NanoVLM: https://github.com/huggingface/nanoVLM. Поместите папку в директорию проекта и переименуйте её в `nanoVLM`.

### Быстрый запуск evaluation

Для 8x8:

```powershell
python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8 --grpo-adapter-path checkpoints/grpo_adapter --episodes 250
```

Для 16x16:

```powershell
python scripts/test_models.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16 --grpo-adapter-path checkpoints/grpo_adapter_16x16 --episodes 250 --max-steps 40 --val-split 0.01
```

### Обучение 8x8 с нуля

```powershell
python scripts/dataset_generation.py --env-size 8 --save-path datasets/dataset_8x8
python scripts/sft.py --env-size 8 --dataset-path datasets/dataset_8x8 --output-dir checkpoints/sft_adapter_8x8 --epochs 3
python scripts/grpo.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8 --output-dir checkpoints/grpo_adapter_8x8
python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8 --grpo-adapter-path checkpoints/grpo_adapter_8x8 --episodes 250
```

### Обучение 16x16

```powershell
python scripts/dataset_generation.py --env-size 16 --save-path datasets/dataset_16x16
python scripts/sft.py --env-size 16 --dataset-path datasets/dataset_16x16 --output-dir checkpoints/sft_adapter_16x16 --epochs 1 --val-split 0.01
python scripts/grpo.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16 --output-dir checkpoints/grpo_adapter_16x16 --max-steps 35 --val-split 0.01
python scripts/test_models.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16 --grpo-adapter-path checkpoints/grpo_adapter_16x16 --episodes 250 --max-steps 40 --val-split 0.01
```

## Структура проекта

```text
├── checkpoints/
│   ├── sft_adapter_8x8/          # SFT LoRA adapter для 8x8
│   ├── grpo_adapter_8x8/         # GRPO LoRA adapter для 8x8
│   ├── sft_adapter_16x16/        # SFT checkpoints для 16x16
│   └── grpo_adapter_16x16/       # GRPO LoRA adapter для 16x16
├── datasets/
│   ├── dataset_8x8/              # экспертный датасет для 8x8
│   └── dataset_16x16/            # экспертный датасет для 16x16
├── nanoVLM/                      # репозиторий NanoVLM
├── notebooks/                    # exploratory notebooks
├── scripts/
│   ├── _bootstrap.py             # настройка import paths для scripts/
│   ├── dataset_generation.py     # генерация экспертных траекторий
│   ├── sft.py                    # supervised fine-tuning
│   ├── grpo.py                   # RL fine-tuning
│   └── test_models.py            # тестирование и оценка моделей
├── src/
│   └── vlm_minigrid_rl/
│       ├── minigrid_utils.py     # MiniGrid reset, BFS expert, environment metrics
│       ├── model_utils.py        # NanoVLM loading, preprocessing, inference, scoring
│       ├── paths.py              # project paths и NanoVLM path setup
│       └── training_utils.py     # seed, split, baselines, action parsing
├── README.md
└── requirements.txt
```

## Возможные улучшения

- Curriculum learning: начать с 8x8 и постепенно увеличивать размер карты.
- Более точный подбор параметров для SFT и GRPO.
- Более надёжная evaluation: несколько seed ranges и доверительные интервалы.
- Реализация и анализ `text+action` формата.

## Итог

В текущем состоянии проекта удалось:

- сгенерировать экспертные BFS trajectories;
- обучить SFT baseline для прямого выбора действий;
- реализовать GRPO fine-tuning;
- добавить majority и expert baselines;
- показать, что на 8x8 SFT достигает `91.2%` success rate, а GRPO повышает результат до `95.2%`;
- показать, что перенос на 16x16 резко усложняет задачу: SFT достигает `43.2%`, GRPO - `58.4%`, expert BFS - `100.0%`.
