# MiniGrid Navigation NanoVLM (SFT + GRPO)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![MiniGrid](https://img.shields.io/badge/MiniGrid-RL%20Environment-green)
![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-orange?logo=weightsandbiases)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-GRPO-purple)

## Описание проекта

В проекте адаптируется vision-language модель NanoVLM для управления агентом в среде MiniGrid EmptyEnv. Агент получает частичное RGB-наблюдение 7x7 клеток и должен выбрать одно из трёх действий: `left`, `right` или `forward`.

Обучение проводится в два этапа:

1. **SFT (Supervised Fine-Tuning)** на экспертных траекториях, построенных BFS-планировщиком.
2. **GRPO-style RL fine-tuning** для дообучения политики через взаимодействие со средой.

Основные среды:

- `MiniGrid-Empty-8x8-v0` — базовая среда, где pipeline уже близок к потолку.
- `MiniGrid-Empty-16x16-v0` — более сложная среда с длинными траекториями и большим числом состояний.

Дополнительно проверяются два свойства политики:

- перенос между размерами карты (`8x8 -> 16x16` и `16x16 -> 8x8`);
- устойчивость к изменению цвета цели с зелёного на красный.

Общая схема pipeline:

![Схема pipeline обучения](docs/figures/architecture/training_pipeline.png)

## Оглавление

1. [Данные и эксперт](#данные-и-эксперт)
2. [Модель и обучение](#модель-и-обучение)
3. [Результаты](#результаты)
4. [Дополнительные эксперименты](#дополнительные-эксперименты)
5. [Запуск проекта](#запуск-проекта)
6. [Структура проекта](#структура-проекта)
7. [Выводы и дальнейшая работа](#выводы-и-дальнейшая-работа)

## Данные и эксперт

Экспертные траектории генерируются с помощью BFS (Breadth-First Search). Внутри BFS состояние включает координаты и направление агента `(x, y, agent_dir)`; `left` и `right` меняют ориентацию, позицию меняет только `forward`.

Для уменьшения искусственного дисбаланса поворотов генератор сравнивает left-first и right-first shortest paths и выбирает путь, который уменьшает накопленный дисбаланс между `left` и `right`.

Каждый пример датасета содержит:

- `ego_image` — частичное RGB-наблюдение агента;
- `global_image` — полный вид среды;
- текстовый промпт;
- экспертное действие (`action`, `action_id`);
- `episode_id`, `step`, `env_size`;
- `agent_pos` — позиция агента `(x, y)` на сетке;
- `agent_dir` — направление агента (`0`–`3`).

SFT и GRPO используют только `ego_image`, `prompt` и `action`; поля `agent_pos` и `agent_dir` нужны для аудита траекторий и анализа датасета.

Датасеты:

| Environment | Path | Episodes | Rows | Action distribution |
|---|---|---:|---:|---|
| 8x8 | `datasets/dataset_8x8` | 1000 | 5280 | `forward=3914`, `left=686`, `right=680` |
| 16x16 | `datasets/dataset_16x16` | 1000 | 10530 | `forward=9105`, `left=726`, `right=699` |

## Модель и обучение

В проекте используется NanoVLM v0.1:

https://github.com/huggingface/nanoVLM/releases/tag/v0.1

Формат входа:

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

Целевой ответ (leading space — отдельный tokenizer id):

```text
 left
 right
 forward
```

В SFT prompt tokens маскируются, поэтому loss считается по assistant action, а не по воспроизведению всего промпта. Train/validation split выполняется на уровне эпизодов, чтобы шаги одного эпизода не попадали одновременно в train и validation.

**Текущий стек обучения (bs32 / AdamW):**

- оптимизатор: `torch.optim.AdamW` (и для SFT, и для GRPO);
- SFT по умолчанию: `BATCH_SIZE=32`, `GRAD_ACCUM=1`;
- GRPO: rollout = `generate_action()` (как eval); PPO/KL scoring — 3-way softmax по ` left` / ` right` / ` forward`; update батчится (`--update-batch-size`, default 32);
- invalid action token в rollout/eval сразу завершает эпизод (без fallback на `forward`).

Validation accuracy используется только как вспомогательная offline-метрика. Основная оценка проводится в среде через `success rate`, `average reward`, `timeouts`, `invalid_action_episodes` и среднюю длину успешной траектории. Для native suites предпочтительны **multi-seed** env eval (42 / 123 / 456); transfer и goal-color — seed 42.

### Протокол environment evaluation

Политика выбирает действие через `model.generate(..., max_new_tokens=1)`. Сгенерированный token id сопоставляется с ` left` / ` right` / ` forward`.

- Если токен распознан — действие выполняется в MiniGrid.
- Если токен не является одним из трёх action-токенов — эпизод завершается с ошибкой (`invalid_action_episodes`).

Эпизод также может завершиться успехом (достижение цели) или timeout-ом (исчерпан лимит шагов).

После SFT модель дообучается через GRPO-style RL loop. Политика инициализируется из SFT adapter, затем запускаются группы rollout-ов в MiniGrid. Для каждой группы считается group-relative advantage, после чего выполняется clipped update с KL-штрафом к reference SFT policy.

Сравнение этапов SFT и GRPO:

![Сравнение SFT и GRPO](docs/figures/architecture/sft_vs_grpo.png)

## Результаты

Environment evaluation: `generate`, 250 episodes, seeds **42 / 123 / 456**. Команды обучения и evaluation — в разделе [Запуск проекта](#запуск-проекта).

Headline checkpoint-ы:

| Env | Best SFT | Best GRPO |
|---|---|---|
| 8x8 | `checkpoints/sft_adapter_8x8_bs32/epoch-3` | `checkpoints/grpo_adapter_8x8_from_bs32_sft3/episode-75` |
| 16x16 | `checkpoints/sft_adapter_16x16_bs32/epoch-3` | `checkpoints/grpo_adapter_16x16_from_bs32_sft3/episode-100` |

### Среда 8x8

На `MiniGrid-Empty-8x8-v0` SFT уже даёт высокое качество, а GRPO дополнительно уменьшает число timeout-ов.

Полный вид среды и частичное наблюдение агента:

| Full view | Agent view |
|---|---|
| ![8x8 full view](docs/figures/8x8/global_image.png) | ![8x8 agent view](docs/figures/8x8/ego_image.png) |

Графики обучения:

| SFT train loss | SFT validation accuracy |
|---|---|
| ![SFT 8x8 train loss](docs/figures/8x8/sft_8x8_loss.png) | ![SFT 8x8 validation accuracy](docs/figures/8x8/sft_8x8_val_accuracy.png) |

| GRPO loss | GRPO success rate |
|---|---|
| ![GRPO 8x8 loss](docs/figures/8x8/grpo_8x8_loss.png) | ![GRPO 8x8 success rate](docs/figures/8x8/grpo_8x8_success_rate.png) |

SFT 8x8 обучается стабильно: train loss быстро падает почти до нуля, а validation accuracy растёт. GRPO 8x8 имеет высокий success rate почти на всём протяжении обучения; отдельные провалы связаны с тем, что метрика считается на небольших группах rollout-ов и чувствительна к конкретным seed-ам.

Статистика split-а:

- train rows: `4735`, validation rows: `545`;
- majority action: `forward`;
- majority validation accuracy: `0.7413`;
- train action distribution: `left=615`, `right=610`, `forward=3510`;
- validation action distribution: `left=71`, `right=70`, `forward=404`.

Результаты тестирования (seed 42):

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Invalid actions | Action distribution |
|---|---:|---:|---:|---:|---:|---|
| Majority-forward | 5.2% | 0.052 | 2.2 | 237/250 | 0/250 | L:0.0% / R:0.0% / F:100.0% |
| Base NanoVLM | 0.8% | 0.008 | 1.5 | 0/250 | 248/250 | L:51.0% / R:12.1% / F:36.9% |
| SFT epoch-3 | 91.6% | 0.898 | 5.6 | 17/250 | 4/250 | L:17.0% / R:11.9% / F:71.1% |
| GRPO episode-75 | 98.0% | 0.961 | 5.4 | 3/250 | 2/250 | L:19.2% / R:11.6% / F:69.2% |
| Expert BFS | 100.0% | 0.982 | 5.2 | 0/250 | 0/250 | L:13.4% / R:13.4% / F:73.2% |

Success rate по seed (native 8x8, только SFT / GRPO):

| seed | SFT epoch-3 | GRPO episode-75 | Δ (GRPO − SFT) |
|---:|---:|---:|---:|
| 42 | 91.6% | 98.0% | +6.4 |
| 123 | 93.6% | 98.0% | +4.4 |
| 456 | 95.2% | 97.2% | +2.0 |
| **mean** | **93.47%** | **97.73%** | **+4.26** |

Вывод: чистая NanoVLM без дообучения почти не решает задачу (`0.8%`, `248/250` invalid actions на seed 42). SFT поднимает mean success до `93.47%`. GRPO даёт mean `97.73%` (+4.26 п.п. к SFT). Лучший GRPO checkpoint — `episode-75`, не финальный `episode-100`.

### Среда 16x16

`MiniGrid-Empty-16x16-v0` сложнее из-за более длинных траекторий. Ошибки действий накапливаются сильнее, поэтому высокая offline accuracy хуже отражает реальное качество политики.

Полный вид среды и частичное наблюдение агента:

| Full view | Agent view |
|---|---|
| ![16x16 full view](docs/figures/16x16/global_image.png) | ![16x16 agent view](docs/figures/16x16/ego_image.png) |

Графики обучения:

| SFT train loss | SFT validation accuracy |
|---|---|
| ![SFT 16x16 train loss](docs/figures/16x16/sft_16x16_loss.png) | ![SFT 16x16 validation accuracy](docs/figures/16x16/sft_16x16_val_accuracy.png) |

| GRPO loss | GRPO success rate |
|---|---|
| ![GRPO 16x16 loss](docs/figures/16x16/grpo_16x16_loss.png) | ![GRPO 16x16 success rate](docs/figures/16x16/grpo_16x16_success_rate.png) |

Для SFT 16x16 validation accuracy растёт, но train loss заметно шумит. GRPO 16x16 также нестабилен: success rate сильно колеблется. Это ожидаемо для более длинной среды с более разреженным reward.

Статистика split-а:

- train rows: `10427`, validation rows: `103`;
- majority action: `forward`;
- majority validation accuracy: `0.8641`;
- train action distribution: `left=719`, `right=692`, `forward=9016`;
- validation action distribution: `left=7`, `right=7`, `forward=89`.

Результаты тестирования (seed 42):

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Invalid actions | Action distribution |
|---|---:|---:|---:|---:|---:|---|
| Majority-forward | 3.2% | 0.032 | 4.1 | 242/250 | 0/250 | L:0.0% / R:0.0% / F:100.0% |
| Base NanoVLM | 0.4% | 0.004 | 1.0 | 0/250 | 249/250 | L:48.9% / R:11.1% / F:40.0% |
| SFT epoch-3 | 77.6% | 0.764 | 17.3 | 44/250 | 12/250 | L:11.2% / R:6.4% / F:82.5% |
| GRPO episode-100 | 84.0% | 0.828 | 16.9 | 33/250 | 7/250 | L:12.6% / R:9.0% / F:78.4% |
| Expert BFS | 100.0% | 0.991 | 10.6 | 0/250 | 0/250 | L:6.8% / R:6.9% / F:86.3% |

Success rate по seed (native 16x16, только SFT / GRPO):

| seed | SFT epoch-3 | GRPO episode-100 | Δ (GRPO − SFT) |
|---:|---:|---:|---:|
| 42 | 77.6% | 84.0% | +6.4 |
| 123 | 75.2% | 81.2% | +6.0 |
| 456 | 81.6% | 84.8% | +3.2 |
| **mean** | **78.13%** | **83.33%** | **+5.20** |

Вывод: на 16x16 метрики зависят от seed (SFT: `75.2–81.6%`). GRPO даёт mean `83.33%` (+5.20 п.п. к SFT). Лучший checkpoint — `episode-100`. Параметры GRPO (`lr=5e-6`, `epsilon=0.1`, `beta=0.1`, `lora_dropout=0.0`) совпадают с 8x8 для сравнения стека; их нельзя автоматически считать оптимальными для каждой среды.

## Дополнительные эксперименты

### Перенос между средами

Проверялось, насколько политика, обученная на одном размере среды, переносится на другой. В этих экспериментах `dataset-path` соответствует тестовой среде, а adapter path — среде, на которой модель была обучена.

#### 8x8 -> 16x16

seed 42, 250 episodes. Adapters: SFT/GRPO, обученные на 8x8 (`epoch-3` / `episode-75`).

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Invalid actions | Action distribution |
|---|---:|---:|---:|---:|---:|---|
| Majority-forward | 3.2% | 0.032 | 4.1 | 242/250 | 0/250 | L:0.0% / R:0.0% / F:100.0% |
| SFT trained on 8x8 | 56.4% | 0.558 | 11.6 | 97/250 | 12/250 | L:42.9% / R:17.6% / F:39.5% |
| GRPO trained on 8x8 | 44.8% | 0.444 | 9.3 | 132/250 | 6/250 | L:66.8% / R:18.1% / F:15.1% |
| Expert BFS | 100.0% | 0.991 | 10.6 | 0/250 | 0/250 | L:6.8% / R:6.9% / F:86.3% |

SFT, обученная на 8x8, частично переносится на 16x16 и достигает `56.4%` success rate, что значительно выше majority baseline, но ниже результата SFT, обученной непосредственно на 16x16 (mean `78.13%`). GRPO, обученная на 8x8, переносится хуже SFT (`44.8%` против `56.4%`) и сильно смещается в сторону поворотов: доля `forward` падает до `15.1%`.

#### 16x16 -> 8x8

seed 42, 250 episodes. Adapters: SFT/GRPO, обученные на 16x16 (`epoch-3` / `episode-100`).

| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Invalid actions | Action distribution |
|---|---:|---:|---:|---:|---:|---|
| Majority-forward | 5.2% | 0.052 | 2.2 | 237/250 | 0/250 | L:0.0% / R:0.0% / F:100.0% |
| SFT trained on 16x16 | 94.0% | 0.920 | 5.9 | 11/250 | 4/250 | L:16.8% / R:12.5% / F:70.7% |
| GRPO trained on 16x16 | 96.0% | 0.940 | 5.8 | 6/250 | 4/250 | L:17.0% / R:12.5% / F:70.5% |
| Expert BFS | 100.0% | 0.982 | 5.2 | 0/250 | 0/250 | L:13.4% / R:13.4% / F:73.2% |

Модели, обученные на 16x16, хорошо переносятся на 8x8. SFT достигает `94.0%`, а GRPO повышает результат до `96.0%`.

Сводная таблица (native — mean по 3 seed; transfer — seed 42):

| Train env | Test env | SFT success | GRPO success | Вывод |
|---|---|---:|---:|---|
| 8x8 | 8x8 | 93.47% | 97.73% | среда простая, GRPO даёт стабильный прирост |
| 8x8 | 16x16 | 56.4% | 44.8% | перенос есть, но GRPO ухудшает (seed 42) |
| 16x16 | 16x16 | 78.13% | 83.33% | сложная среда, прирост GRPO заметный |
| 16x16 | 8x8 | 94.0% | 96.0% | перенос сильный, GRPO помогает (seed 42) |

Transfer и goal-color — только seed 42.

Вывод: перенос асимметричен. Обучение на 16x16 даёт политику, которая хорошо переносится на более короткую 8x8 среду, а GRPO дополнительно улучшает этот перенос. Обратное направление слабее: модели, обученные на 8x8, хуже работают на 16x16, а GRPO-донастройка под короткий горизонт дополнительно ухудшает перенос.

### Изменение цвета цели

seed 42, 250 episodes. Adapters: 8x8 SFT/GRPO (`epoch-3` / `episode-75`). Команды — в [Запуск проекта](#запуск-проекта).

Дополнительно проверялась устойчивость политики к изменению цвета цели: модели, обученные на зелёной цели, тестировались на красной. Это показывает, выучила ли модель обобщённое поведение навигации к цели или опирается на конкретный цветовой паттерн.

Два параметра задаются независимо:

- `--goal-color` - фактический цвет цели в MiniGrid;
- `--prompt-goal-color` - цвет цели, указанный в промпте модели.

| Visual goal color | Prompt color | Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts | Invalid actions | Action distribution |
|---|---|---|---:|---:|---:|---:|---:|---|
| green | green | SFT | 91.6% | 0.898 | 5.6 | 17/250 | 4/250 | L:17.0% / R:11.9% / F:71.1% |
| green | green | GRPO | 98.0% | 0.961 | 5.4 | 3/250 | 2/250 | L:19.2% / R:11.6% / F:69.2% |
| red | red | SFT | 40.4% | 0.394 | 7.4 | 143/250 | 6/250 | L:38.0% / R:22.3% / F:39.8% |
| red | red | GRPO | 8.0% | 0.078 | 7.2 | 220/250 | 10/250 | L:54.6% / R:28.8% / F:16.6% |
| red | green | SFT | 44.8% | 0.437 | 6.7 | 134/250 | 4/250 | L:32.0% / R:21.4% / F:46.6% |
| red | green | GRPO | 9.6% | 0.094 | 7.2 | 221/250 | 5/250 | L:52.7% / R:28.7% / F:18.7% |

Вывод: SFT частично переносит навигацию на красную цель, но качество всё равно падает с `91.6%` до `40.4–44.8%`. GRPO резко теряет устойчивость: несмотря на `98.0%` в green/green, на красной цели результат падает до `8.0–9.6%`, а доля поворотов растёт. Модель в значительной степени опирается на визуальный паттерн зелёной клетки, а не на абстрактное понятие цели.

## Запуск проекта

### Установка

1. Клонируйте репозиторий.
2. Установите зависимости:

```powershell
pip install -r requirements.txt
```

3. Скачайте [NanoVLM](https://github.com/huggingface/nanoVLM/releases/tag/v0.1) и поместите папку в корень проекта под именем `nanoVLM`.

Если `wandb` недоступен или не нужен, добавьте `--no-wandb` к командам обучения.

Горизонты Empty (не смешивать со старыми командами `--max-steps 12` / `40`): GRPO train = `L_max` (8×8: 12, 16×16: 28), env eval = `L_max + max(4, p95 steps-to-see)` (8×8: 16, 16×16: 38). Таблицы выше измерены на старом едином горизонте.

### Датасеты

```powershell
python scripts/dataset_generation.py --env-size 8 --save-path datasets/dataset_8x8
python scripts/dataset_generation.py --env-size 16 --save-path datasets/dataset_16x16
```

### Обучение и evaluation: 8x8

```powershell
python scripts/sft.py --env-size 8 --dataset-path datasets/dataset_8x8 --output-dir checkpoints/sft_adapter_8x8_bs32 --epochs 3

python scripts/grpo.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8_bs32/epoch-3 --output-dir checkpoints/grpo_adapter_8x8_from_bs32_sft3 --lr 5e-6 --epsilon 0.1 --beta 0.1 --lora-dropout 0.0

python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_8x8_from_bs32_sft3/episode-75 --episodes 250
```

### Обучение и evaluation: 16x16

```powershell
python scripts/sft.py --env-size 16 --dataset-path datasets/dataset_16x16 --output-dir checkpoints/sft_adapter_16x16_bs32 --epochs 3 --val-split 0.01

python scripts/grpo.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16_bs32/epoch-3 --output-dir checkpoints/grpo_adapter_16x16_from_bs32_sft3 --val-split 0.01 --lr 5e-6 --epsilon 0.1 --beta 0.1 --lora-dropout 0.0

python scripts/test_models.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_16x16_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_16x16_from_bs32_sft3/episode-100 --episodes 250 --val-split 0.01
```

### Transfer

`dataset-path` / `--env-size` — тестовая среда; adapters — среда обучения.

```powershell
# 8x8 -> 16x16
python scripts/test_models.py --env-size 16 --dataset-path datasets/dataset_16x16 --sft-adapter-path checkpoints/sft_adapter_8x8_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_8x8_from_bs32_sft3/episode-75 --episodes 250 --val-split 0.01

# 16x16 -> 8x8
python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_16x16_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_16x16_from_bs32_sft3/episode-100 --episodes 250 --val-split 0.1
```

### Goal-color (8x8)

```powershell
python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_8x8_from_bs32_sft3/episode-75 --episodes 250 --goal-color red --prompt-goal-color red

python scripts/test_models.py --env-size 8 --dataset-path datasets/dataset_8x8 --sft-adapter-path checkpoints/sft_adapter_8x8_bs32/epoch-3 --grpo-adapter-path checkpoints/grpo_adapter_8x8_from_bs32_sft3/episode-75 --episodes 250 --goal-color red --prompt-goal-color green
```

## Структура проекта

```text
vlm-minigrid-rl/
├── checkpoints/
│   ├── sft_adapter_8x8_bs32/              # SFT LoRA adapter для 8x8 (bs32, headline)
│   ├── grpo_adapter_8x8_from_bs32_sft3/   # GRPO LoRA для 8x8 (headline: episode-75)
│   ├── sft_adapter_16x16_bs32/            # SFT LoRA adapter для 16x16 (bs32, headline)
│   ├── grpo_adapter_16x16_from_bs32_sft3/ # GRPO LoRA для 16x16 (headline: episode-100)
│   ├── sft_adapter_8x8/                   # historical SFT 8x8
│   ├── grpo_adapter_8x8/                  # historical GRPO LoRA (pre-generate protocol)
│   ├── grpo_adapter_8x8_generate/         # historical generate GRPO 8x8
│   ├── sft_adapter_16x16/                 # historical SFT 16x16
│   ├── grpo_adapter_16x16/                # historical GRPO LoRA (pre-generate protocol)
│   └── grpo_adapter_16x16_generate/       # historical generate GRPO 16x16
├── datasets/
│   ├── dataset_8x8/              # экспертный датасет для 8x8
│   └── dataset_16x16/            # экспертный датасет для 16x16
├── docs/
│   └── figures/                  # графики обучения и примеры изображений среды
├── nanoVLM/                      # репозиторий NanoVLM
├── notebooks/
│   └── quick_test.ipynb          # offline check SFT/GRPO vs expert
├── scripts/
│   ├── _bootstrap.py             # настройка import paths для scripts/
│   ├── dataset_generation.py     # генерация экспертных траекторий
│   ├── sft.py                    # supervised fine-tuning
│   ├── grpo.py                   # RL fine-tuning
│   ├── run_tests.py              # orchestration eval suites + multi-seed compare
│   └── test_models.py            # тестирование и оценка моделей
├── src/
│   └── vlm_minigrid_rl/
│       ├── experiment_config.py  # пути и defaults экспериментов
│       ├── minigrid_utils.py     # MiniGrid reset, BFS expert, environment metrics
│       ├── model_utils.py        # NanoVLM loading, preprocessing, inference, scoring
│       ├── paths.py              # project paths и NanoVLM path setup
│       └── training_utils.py     # seed, split, baselines, action parsing
├── README.md
└── requirements.txt
```

## Выводы и дальнейшая работа

В текущем состоянии проекта удалось:

- сгенерировать экспертные BFS trajectories;
- обучить SFT baseline для прямого выбора действий;
- реализовать GRPO fine-tuning;
- получить на 8x8 SFT mean **93.47%** и GRPO mean **97.73%** (+4.26 п.п.);
- получить на 16x16 SFT mean **78.13%** и GRPO mean **83.33%** (+5.20 п.п.);
- показать асимметричный перенос между 8x8 и 16x16;
- показать слабую устойчивость к изменению цвета цели.

Дальнейшие направления исследования:

- **Поэтапное обучение и обучение на смешанных средах разного размера** - обучать модель последовательно на средах разного размера или на объединённом датасете для улучшения устойчивости.
- **Prompt engineering** - сравнение разных промптов.
- **Формат `text+action` и Chain-of-Thought** - генерация краткого описания видимой среды или плана перед выбором итогового действия.
- **VLA-подход (Vision-Language-Action)** - добавление отдельной головы выбора действия вместо генерации действия как текстового токена.
