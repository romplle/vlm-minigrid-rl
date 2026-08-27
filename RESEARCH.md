# Research state — MiniGrid Navigation NanoVLM

Публичный канон экспериментов для GitHub и внешних рецензентов. Дата среза: **2026-08-27**.

`README.md` — человеческий отчёт линейки A (короткий SFT, 3 эпохи) и команды запуска. Этот файл — научный каталог: что измерено, что отвергнуто, какие цифры с какими **нельзя** складывать.

Локальные черновики главы, сырые JSON и INDEX живут только на машине автора (`docs/experiments/`, `docs/thesis/`, `runs/`) и **не** являются частью git. Цифр, которых нет здесь или в README, выдумывать нельзя.

---

## 1. Постановка

**Стенд.** Hugging Face [NanoVLM v0.1](https://github.com/huggingface/nanoVLM/releases/tag/v0.1) (~222M) + LoRA. Среда: MiniGrid Empty, частичный RGB ego-view 7×7. Действия: `left` / `right` / `forward`. В тексте ответа токены **с ведущим пробелом**: ` left`, ` right`, ` forward` — это другие tokenizer id, чем `left` без пробела.

**Пайплайн.** BFS-эксперт (выбор left-first vs right-first shortest path, чтобы не копить L/R-дисбаланс) → SFT LoRA на одном action-токене → опционально GRPO LoRA в среде → env eval.

**Метрика.** Success rate в среде (обычно 250 эпизодов). Не validation action accuracy: там доминирует `forward`. Сопутствующие: `timeouts`, `invalid_action_episodes`, `avg_steps` на победе. `avg_reward` MiniGrid зависит от `env.max_steps` — не сравнивать reward между разными горизонтами.

**Научный вопрос (после того как pipeline заработал).** Когда пост-SFT GRPO поднимает in-domain success, а когда он затачивается на shortcut (цвет цели, короткий горизонт) и вредит OOD? Чинится ли OOD-провал test-time compute без новых весов?

**Связанная диагностическая ось.** Протокольный декодер `generate` (полный словарь) и 3-way softmax по трём action-токенам — **не одно и то же**. GRPO scoring / PPO используют 3-way; rollout и eval используют `generate`.

---

## 2. Протокол (инварианты)

- **Train ≡ eval = `generate`:** `model.generate(..., max_new_tokens=1)`. Невалидный токен сразу заканчивает эпизод (нет fallback на `forward`); счётчик `invalid_action_episodes`.
- **Scoring / PPO / KL:** 3-way softmax по ` left` / ` right` / ` forward` (`score_action_logits`). Не full-vocab log-prob. KL: DeepSeekMath / Schulman k2 **только по rollout-действию**.
- **SFT-формат:** префикс кончается на `Assistant:` без хвостового пробела; цели — ` left` / ` right` / ` forward`.
- **Оптимизатор:** `torch.optim.AdamW`. SFT default batch 32. GRPO update батчится (`--update-batch-size`, default 32); rollout последовательный.
- **Headline GRPO HP** (если не оговорено иное): `lr=5e-6`, `epsilon=0.1`, `beta=0.1`, `lora-dropout=0.0`. argparse-дефолты `scripts/grpo.py` другие — для воспроизведения headline их надо передавать явно.
- **Primary GRPO checkpoint в главе:** заранее объявленный `episode-100`, без argmax по промежуточным сохранениям. README 8×8 **97.73%** — это `episode-75` (исторический pick).

Нет флагов `--rollout-mode` / `--action-mode` на поддерживаемом pipeline. Новые train-раны — всегда новый `--output-dir`; существующие checkpoint-директории не перезаписывать.

---

## 3. Две линейки и два горизонта — не смешивать

### 3.1 Линейки весов

| Линейка | Корни | Что нести | Чего не нести |
|---------|--------|-----------|----------------|
| **A — README headline, 3 эпохи SFT** | `sft_adapter_*_bs32`, `grpo_adapter_*_from_bs32_sft3` | 8×8 SFT e3 **93.47%**; глава канонит GRPO **ep100 97.20%**. 16×16 SFT e3 **78.13%** → GRPO ep100 **83.33%** (один GRPO train seed) | README 8×8 **97.73% @episode-75** как «канон главы» |
| **B — длинный SFT e10 / handoff** | `sft_adapter_*_bs32_e10`, `grpo_adapter_*_handoff_from_e10_*` | 8×8 лучший SFT e9; GRPO **98.27%**, Δ **+0.40±0.23**. 16×16 лучший SFT e8 **82.80%**; F1 GRPO mean на e3 **82.13%**, на e8 **82.00%** | Одноseed A1 как реплика; 83.33% линейки A рядом с 82.80% линейки B как «один эксперимент» |

Отдельный **старый стек** (SFT batch 6 / AdamW8bit): `2026-07-sft-8x8-e10` (лучший SFT e6 mean **97.73%**) и `2026-07-grpo-8x8-from-sft-e6` (GRPO ep75 mean **98.13%**). Это **не** bs32. Проценты не класть в одни таблицы с A/B.

DoorKey — третья семья сред и 5 действий. Проценты **не** мешать с Empty A/B.

### 3.2 Горизонт среды

Почти все цифры ниже (README, линейки A/B, C1, TTAug D) сняты на **старом unified horizon**:

| Среда | Старый eval / типичный train | Примечание |
|-------|------------------------------|------------|
| Empty 8×8 | **12** | C1 goal-red и in-domain TTAug — тоже 12 |
| Empty 16×16 | **40** eval (GRPO train часто 35) | C1 transfer и TTAug transfer — 40 |
| DoorKey 6×6 | **30** | не Empty |
| DoorKey 8×8 (superseded) | **42** train=eval | не сравнивать с lmax |

В коде (локальный `main`, ещё не обязательно на GitHub) горизонт **разведён**: `train_max_steps = L_max`, `eval_max_steps = L_max + max(4, p95 steps-to-see)`; GRPO пинит `env.unwrapped.max_steps` на **train**, чтобы MiniGrid reward `1 − 0.9·t/max_steps` различал короткие победы.

| Env | L_max | train | eval |
|-----|------:|------:|-----:|
| Empty 8×8 | 12 | 12 | 16 |
| Empty 16×16 | 28 | 28 | 38 |
| DoorKey 6×6 | 19 | 19 | 29 |
| DoorKey 8×8 | 26 | 26 | 41 |

**Не обновлять README-таблицы** под новый горизонт, пока не будет нового multi-seed eval. Не склеивать % @12 с % @16 и @40 с @38. C1/D нарочно считались на старом горизонте, чтобы быть сравнимыми с собой.

---

## 4. Линейка A — короткий SFT (то, что в README)

Стек: bs32, AdamW, generate, 250×3 eval seeds (42 / 123 / 456), **старый горизонт**.

| Среда | SFT epoch-3 mean | GRPO | Δ | Комментарий |
|-------|-----------------:|------|--:|-------------|
| Empty 8×8 | **93.47%** | README **97.73% @ep75**; глава **97.20% @ep100** | +4.26 / +3.73 | Лучший промежуточный pick ≠ primary главы |
| Empty 16×16 | **78.13%** | **83.33% @ep100** | +5.20 | **Один** GRPO train seed (42). F1 на том же бюджете e3: mean GRPO **82.13%**, Δ **+4.00** |

База NanoVLM без SFT: ~0.8% / 0.4%, почти все эпизоды — invalid token. Majority-forward: ~5% / ~3%. Expert BFS: 100%.

**Transfer / goal-color в README — seed 42, линейка A (e3 / ep75 или ep100), не C1.**

| Train → test | SFT | GRPO | Вывод |
|--------------|----:|-----:|--------|
| 8×8 → 8×8 | 93.47% | 97.73% @ep75 | in-domain, GRPO помогает |
| 8×8 → 16×16 | 56.4% | 44.8% | GRPO **хуже** SFT |
| 16×16 → 16×16 | 78.13% | 83.33% | in-domain |
| 16×16 → 8×8 | 94.0% | 96.0% | перенос сильный, GRPO помогает |

Goal-color 8×8 (зелёные веса, красная цель; seed 42, линейка A):

| Visual / prompt | SFT | GRPO |
|-----------------|----:|-----:|
| green / green | 91.6% | 98.0% |
| red / red | 40.4% | 8.0% |
| red / green | 44.8% | 9.6% |

SFT частично держится; GRPO резко падает и крутит повороты. Это **одноseed** иллюстрация того же знака, что потом закрыл C1 на линейке B.

Эксперимент: `2026-07-bs32-sft-grpo`. Rerun transfer/goal-color/16×16 skim: `2026-07-readme-headline-rerun`.

---

## 5. Линейка B — длинный SFT и handoff

### 5.1 Остаток GRPO после e10 (`2026-07-sft-e10-bs32`)

Тот же стек bs32, SFT до 10 эпох. Seed-42 residual мал: 8×8 97.87% → 98.53% (+0.66); 16×16 82.80% → 82.93% (+0.13). **Канон главы — 3 train seeds** (§5 ниже): 8×8 GRPO **98.27%** (+0.40); 16×16 GRPO **82.00%** (−0.80).

GRPO наиболее полезен, когда SFT **недообучен** (3 эпохи), не после длинного sweep.

### 5.2 A1 Empty ladder (`2026-07-handoff-empty`)

8×8: сырой Δ +14.1 pp на SFT-e1 → +0.7 pp на e9 (seed 42, pick эпизода в таблицах A1 — архив, не канон). OOD на seed 42: GRPO вредит. 16×16 A1 — один train seed 42, ~+0.9 pp «горячее» тройки F1; для 16×16 тезисные цифры — F1, не A1.

### 5.3 F1 training-seed 16×16 (`2026-07-grpo-trainseed-16x16`) — **PASS**

Бюджеты e1 / e3 / e5 / e8; GRPO seeds 42 / 123 / 456; ep100; generate.

| SFT epoch | SFT % | mean Δ (3 train seeds) | sd | ER |
|----------:|------:|-----------------------:|---:|---:|
| 1 | 63.33 | **+4.18** | 2.02 | 11.4% |
| 3 | 78.13 | **+4.00** | 1.87 | 18.3% |
| 5 | 79.60 | +1.33 | 1.96 | 6.5% |
| 8 | **82.80** | **−0.80** | 0.96 | −4.7% |

Paired early (e1,e3) vs late (e5,e8): mean contrast **+3.82** pp, t(2)=**7.27**, знак 3/3. Ни одно среднее GRPO-arm не бьёт plain SFT e8 (82.80%) под `generate`.

### 5.4 8×8 e9 equivalence (`2026-08-grpo-trainseed-8x8`) — **PASS**

SFT e9 **97.87%**; GRPO ep100, seeds 42/123/456.

| | GRPO % | Δ | ER |
|--|-------:|--:|---:|
| mean ± sd | **98.27** | **+0.40 ± 0.23** | **18.7%** |

Знак 3/3 плюс. Equivalence в полосе ±1 pp **и** 18.7% residual error reduction. Абсолютный чемпион под `generate` — **SFT+GRPO**, не plain SFT. Вывод «длинный SFT побеждает, RL не нужен» — **16×16-специфичен**.

### 5.5 Phase 0 (`2026-07-phase0-reanalysis`)

Primary checkpoint = ep100 (argmax-selection отставлен). G0 **INCONCLUSIVE**, затем **retired**. 16×16 Δ в одноseed A1 ~1 pp оптимистична.

---

## 6. Декодер: `generate` vs greedy 3-way (`argmax3`)

Диагностика, **не** смена протокола. Headline и F1 остаются на `generate`. `argmax3` = argmax по трём action-логитам; invalid token физически невозможен.

### 6.1 Empty-16×16 (`2026-08-greedy-multiseed-16x16`)

6 бюджетов (e1/3/5/7/8/9) × 3 GRPO train seeds. Одноseed-предшественник `2026-08-greedy-sampled-16x16` (вердикт `policy_improvement`) **superseded**.

| Факт | Число |
|------|--------|
| Лучший SFT `generate` | e8 **82.80%** |
| Лучший SFT greedy | e5 **92.40%** |
| Лучший GRPO greedy (mean) | e5 **92.00%** (Δ −0.40 vs greedy SFT e5) |
| Плато ~83% = потолок среды | **Отозвано.** Среда допускает ≥92% memoryless |
| F3 на полной сетке | **`mixed`** |
| Greedy SFT e7 | **65.87%** (коллапс, таймауты 34.1%; sampled 78.93%) |

Под **каждым** декодером на 16×16 лучшая модель — SFT-чекпойнт, не SFT+GRPO. Декодер меняет, *какая* эпоха лучшая. Greedy хрупок (циклы). Δ_greedy на e7 не интерпретируема (сломан baseline). Гипотеза «вся кривая Δ — заострение при семплировании» **отвергнута** (Δ_greedy e1 = +10.31 pp). Положительная «Δ_greedy повторяет Δ_sampled» **не** поддержана.

Устойчивый эффект GRPO на 16×16 под greedy: таймауты SFT 7.6–34.1%, GRPO **8.0–12.1%** на всех шести бюджетах (стабилизация против зацикливания, не обязательно рост среднего success).

### 6.2 Empty-8×8 C2 (`2026-08-greedy-8x8`)

A1-эпохи; GRPO train seed **42** (не тройка). Коллапса e7 **нет** (greedy SFT e7 = 96.00%). Лучший greedy SFT = e9 **98.27%**; там GRPO **вредит** (Δ_greedy **−2.00**). Абсолютный чемпион greedy — GRPO-from-e2 **99.07%**. «Под каждым декодером чемпион = SFT» остаётся 16×16-специфичным. F3 на этой лестнице: `policy_improvement` (один train seed).

---

## 7. Headroom vs entropy vs rejuvenation

### 7.1 Δ ~ headroom (`2026-08-matched-headroom-16x16` + пул F1)

Шесть 16×16 бюджетов × 3 train seeds; entropy из A2.

| ep | entropy | headroom | mean Δ |
|---:|--------:|---------:|-------:|
| 1 | 0.422 | 36.67 | +4.18 |
| 3 | 0.180 | 21.87 | +4.00 |
| 5 | 0.310 | 20.40 | +1.33 |
| 7 | 0.206 | 21.07 | +2.89 |
| 8 | 0.256 | 17.20 | −0.80 |
| 9 | 0.259 | 20.27 | +2.80 |

Spearman(headroom, Δ) = **+0.943**; Spearman(entropy, Δ) = **+0.029**; Spearman(epoch, Δ) = −0.714. Нет «двух режимов» и нет монотонного спада Δ по номеру эпохи (e7/e9 снова ~+2.8 при headroom ~20).

Matched-headroom (e5/e7/e9, headroom ~20.3–21.1): порядок по энтропии **не детектируется** (крупнейший контраст t=0.92). Ярлык оркестратора `plasticity_rejected` **не** читать как powered rejection.

### 7.2 A2 (`2026-07-plasticity-rejuvenation`)

С ростом SFT на 8×8 entropy↓ / max_prob↑; на 16×16 немонотонно. `adv_near_zero_frac` вакуумна. Trajectory fusion `θ ← (1−α)·θ_e9 + α·θ_e1`: лучший α=0; α=1 ≈ −9.6 pp (8×8) / −14.5 pp (16×16) vs e9. Это **не** оператор rejuvenation из literature anchor.

### 7.3 F4 base-anchored LoRA

Прокси: `W' = α · W_LoRA(e8)` (base + α·ΔLoRA), затем GRPO seed 42, ep100. Порог PASS: +2 pp над SFT e8 (84.80%). Neuron reset статьи **не** реализовывался. 3-way entropy 0.18–0.42 несоизмерима с OverSFT LLM (entropy ~0.02).

Экран α≤0.85 (`2026-08-base-anchored-16x16`): вердикт **`NOT_TESTED_INFORMATIVE_ALPHA`**. α=0.5 уничтожает политику до GRPO (scaled SFT 2%). α=0.85: GRPO 78.53% (−4.27 vs e8).

Informative α∈{0.90, 0.95} (`2026-08-base-anchored-informative-16x16`): **`NEGATIVE`**. Лучший α=0.95: GRPO **83.20%** (**+0.40** pp < +2). GRPO частично отыгрывает сжатие LoRA, но не проходит порог.

---

## 8. OOD — C1 (`2026-08-ood-multiseed-8x8`) — **`HURTS_REPLICATED`**

Замороженные адаптеры линейки B: `sft_adapter_8x8_bs32_e10/epoch-{1,5,9}` + `grpo_adapter_8x8_handoff_from_e10_e{1,5,9}/episode-100`. Eval seeds 42/123/456, 250 ep. Горизонт **старый**: transfer 40, goal-red 12. Seed 42 совпадает с A1.

**Transfer 8→16** (mean ± sample sd):

| SFT epoch | SFT | GRPO | Δ pp | 3/3 хуже |
|----------:|----:|-----:|-----:|:--------:|
| 1 | 49.73% | 41.47% | −8.27 ± 1.22 | да |
| 5 | 59.87% | 45.33% | −14.53 ± 2.20 | да |
| **9** | **57.47%** | **45.07%** | **−12.40 ± 4.80** | да |

**Goal-red 8×8:**

| SFT epoch | SFT | GRPO | Δ pp | 3/3 хуже |
|----------:|----:|-----:|-----:|:--------:|
| 1 | 30.93% | 9.60% | −21.33 ± 1.15 | да |
| 5 | 40.27% | 4.13% | −36.13 ± 3.03 | да |
| **9** | **17.87%** | **2.40%** | **−15.47 ± 1.97** | да |

Все шесть клеток: GRPO хуже на всех трёх eval seeds. На том же e9 **in-domain** Δ = **+0.40 ± 0.23** pp. Знаки противоположны.

Не измерено в C1: greedy OOD, native 16×16 OOD, transfer 16→8, mismatch prompt/goal, OOD по GRPO **train** seed.

---

## 9. Тема D — photometric TTAug (`2026-08-ttaug-ood-8x8`) — **`HURTS_MORE`**

**Вопрос.** Чинит ли test-time compute (Kaya et al., arXiv:2510.03574: среднее next-token распределений по K видамм) C1 OOD **без новых весов**?

**Дизайн.** Те же e9 C1 адаптеры, read-only. Seed **42**, 250 ep. Горизонты как C1 (12 / 12 / 40), не новые 16/38. Ауги: brightness / contrast / Gaussian noise / box blur; view 0 = identity. **Нет** flip/rotate (ломают left/right) и **нет** hue (confound goal-color).

| Метод | Декодер |
|-------|---------|
| **G1** | `generate`, K=1, без аугментаций (реcompute, не копипаст C1 JSON) |
| **S1** | argmax среднего 3-way softmax, K=1 (identity) |
| **T8** | mean 3-way softmax по K=8 видамм |

**In-domain 8×8 (max_steps=12):**

| Метод | SFT | GRPO | action mass GRPO |
|-------|----:|-----:|------------------|
| G1 | 98.0% | **99.6%** | ~15/16/69 L/R/F |
| S1 | 4.4% | 4.4% | **~94% `right`** |
| T8 | 0.0% | 0.0% | **~97% `right`** |

**Goal-red:** G1 SFT 14.8% / GRPO **1.6%** (Δ **−13.2** pp). S1/T8 = 0. S1-GRPO: **100% `right`**.

**Transfer 8→16:** G1 SFT 57.6% / GRPO **45.2%** (Δ **−12.4** pp). S1 12.0/12.0. T8 10.0/11.6. S1/T8 ~95–97% `right`.

G1 воспроизводит знак C1. T8 хуже G1 и in-domain, и на OOD. Seeds 123/456 **не** гонялись: seed 42 достаточно, чтобы отвергнуть «TTC чинит C1».

**Интерпретация.** Отвергнуто не «любой test-time compute», а фотометрическое усреднение **3-way** распределений. S1 уже на in-domain падает с ~99% до 4%: узкий 3-way scoring, которым считается GRPO, **не ведёт** в env так, как `generate`. T8 усредняет сломанную голову. Не закрыто: majority по **generate** (M8), hue, TTAdapt, K-sweep, эпохи 1/5.

Код D — отдельная ветка (`ttaug-ood`); на момент среза мог быть ещё не в `main`. Результат от этого не меняется.

---

## 10. DoorKey (боковая ветка, не глава Empty)

**6×6** (`2026-08-handoff-doorkey-6x6`), eval=30, 5-way `left/right/forward/pickup/toggle`. SFT e1 **92.13%** → e8 **99.47%** (250×3). Все эпохи >90% → 9 GRPO **не стартовали** (`all_above_90` / `ladder_aborted`). Стенд живой (expert 100%, pickup/toggle используются); задача слишком лёгкая для формы Δ(SFT) как на Empty. Shape vs Empty **не идентифицирован**. Это saturation imitation, не null GRPO.

**8×8 superseded** (`2026-08-handoff-doorkey-8x8`): unified `max_steps=42`. SFT e1 84.53% → e8 98.40%. Не сравнивать с lmax.

**8×8 lmax** (`2026-08-handoff-doorkey-8x8-lmax`): train 26 / eval 41, те же SFT веса. SFT e1 **82.93%** → e8 **98.93%**; abort=None. 9 GRPO в новых директориях `*_lmax_*` на срезе были **in progress**. DoorKey **не** сюжетная линия главы Empty; проценты не мешать с A/B. Продолжение 9 GRPO — отдельное решение, не «добить ради Empty».

---

## 11. Прочие concluded / historical

| id | Роль | Результат |
|----|------|-----------|
| `2026-07-grpo-protocol-compare` | legacy `restricted_sample` vs generate | 8×8 SFT 94.8 / legacy 97.47 / generate 97.07; 16×16 80.13 / 82.27 / 82.67. Generate ≈ legacy на mean; дальше только generate |
| `2026-07-action-distribution` | L/R skew | Политики skew относительно эксперта; success ≠ L1-to-expert. Не заменяет env-success |
| `2026-07-full-pipeline-eval` | `run_tests.py --pipeline all` | Снимок агрегата, не канон главы |
| `2026-07-sft-8x8-e10` | старый стек batch 6 | лучший SFT e6 mean **97.73%** |
| `2026-07-grpo-8x8-from-sft-e6` | GRPO с того SFT | ep75 mean **98.13%** (+0.4 vs strong SFT) |
| `2026-07-sft-16x16-sweep` | superseded | заменён `2026-07-bs32-sft-grpo` |
| `2026-06-grpo-hp-validation` | superseded | A/B/C HP → conservative README HP |
| `2026-06-grpo-investigation` | superseded | probes leading-space / scoring |
| `2026-06-grpo-16x16-validation` | superseded | seed-42 skim |
| `2026-06-grpo-ppo-ratio-fix` | historical | лог фикса PPO-ratio |
| `2026-06-sft-16x16-3epoch` | historical | ранние 3-epoch 16×16 notes |

Наблюдаемость 16×16 (Phase 0 audit, не отдельный id в таблице выше): ~24.6% стартов с пустым ego-view; цель видна ~16.8% стартов (на 8×8 пустых стартов 0%). Объясняет таймауты и почему нет 100%, **не** объясняет разрыв 83→92 (это декодер).

---

## 12. Сводка утверждений

| Утверждение | Статус |
|-------------|--------|
| При слабом SFT GRPO даёт большой in-domain выигрыш | Да (A; F1 e1/e3) |
| Early vs late контраст на 16×16 | Да (F1 t=7.27) |
| На лучшем SFT GRPO ≈ 0 в сырых pp на обеих сетках | Equivalence ±1 pp; на 8×8 ER 18.7% и чемпион = SFT+GRPO |
| Ни один средний GRPO не бьёт лучший SFT **своего** декодера | Да на **16×16** оба декодера; на 8×8 generate-чемпион = SFT+GRPO, greedy-чемпион = GRPO e2 99.07% |
| Плато ~83% = потолок наблюдаемости | **Нет (отозвано)** |
| Δ — артефакт семплирования | **Нет** |
| Δ_greedy повторяет Δ_sampled | Не подтверждено на 16×16 (`mixed`) |
| Δ на 16×16 следует за headroom, не за энтропией | Да (ρ +0.943 / +0.029) |
| Entropy-пластичность при fixed headroom | Не детектируется |
| Trajectory- / base-anchored fusion помогает late SFT | **Нет** (A2 null; F4 informative **NEGATIVE**) |
| «Два режима» / Δ монотонно падает с эпохой | Нет |
| GRPO улучшает OOD (transfer 8→16, goal-red) | **Нет** (`HURTS_REPLICATED`) |
| Photometric TTAug (mean 3-way, K=8) чинит C1 | **Нет** (`HURTS_MORE`); 3-way голова схлопывается в `right` |

---

## 13. Карта checkpoint’ов (читать, не перезаписывать)

| Роль | Путь |
|------|------|
| Headline SFT 8×8 / 16×16 | `checkpoints/sft_adapter_{8,16}x{8,16}_bs32/epoch-3` |
| Headline GRPO (глава) | `..._from_bs32_sft3/episode-100` |
| Headline GRPO 8×8 (README pick) | `..._from_bs32_sft3/episode-75` |
| Лучший SFT линейки B | `sft_adapter_8x8_bs32_e10/epoch-9`, `sft_adapter_16x16_bs32_e10/epoch-8` |
| C1 / D e9 | `sft_adapter_8x8_bs32_e10/epoch-9` + `grpo_adapter_8x8_handoff_from_e10_e9/episode-100` |
| F1 GRPO 16×16 | `grpo_adapter_16x16_handoff_from_e10_e{1,3,5,8}/` (+ `_seed{123,456}`) |
| 8×8 e9 GRPO seeds | `grpo_adapter_8x8_handoff_from_e10_e9/` (+ `_seed{123,456}`) |
| F4 scaled | `sft_adapter_16x16_base_scaled_e8_a{05,07,085,09,095}/`, `grpo_adapter_16x16_base_anchored_e8_a{…}/` |

SFT adapter лежит только в `epoch-*`, не в родителе папки. Если флаги путей опущены, скрипты резолвят **legacy** `sft_adapter_{8,16}x{8,16}`, не `*_bs32`. Для текущей работы пути передавать явно.

---

## 14. Код (поддерживаемый pipeline)

Entrypoints: `scripts/dataset_generation.py`, `sft.py`, `grpo.py`, `test_models.py`, `run_tests.py` (+ `_bootstrap.py`). Общее: `src/vlm_minigrid_rl/`. Ablation-оркестраторы (handoff, OOD, TTAug, DoorKey) живут на исследовательских ветках, не обязаны быть в `main`.

Eval: multi-seed native suites; GRPO грузить **вместе** с родительским SFT (`load_vlm_model_with_adapters([sft, grpo])`).

---

## 15. Что заморожено и что ещё открыто

**Заморожено как глава Empty in-domain.** Не нужна ещё одна SFT-лестница / HP-sweep ради +0.X pp in-domain. Старые веса читать можно; headline-таблицы README не переписывать до нового multi-seed на новом горизонте.

**DoorKey** не продолжает сюжет Empty: 6×6 слишком лёгок для формы handoff; 8×8 — отдельная ветка.

**Открыто (не догма, а дыры в claim):**

1. **Механизм C1.** Когда GRPO учит навигацию, а когда shortcut (цвет / геометрия / горизонт). Factorized interventions на замороженных C1 весах — следующий содержательный вопрос после провала D.
2. **Декодер vs scoring.** G1 живой, S1 мёртвый: 3-way голова, которой считается GRPO, не совпадает с `generate`. Это constrains любые «починки», которые сидят на 3-way softmax (включая TTAug-Kaya).
3. **Leftover D.** Majority по `generate` на фотометрических видах (M8) на тех же C1 весах — единственный дешёвый TTC, который сидел бы на **рабочем** декодере. Не прогнан.
4. Не закрыто из ограничений главы: greedy OOD, native 16×16 OOD, OOD по GRPO train seed, причина циклов greedy e7 на 16×16.

**Не делать** без новой фальсифицируемой гипотезы, которая меняет строку из §12: mixed-size curriculum, prompt/CoT/VLA «на всякий случай», смена протокола на argmax3 как новый headline, полный categorical KL, новый RL-фреймворк, flip/rotate/hue как primary, перезапись checkpoint dirs, обновление README % до multi-seed на новом горизонте.

---

## 16. Как цитировать этот файл

Для внешнего разбора: читать **этот файл целиком**, затем таблицы README как линейку A. Не восстанавливать локальный INDEX. Если конфликт README (ep75, future-work 2026-07) vs этот файл — канон здесь, кроме команд запуска и картинок pipeline.
