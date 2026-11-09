# MOSS-SoundEffect v2.0 — описание репозитория

Описание подготовлено по карточке модели и данным Hugging Face API
(на 9 ноября 2026 года).

- Страница модели: [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
- Проект с кодом: [OpenMOSS/MOSS-TTS — каталог moss_soundeffect_v2](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2)

## Что это такое

**MOSS-SoundEffect v2.0** — открытая модель генерации звука по текстовому описанию
(text-to-audio / sound effects generation) от команды OpenMOSS. По промпту на
естественном языке она создаёт неречевые звуки: природное и городское окружение,
животных и существ, звуки действий человека, а также короткие музыкальные и
перкуссионные фрагменты.

В семействе MOSS-TTS это специализированная модель «текст → звук», которая
отвечает именно за неречевое аудио. Версия 2.0 заменила дискретно-токенную
авторегрессионную основу v1 (`MossTTSDelay`) на непрерывную схему
**Diffusion Transformer + Flow Matching**.

Репозиторий на Hugging Face — это прежде всего хранилище весов и конфигурации
модели в формате библиотеки `diffusers`, а не самостоятельный проект с кодом.
Код пайплайна инференса, демо на Gradio и рецепты дообучения находятся в
GitHub-репозитории MOSS-TTS (каталог `moss_soundeffect_v2`).

## Ключевые особенности

- **Архитектура:** Diffusion Transformer (DiT) с обучением по схеме Flow
  Matching, декодер DAC VAE и текстовый энкодер Qwen3.
- **Объём модели:** вариант DiT `1.3B` — 1,3 млрд параметров.
- **Длительность:** стабильная генерация до **30 секунд** за один вызов; на
  этапе обучения к промпту добавляется тег длительности.
- **Частота дискретизации:** 48 кГц.
- **Промпты:** модель обучена на английских и китайских описаниях.
- **Охват звуковых эффектов:** природа, город, животные и существа, действия
  человека, короткие музыкальные/перкуссионные клипы.
- **Формат:** пайплайн `MossSoundEffectPipeline` для `diffusers`, веса в
  `safetensors`.

## Состав репозитория

| Путь | Назначение |
|---|---|
| `README.md` | Карточка модели с описанием и примером использования. |
| `model_index.json` | Описание пайплайна `diffusers` (`MossSoundEffectPipeline`). |
| `scheduler/scheduler_config.json` | Конфигурация планировщика Flow Matching. |
| `text_encoder/` | Текстовый энкодер Qwen3-1.7B: два шарда весов `safetensors`, индекс, конфиги. |
| `tokenizer/` | Токенизатор Qwen3 (`tokenizer.json`, `vocab.json`, `merges.txt`). |
| `transformer/diffusion_pytorch_model.safetensors` | Веса DiT-трансформера. |
| `vae/vae_128d_48k.pth` | VAE-декодер DAC: 128-мерный латент, 48 кГц. |

## Рекомендуемые параметры инференса

| Параметр | Значение по умолчанию | Описание |
|---|---:|---|
| `num_inference_steps` | 100 | Число шагов решателя Flow Matching. |
| `cfg_scale` | 4.0 | Вес classifier-free guidance. |
| `sigma_shift` | 5.0 | Сдвиг планировщика, применяемый на каждом вызове. |
| `seconds` | 10.0 | Длительность аудио; максимум — 30 секунд. |

## Быстрый старт

Рекомендуется отдельное окружение Python 3.12, чтобы избежать конфликтов
зависимостей с основным окружением MOSS-TTS. В примере используется сборка
PyTorch для CUDA 12.8.

```bash
conda create -n moss-soundeffect-v2 python=3.12 -y
conda activate moss-soundeffect-v2

git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS/moss_soundeffect_v2
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[torch-cu128]"
```

Для установки только для инференса (без дополнений для дообучения) используется
та же команда без extra `finetune`.

```python
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

pipe = MossSoundEffectPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-SoundEffect-v2.0",   # этот репозиторий или локальный каталог
    torch_dtype=torch.bfloat16,
    device="cuda",
)

audio = pipe(
    prompt="A dog barking loudly in a park.",
    seconds=10,
    num_inference_steps=100,
    cfg_scale=4.0,
)                                            # тензор波形 (B, C, T)
pipe.save_audio(audio, "out.wav")
```

## Примечания

- DiT обёрнут в `torch.compile` + Triton CUDA Graph, поэтому первый вызов может
  занять несколько минут на компиляцию.
- При ошибках TorchDynamo/Triton перед запуском Python можно установить
  `TORCHDYNAMO_DISABLE=1`.
- Для дообучения используется extra `finetune`; демо на Gradio и рецепты
  дообучения описаны в GitHub-репозитории проекта.

## Метаданные и лицензия

- **Автор/организация:** OpenMOSS-Team
- **Лицензия:** Apache 2.0
- **Базовая модель:** `Qwen/Qwen3-1.7B`
- **Библиотека:** `diffusers`
- **Тип задачи (`pipeline_tag`):** `text-to-audio`
- **Языки промптов:** английский (`en`), китайский (`zh`)
- **Теги:** `diffusion`, `flow-matching`, `sound-effects`, `audio-generation`
- **Популярность на момент описания:** около 5,9 тыс. загрузок и 158 отметок
  «нравится» (данные Hugging Face API).

## Ссылки

- Модель на Hugging Face: <https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0>
- Исходный код и документация: <https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2>
