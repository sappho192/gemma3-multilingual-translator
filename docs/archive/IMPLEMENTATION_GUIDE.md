# FFXIV 번역 모델(범용→게임 특화) 구축 가이드 (for coding agents)

> Gemma3 **Base + LoRA**, **가중 병합**(TIES/DARE) + **상용구(Autotext) 토큰 보호**를 한 번에 적용하는 실전 절차입니다.
> 아래 스텝을 **순서대로 실행**하면 됩니다. (원본 리포: `sappho192/gemma3-exercise` 기반)

---

## 0) 준비물

* Python 3.10+
* `transformers>=4.43`, `peft>=0.12`, `datasets`, `accelerate`, `sentencepiece`
* (선택) ONNX 변환용: 리포의 `uv run build_gemma.py`
* 모델: `google/gemma-3-270m` **(Base, instruct 아님)** 접근 권한

---

## 1) 리포 클론 & 기본 설치

```bash
git clone https://github.com/sappho192/gemma3-exercise.git
cd gemma3-exercise
# 리포가 안내하는 설치 절차가 있다면 우선 수행
# 예: uv / pip 등
```

---

## 2) 스페셜 토큰 “굽기”(한 번만)

> 상용구 보호용 특수 토큰 **ASCII** 버전: `"<<AT:"`, `">>"`
> 새 토큰은 **베이스**에만 추가합니다. (LoRA들은 modules_to_save 사용하지 않음)

```python
# scripts/bake_special_tokens.py (새 파일)
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_ID = "google/gemma-3-270m"  # Base
SPECIAL_TOKENS = {"additional_special_tokens": ["<<AT:", ">>"]}

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
added = tok.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForCausalLM.from_pretrained(BASE_ID)
if added > 0:
    model.resize_token_embeddings(len(tok))

tok.save_pretrained("./base_with_at_tokens")
model.save_pretrained("./base_with_at_tokens")
print("Saved to ./base_with_at_tokens")
```

실행:

```bash
python scripts/bake_special_tokens.py
```

---

## 3) 데이터 포맷 & 전처리

### 3.1 병렬 데이터 CSV

`src, tgt, src_lang, tgt_lang` 열을 권장(최소 `src,tgt` 가능).

```csv
src,tgt,src_lang,tgt_lang
<<AT:1023>>久しぶり〜,<<AT:1023>>오랜만이야~,ja,ko
おはよう！,おはよう！,ja,ja
"Let's start.",始めましょう。,en,ja
```

> **중요:** 상용구는 학습/추론 **입·출력에서 동일 문자열**로 보존합니다(모델이 “복사”를 배우게 함).

### 3.2 상용구 정규화/복원 유틸

```python
# scripts/autotext_utils.py
import re

AT_OPEN, AT_CLOSE = "<<AT:", ">>"
AT_RE = re.compile(r"<<AT:(\d+)>>")

def normalize_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("　"," ").strip())

def normalize_autotext(text: str, lang: str, to_id: dict) -> str:
    # to_id: {(lang, phrase_norm) -> id}
    def repl(m):
        phrase = normalize_phrase(m.group(1))
        id_ = to_id.get((lang, phrase))
        return f"{AT_OPEN}{id_}{AT_CLOSE}" if id_ is not None else m.group(0)
    return re.sub(r"<autotext>(.*?)</autotext>", repl, text)

def restore_autotext(text: str, ui_lang: str, from_id: dict) -> str:
    # from_id: {(ui_lang, id) -> phrase}
    def sub(m):
        i = int(m.group(1))
        return from_id.get((ui_lang, i), m.group(0))
    return AT_RE.sub(sub, text)
```

---

## 4) 학습 입력 포맷(리포 SFT 흐름에 연결)

> 리포의 이모지 예시 대신, **번역 고정 패턴**을 사용합니다.

```python
# scripts/format_dataset.py
SEPLINE = "###"

def format_translation(ex):
    prefix = f"<src:{ex.get('src_lang','ko')}><tgt:{ex.get('tgt_lang','ja')}>"
    return {"text": f"{prefix}\n{ex['src'].strip()}\n{SEPLINE}\n{ex['tgt'].strip()}"}
```

노트북/학습 스크립트에서:

* 모델/토크나이저 로드 경로를 `./base_with_at_tokens` 로 교체
* 데이터 로딩 직후 `normalize_autotext` 적용(입/출 모두)
* `dataset.map(format_translation)`, `dataset_text_field="text"`

---

## 5) LoRA 설정 & 두 단계 학습

> **modules_to_save 쓰지 않습니다.** (새 토큰은 이미 베이스에 구워져 있음)

```python
# scripts/lora_config.py
from peft import LoraConfig
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM"
)
```

### 5.1 범용 LoRA(A)

* 데이터: ko↔en, ja↔en, ko↔ja 혼합(AT 포함 예시 충분히)
* 저장 경로: `adapters/translator-general`

### 5.2 게임 특화 LoRA(B)

* 데이터: FFXIV 채팅/대사/UI + 범용 10–20% **replay 샘플**
* 저장 경로: `adapters/translator-game`

> 리포의 학습 루틴(노트북/스クリプト)을 그대로 쓰되, **모델 경로/포맷/데이터**만 위처럼 변경.

---

## 6) 가중 병합(TIES/DARE) → 단일 어댑터(선택: 완전 병합)

```python
# scripts/merge_adapters.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "./base_with_at_tokens"
A = "adapters/translator-general"
B = "adapters/translator-game"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")

m = PeftModel.from_pretrained(base, A, adapter_name="A")
m.load_adapter(B, adapter_name="B")

# 가중 병합: 게임 특화에 약간 가중
m.add_weighted_adapter(
    adapters=["A", "B"],
    weights=[1.0, 1.2],
    adapter_name="AplusB",
    combination_type="ties",  # or "dare"
    density=0.2
)
m.set_adapter("AplusB")

# (선택) 완전 병합 → 베이스 가중치에 bake-in
m.merge_and_unload()
m.save_pretrained("./merged_translator_at")
tok.save_pretrained("./merged_translator_at")

print("Saved merged model to ./merged_translator_at")
```

실행:

```bash
python scripts/merge_adapters.py
```

---

## 7) 디코딩 보호(AT 패턴 파손 방지)

```python
# scripts/decoding_guard.py
from transformers import AutoTokenizer
import torch

def build_prefix_allowed_tokens_fn(tokenizer: AutoTokenizer):
    DIGIT_TOKS = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
    OPEN_TOK = tokenizer.convert_tokens_to_ids("<<AT:")
    CLOSE_TOK = tokenizer.convert_tokens_to_ids(">>")

    def fn(batch_id, input_ids: torch.LongTensor):
        # 상태: 0=일반, 1=막 열림(숫자만), 2=숫자 진행(숫자/닫힘)
        state = 0
        for tid in reversed(input_ids.tolist()):
            if state == 0 and tid == CLOSE_TOK:
                continue
            if state == 0 and tid == OPEN_TOK:
                state = 1; break
            if state in (1,2) and tid == OPEN_TOK:
                state = 1; break
            if state in (1,2) and tid in DIGIT_TOKS:
                state = 2; break
        if state == 0:   return None
        if state == 1:   return DIGIT_TOKS
        return DIGIT_TOKS + [CLOSE_TOK]
    return fn
```

사용 예:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./merged_translator_at", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("./merged_translator_at", torch_dtype="auto", device_map="auto")

prompt = "<src:ja><tgt:ko>\n<<AT:1023>>久しぶり〜\n###\n"
guard = build_prefix_allowed_tokens_fn(tok)
out = model.generate(
    **tok(prompt, return_tensors="pt").to(model.device),
    max_new_tokens=64, do_sample=False, temperature=0.0, num_beams=4,
    prefix_allowed_tokens_fn=guard
)
print(tok.decode(out[0], skip_special_tokens=False))
```

---

## 8) ONNX 변환(리포 스크립트 재사용)

```bash
uv run build_gemma.py \
  -m ./merged_translator_at \
  -o ./merged_translator_at_onnx \
  -p fp32 fp16 q4 q4f16
```

---

## 9) 전처리↔후처리 파이프라인(런타임)

```python
# 예시: ja 원문 → ko 번역, UI 언어가 en일 때 AT치환
from scripts.autotext_utils import normalize_autotext, restore_autotext
from transformers import AutoTokenizer, AutoModelForCausalLM

to_id = {("ja", "おはようございます！"): 1023}         # 예시
from_id = {("en", 1023): "Good morning!"}            # 예시

raw = "<autotext>おはようございます！</autotext>久しぶり〜"
norm = normalize_autotext(raw, "ja", to_id)          # "<<AT:1023>>久しぶり〜"

prompt = f"<src:ja><tgt:ko>\n{norm}\n###\n"
tok = AutoTokenizer.from_pretrained("./merged_translator_at", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("./merged_translator_at", torch_dtype="auto", device_map="auto")

guard = build_prefix_allowed_tokens_fn(tok)
gen = model.generate(
    **tok(prompt, return_tensors="pt").to(model.device),
    max_new_tokens=96, do_sample=False, temperature=0.0, num_beams=4,
    prefix_allowed_tokens_fn=guard
)
txt = tok.decode(gen[0], skip_special_tokens=False)
# 출력에서 번역만 추출(세퍼레이터 '###' 뒤)
translated = txt.split("###")[-1].strip()

# UI 언어(en)로 AT 복원
final = restore_autotext(translated, "en", from_id)
print(final)
```

---

## 10) 빠른 A/B 스윕(가중 병합 가중치 탐색)

```python
weights_grid = [(1.0,1.0), (1.0,1.2), (0.8,1.2), (1.2,1.0)]
for wa, wb in weights_grid:
    # add_weighted_adapter(..., weights=[wa, wb]) 후 dev셋 측정
    # 메트릭: AT 보존율(정규식 일치), 숫자/괄호/태그 일치율, 용어집 정확도, COMET/chrF
    pass
```

---

## 11) 체크리스트(요약)

* [ ] **Base** 모델에 `"<<AT:"`, `">>"` **스페셜 토큰 추가 & 저장**
* [ ] 데이터 전처리: `<autotext>…</autotext>` → `<<AT:id>>`
* [ ] 학습 입력 포맷: `<src:..><tgt:..>\nSRC\n###\nTGT`
* [ ] **LoRA A(범용)**, **LoRA B(게임)** 학습 (modules_to_save **미사용**)
* [ ] **가중 병합**(TIES/DARE) → `merge_and_unload()`(선택)
* [ ] 디코딩 가드(`prefix_allowed_tokens_fn`) 적용
* [ ] 후처리: UI 언어별 `id→phrase` 치환

---

### 참고 운영 팁

* **형식 보존 실패 시**: 정규식으로 `<<AT:\d+>>` 패턴을 검증하고, 깨지면 **원문에서 해당 스팬 그대로 복원** 후 제출
* **데이터 증강**: `<<AT:id>>` 주변에 이모지/기호/띄어쓰기 변주 → 보존력↑
* **ko↔ja 직접 병렬**도 반드시 포함(영어 pivot에만 의존하지 않기)

