Domain LLM 개발
## 방법 / 도구
* Unsolth: Fine-tuning을 위한 도구
* Ollama, Webui: 모델 실행을 위한 도구
* ChatML: Huggingface 표준 데이터세트 도구
## Dataset 제작
* 교수님 원본 데이터를 기반으로 작업 + ChatGPT 활용
```text
실험: AEM
유효 면적: 25 cm²
분리막: sustainion 
Cathode 촉매: Ni–Mo 합금 (20%), Ni foam-Ni (20%), ZZ (60%)
Anode 촉매: Ni–Fe-Ox (60%), NiCoOx, RuO₂ (20%), ZZ (20%)
전극 지지체:	 Ni foam (50%), carbon paper (50%) 
바이폴라 플레이트:  스테인리스 (70%), Ti 코팅 SS (30%)
씰링: 실리콘 (100%)
엔드플레이트: 기계적 압착

전원 범위: 0–3 V / 0–50 A DC 전원 
급수: DI water
온도조절장비: 히터
사용 장비: 가스–액체 분리기, MFC, 
관측 장비: 압력계, GC, 
안전: 배압 밸브, 수소 센서, N2 purge
기타 재료: O-링·개스킷 예비품, 토크 렌치, 질소


(운전 조건)
전해액: KOH 0.1 - 2M (범위에서 랜덤 추출)  
펌프: 유량 5–15 mL min⁻¹ (범위에서 랜덤 추출)
온도: 25–80 ℃ (범위에서 랜덤 추출)
압력: 1 - 3bar (범위에서 랜덤 추출)

(성능)
H2 생산량: 0.3 - 1.2 g/h
O2 생산량: 2 - 10 g/h
O2 순도: 99 - 99.9% 
Faradaic 효율: 97 - 99%
H2 순도: 99.9%

**
(1) 온도가 높으면, 생산량이 높도록 
(2) ZZ 촉매의 경우 생산량이 상위 10%
```

* 100개를 목표 => 퀄리티 중요
	* https://docs.unsloth.ai/basics/datasets-guide
	* https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me
* https://www.youtube.com/watch?v=YZW3pkIR-YE
* https://databoom.tistory.com/entry/fine-tunning-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A4%80%EB%B9%84%ED%95%98%EA%B8%B0
* https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama
	* https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb#scrollTo=HvOPfPnet76H
* https://huggingface.co/datasets/vicgalle/alpaca-gpt4/viewer?views%5B%5D=train
## 코드
```bash
# 환경 세팅
conda create -n llm python=3.11
conda activate llm

pip install datasets, unsloth, hf_xet

# 선택적
pip install label-studio
label-studio run

# 빌드 오류 해결 https://github.com/unslothai/unsloth/issues/748
apt-get install cmake curl
apt-get install libcurl4-openssl-dev libxtst-dev libssl-dev
cmake -B build
cmake --build build --config Release
cd /workspace/LLM/llama.cpp/build/bin
cp llama-quantize /workspace/LLM/llama.cpp

# 이후 modelfile ollama에 넣어서 사용
python trainer.py
python ollama_runner.py
```

```python title:LLM_trainer
from unsloth import FastLanguageModel, is_bfloat16_supported, standardize_sharegpt, apply_chat_template, to_sharegpt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

dataset = load_dataset('json', data_files='questions.json', split="train")

print(dataset.column_names)

dataset = to_sharegpt(
    dataset,
    merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
    output_column_name = "output",
    conversation_extension = 3, # Select more to handle longer conversations
)

dataset = standardize_sharegpt(dataset)

chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    # eval_dataset = dataset_eval,
    # compute_metrics=compute_metrics
    args = TrainingArguments(
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 100,
        num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none" # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

# Inferencing
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
messages = [                    # Change below!
    {"role": "user", "content": "Describe about AEM experiment as far as you know"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)

# Saving
model.save_pretrained_gguf("model_test", tokenizer, quantization_method = "q4_k_m")
```

```json title:Questionaire
[
    {
        "instruction":"what is your domain knowlege?",
        "output":"my domain knowledge is anion exchange membrane (AEM).",
        "input": "",
        "text":""
    },
    {
        "instruction":"what is AEM experiment?",
        "output":"Anion Exchange Membrane (AEM) is a polymer separator that conducts hydroxide ions (OH⁻) while blocking gases in alkaline water electrolyzers. It lets OH⁻ migrate from cathode to anode, enabling H₂ generation at the cathode and O₂ at the anode without gas mixing. Operating in alkaline media means you can use inexpensive Ni- or Fe-based catalysts, making AEM electrolyzers a cheaper alternative to acidic PEM systems.",
        "input": "",
        "text":""
    },
    {
        "instruction":"How large the reactor in AEM experiment?",
        "output":"Active area volume is 25cm^2.",
        "input": "",
        "text":""
    },
    {
        "instruction":"What kind of membrane used in AEM experiment?",
        "output":"The setup uses a Sustainion anion-exchange membrane (AEM) with a 25 cm² active area. ",
        "input": "",
        "text":""
    },
    {
        "instruction":"What kind of cathode catalyst used in aem experiment?",
        "output":"The cathode uses a three-component catalyst mix 1) Ni–Mo alloy, 2) Nickel on Ni foam, 3) Proprietary “ZZ” catalyst, which consistently delivers top-tier hydrogen output. ​",
        "input": "",
        "text":""
    },
    {
        "instruction":"What kind of anode catalyst used in aem experiment?",
        "output":"The cathode uses a three-component catalyst mix 1) Ni–Fe-Ox, 2) NiCoOx + RuO₂, 3) Proprietary “ZZ” catalyst, which consistently delivers top-tier hydrogen output. ​",
        "input": "",
        "text":""
    },
    {
        "instruction":"What is best catalyst in aem experiment?",
        "output":"Based on the performance note in the experiment log, the proprietary “ZZ” catalyst is the top performer—it consistently pushes the cell’s hydrogen (and oxygen) production into the upper 10 % of all runs, outperforming the other Ni-based oxide mixes on both electrodes. ​",
        "input": "",
        "text":""
    },
    {
        "instruction":"What kind of electrode support used in aem experiment?",
        "output":"The electrodes are made from nickel foam or carbon paper. ​",
        "input": "",
        "text":""
    },
    {
        "instruction":"What kind of bioplar plate used in aem experiment?",
        "output":"The bipolar plates are metallic: plain stainless steel or titanium-coated stainless steel.​",
        "input": "",
        "text":""
    },
    {
        "instruction":"How end plate assembled in aem experiment?",
        "output":"The cell’s end plates are mechanically compressed —i.e., clamped/bolted together to apply uniform pressure across the stack and keep the membrane-electrode assembly sealed during operation. ​",
        "input": "",
        "text":""
    },
    {
        "instruction":"How much power applied in aem experiment?",
        "output":"A programmable DC source delivers 0 – 3 V and up to 50 A, giving a maximum electrical input of roughly 150 W (3 V × 50 A) for the AEM cell.​",
        "input": "",
        "text":""
    },
    {
        "instruction":"How can i can control temperature in aem experiment?",
        "output":"Use a simple closed-loop heating setup.​",
        "input": "",
        "text":""
    },
    {
        "instruction":"What equipments used in aem experiment?",
        "output":"For fluid handling, Gas–liquid separator and Mass-flow controller (MFC) are used. For Instrumentation, Pressure gauge and Gas chromatograph (GC) are used.​ Finally for safety, Back-pressure valve, Hydrogen sensor and N₂ purge line are used",
        "input": "",
        "text":""
    },
    {
        "instruction":"Was MFC used in aem experiment?",
        "output":"Yes — a mass-flow controller (MFC) is explicitly listed among the devices used in the AEM setup. ",
        "input": "",
        "text":""
    },
    {
        "instruction":"What is the molar concentration of the electrolyte in aem experiment?",
        "output":"molar conentration in aem experiment is between 0.1 and 2. ",
        "input": "pick number randomly between 0.1 ~ 2",
        "text":""
    },
    {
        "instruction":"What is the flow rate in operating condition? in aem experiment?",
        "output":"The electrolyte flow rate under operating (“drive”) conditions is 5 ~ 15 mL·min⁻¹.",
        "input": "pick number randomly between 5 ~ 15",
        "text":""
    },
    {
        "instruction":"What is the temperature in operating condition? in aem experiment?",
        "output":"The cell is operated at a temperature within 25 ~ 80 °C. ",
        "input": "pick number randomly between 25 ~ 80",
        "text":""
    },
    {
        "instruction":"What is the pressure in operating condition? in aem experiment?",
        "output":"The cell is run at a pressure between 1 and 3 bar. ​ ",
        "input": "pick number randomly between 1 ~ 3",
        "text":""
    },
    {
        "instruction":"How many hydrogen produced in aem experiment?",
        "output":"Hydrogen production rates in the AEM experiment ranged from 0.3 g h⁻¹ to 1.2 g h⁻¹.​ ",
        "input": "",
        "text":""
    },
    {
        "instruction":"How many o2 produced in aem experiment?",
        "output":"Oxygen is produced at a rate of 2 to 10 g h⁻¹ in the AEM experiment.​ ",
        "input": "",
        "text":""
    },
    {
        "instruction":"What is the purity of the oxygen?",
        "output":"The oxygen purity in the AEM experiment is between 99 % and 99.9 %.​ ",
        "input": " ",
        "text":""
    },
    {
        "instruction":"What is the purity of the h2 in aem experiment?",
        "output":"The hydrogen purity in the AEM experiment is 99.9 %. ",
        "input": " ",
        "text":""
    },
    {
        "instruction":"How much the effieciecy of faradaic in aem experiment?",
        "output":"The Faradaic efficiency in the AEM experiment ranges from 97 % to 99 %.",
        "input": " ",
        "text":""
    }
]
```

```python title:Ollama_importer
import subprocess
import os
import time

# Wait for a few seconds for Ollama to load!
subprocess.Popen(["ollama", "serve"])
time.sleep(4)
print("ollama running")

os.system("ollama create unsloth_model -f ./model_test/Modelfile")
time.sleep(4)
print("model_loaded")

os.system("conda init")
os.system("conda activate ollama")
os.system("open-webui serve --port 6864")
print("GUI_loaded")
```

## 결과
*  정상작동 확인
  ![[Pasted image 20250430082749.png]](https://github.com/coport-uni/unsolth_llm_peal/blob/main/images/Pasted%20image%2020250430082749.png)
* 일반 llama3.1과 비교
  ![[Pasted image 20250430093439.png]](https://github.com/coport-uni/unsolth_llm_peal/blob/main/images/Pasted%20image%2020250430093439.png)
* Rank 및 batch 개선후
  ![[Pasted image 20250430114717.png]](https://github.com/coport-uni/unsolth_llm_peal/blob/main/images/Pasted%20image%2020250430114717.png)
