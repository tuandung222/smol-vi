---
sidebar_position: 3
sidebar_label: 'Ablation & Baseline'
---

# Mọi Mô hình Lớn Bắt đầu từ Ablation Nhỏ

Trước khi có thể bắt đầu huấn luyện một LLM (Large Language Model — Mô hình Ngôn ngữ Lớn), chúng ta cần đưa ra nhiều quyết định sẽ định hình hiệu suất và hiệu quả huấn luyện của mô hình. Kiến trúc nào sẽ phục vụ tốt nhất trường hợp sử dụng của chúng ta? Nên dùng optimizer (bộ tối ưu) và learning rate schedule (lịch trình tốc độ học) nào, và nên trộn những nguồn dữ liệu nào?

Cách các quyết định này được đưa ra là một câu hỏi thường gặp. Đôi khi người ta kỳ vọng rằng chúng đòi hỏi suy nghĩ sâu sắc. Và mặc dù tư duy chiến lược là thiết yếu, chỉ lý luận thôi là không đủ. Mọi thứ không phải lúc nào cũng trực quan với LLM, và các giả thuyết về những gì nên hoạt động đôi khi không đúng trong thực tế.

:::note 📝 Ví dụ phản trực giác
Lấy [arXiv](https://arxiv.org/) làm ví dụ — một bộ sưu tập khổng lồ kiến thức khoa học của nhân loại. Theo trực giác, huấn luyện trên dữ liệu STEM phong phú như vậy sẽ tạo ra mô hình vượt trội, đúng không? Trong thực tế, không hẳn — đặc biệt cho mô hình nhỏ, nơi nó thậm chí có thể *làm giảm* hiệu suất. Tại sao? Dù các bài báo arXiv đầy kiến thức, chúng rất chuyên biệt và được viết trong phong cách học thuật hẹp, khá khác biệt so với văn bản đa dạng, tổng quát mà mô hình học tốt nhất.
:::

Vậy làm sao biết cái gì hoạt động? **Chúng ta chạy rất nhiều thí nghiệm, như những nhà thực nghiệm giỏi!** Machine learning không phải toán thuần túy, mà thực sự là một khoa học thực nghiệm.

> Về nhiều mặt, machine learning giống như nhiệt động lực học trước khi phát hiện ra cơ học thống kê: Chúng ta có các định luật thực nghiệm và nguyên tắc thiết kế đáng tin cậy hoạt động tốt đáng kể, ngay cả khi các giải thích lý thuyết sâu hơn vẫn đang nổi lên.

Hai thuộc tính chính chúng ta muốn từ ablation:

1. **Tốc độ:** Chúng nên chạy càng nhanh càng tốt để có thể lặp thường xuyên
2. **Độ tin cậy:** Chúng nên cung cấp khả năng phân biệt mạnh — nếu các metric không thể phân biệt có ý nghĩa giữa các thiết lập khác nhau sớm, ablation có thể tiết lộ ít thông tin (và nếu chúng nhiễu, chúng ta có nguy cơ đuổi theo nhiễu!)

---

## Chọn Baseline (Điểm Xuất Phát)

Mỗi mô hình thành công đều xây dựng trên **nền tảng đã được chứng minh** và sửa đổi nó cho nhu cầu riêng:

- Khi Qwen huấn luyện gia đình mô hình đầu tiên, họ bắt đầu từ kiến trúc Llama
- Khi Meta huấn luyện Llama 3, họ bắt đầu từ Llama 2
- Kimi K2 bắt đầu từ kiến trúc MoE của DeepSeek-V3

**Tại sao?** Thiết kế kiến trúc tốt và thiết lập huấn luyện mất nhiều năm lặp đi lặp lại qua nhiều tổ chức. Transformer chuẩn và các optimizer như Adam đã được tinh chỉnh qua hàng nghìn thí nghiệm. Bắt đầu từ nền tảng đã chứng minh nghĩa là kế thừa toàn bộ kiến thức tích lũy đó. Bắt đầu mới nghĩa là tự phát hiện lại mọi vấn đề.

Để tạo điểm xuất phát tốt, một kiến trúc nên:

- ✅ **Phù hợp với ràng buộc** — tương thích với mục tiêu triển khai và trường hợp sử dụng
- ✅ **Đã chứng minh ở quy mô lớn** — đã chạy hàng nghìn tỷ token ở kích thước tương tự hoặc lớn hơn
- ✅ **Được tài liệu hóa tốt** — với siêu tham số đã chứng minh trong mô hình mở
- ✅ **Hỗ trợ framework tốt** — được hỗ trợ trong framework huấn luyện và suy luận bạn dự định dùng

### Bảng Các Lựa Chọn Baseline 2025

| Loại kiến trúc | Mô hình | Kích thước |
|----------------|---------|------------|
| **Dense** (Dày đặc) | [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) | 8B, 70B, 405B |
| | [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | 1B, 3B |
| | [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | 0.6B → 32B |
| | [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) | 1B → 27B |
| | [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) | 135M, 360M, 1.7B |
| | [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B |
| **MoE** (Hỗn hợp Chuyên gia) | [Qwen3 MoE](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | 30B-A3B |
| | [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) | 20B |
| | [Kimi Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct) | 16B-A3B |
| | [Kimi K2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | 1T-A32B |
| | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 671B |
| **Hybrid** (Lai) | [Zamba2](https://huggingface.co/Zyphra/models?search=zamba2) | 1.2B → 7B |
| | [Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | 0.5B → 34B |
| | [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 80B-A3B |
| | [MiniMax-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) | 456B |

> Tìm loại kiến trúc của bạn, chọn baseline gần với số lượng tham số bạn muốn. Đừng suy nghĩ quá nhiều — kiến trúc bạn bắt đầu không phải bất biến.

---

## Sửa Đổi Baseline: Kỷ Luật Giảm Rủi Ro (Derisking)

Bạn có một baseline hoạt động và phù hợp. Bạn *có thể* dừng ở đây, huấn luyện nó, và (giả sử hỗn hợp dữ liệu tốt) có khả năng nhận được mô hình khá tốt. Nhưng baseline không được tối ưu cho ràng buộc cụ thể của bạn. Điều này có nghĩa là có thể có các sửa đổi đáng thực hiện. Tuy nhiên, mỗi thay đổi kiến trúc đều mang rủi ro.

Kỷ luật giữ bạn đúng hướng là **derisking (giảm rủi ro):**

:::important ⚠️ Nguyên tắc Vàng
**Không bao giờ thay đổi bất cứ điều gì trừ khi bạn đã kiểm tra rằng nó có ích.**

Một thay đổi được "derisked" khi kiểm tra cho thấy nó hoặc cải thiện hiệu suất trên khả năng mục tiêu hoặc cung cấp lợi ích có ý nghĩa (nhanh hơn khi suy luận, ít bộ nhớ hơn, ổn định hơn) mà không làm giảm hiệu suất vượt quá giới hạn chấp nhận được.
:::

**Phương pháp:**
1. Bắt đầu bằng kiểm tra các thay đổi hứa hẹn so với baseline hiện tại
2. Khi có thứ hoạt động, tích hợp nó để tạo baseline mới
3. Kiểm tra thay đổi tiếp theo so với baseline mới đó

Trước khi kiểm tra bất kỳ sửa đổi nào, tự hỏi hai câu hỏi:
- *Điều này có giúp trường hợp sử dụng cụ thể của tôi không?*
- *Điều này có tối ưu hóa quá trình huấn luyện của tôi không?*

Nếu không rõ ràng, **bỏ qua nó**.

---

## Chọn Framework Huấn luyện

Quyết định đầu tiên: framework nào để huấn luyện mô hình và chạy tất cả ablation.

:::warning ⚠️ Cảnh báo
**Đừng làm anh hùng và chuyển đổi framework giữa ablation và lần chạy cuối cùng.** Đó là con đường đến đau khổ.
:::

Ba cân nhắc chính:
1. Framework phải **hỗ trợ kiến trúc mục tiêu**, hoặc cho phép mở rộng dễ dàng
2. Cần **ổn định và sẵn sàng sản xuất**, không bị hỏng bí ẩn giữa chừng
3. Nên cung cấp **throughput mạnh** để lặp nhanh

### Bảng So Sánh Framework

| Framework | Ưu điểm | Nhược điểm | Dòng mã |
|-----------|---------|------------|---------|
| **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** | Đã trải qua thử thách thực tế nhiều năm; throughput tốt; tính năng sản xuất đầy đủ | Codebase phức tạp, khó điều hướng khi mới | ~124k |
| **[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)** | Tiên phong ZeRO optimization; đã chạy BLOOM, GLM | Codebase lớn, khó khăn khi debug hoặc tùy chỉnh | ~194k |
| **[TorchTitan](https://github.com/pytorch/torchtitan)** | Nhẹ, module hóa; tuyệt vời cho thí nghiệm nhanh | Mới hơn, chưa ổn định hoàn toàn | ~14k |
| **[Nanotron](https://github.com/huggingface/nanotron)** | Linh hoạt hoàn toàn; hiểu sâu; hỗ trợ đầy đủ tính năng sản xuất | Cần đầu tư xây dựng; MoE đang phát triển | — |

> **Mẹo:** Nếu nhiều framework hỗ trợ nhu cầu của bạn, so sánh throughput trên phần cứng cụ thể. Cho thí nghiệm nhanh, codebase đơn giản hơn thường thắng.

---

## Thiết Lập Ablation

### Mục tiêu

Chạy thí nghiệm ở **quy mô nhỏ** và nhận kết quả có thể **ngoại suy với độ tin cậy** sang lần chạy sản xuất cuối cùng.

Hai cách tiếp cận chính:
1. **Mô hình kích thước đầy đủ, ít token hơn** — VD: SmolLM3 3B trên 100B token thay vì 11T
2. **Mô hình proxy nhỏ hơn** — VD: Kimi K2 dùng mô hình MoE 3B với 0.5B tham số hoạt động cho ablation

:::tip 💡 Quy tắc chuyển đổi
Nếu thứ gì đó *làm giảm hiệu suất* ở quy mô nhỏ, bạn có thể tự tin loại bỏ nó ở quy mô lớn. Nếu thứ gì đó *hoạt động* ở quy mô nhỏ, hãy đảm bảo bạn đã huấn luyện trên đủ token để kết luận với xác suất cao rằng phát hiện sẽ ngoại suy.
:::

### Thiết lập của SmolLM3

Chúng tôi dùng **mô hình 1B transformer** theo kiến trúc [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) được huấn luyện trên **45B token**. Điều này mất khoảng **1.5 ngày** để huấn luyện trên một node 8 GPU H100 (42k token/giây/GPU).

> Chúng tôi huấn luyện 45B token để đảm bảo tín hiệu ổn định, mặc dù ~35B là [Chinchilla-optimal](https://arxiv.org/abs/2203.15556) cho kích thước mô hình này.

### Cấu hình Baseline YAML

```yaml
## Datasets and mixing weights
data_stages:
  - data:
      dataset:
        dataset_folder:
          - fineweb-edu
          - stack-edu-python
          - finemath-3plus
        dataset_weights:
          - 0.7
          - 0.2
          - 0.1

## Model architecture, Llama 3.2 1B configuration
model:
  model_config:
    hidden_size: 2048
    num_hidden_layers: 16
    num_attention_heads: 32
    num_key_value_heads: 8
    intermediate_size: 8192
    max_position_embeddings: 4096
    rope_theta: 50000.0
    tie_word_embeddings: true

## Training hyperparameters, AdamW with cosine schedule
optimizer:
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0005
    lr_decay_starting_step: 2000
    lr_decay_steps: 18000
    lr_decay_style: cosine
    lr_warmup_steps: 2000
    lr_warmup_style: linear
    min_decay_lr: 5.0e-05
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW

## Parallelism, 1 node
parallelism:
  dp: 8    # Data parallel trên 8 GPU
  tp: 1    # Không cần tensor hoặc pipeline parallelism ở quy mô 1B
  pp: 1

## Tokenizer
tokenizer:
  tokenizer_max_length: 4096
  tokenizer_name_or_path: HuggingFaceTB/SmolLM3-3B

## Batch size, sequence length và tổng huấn luyện
tokens:
  batch_accumulation_per_replica: 16
  micro_batch_size: 3
  # GBS (global batch size) = dp * batch_acc * MBS * sequence = 1.5M tokens
  sequence_length: 4096
  train_steps: 20000  # GBS * 20000 = 30B
```

Cho các ablation, chúng tôi sẽ sửa đổi các phần khác nhau tùy theo những gì đang kiểm tra:
- Phần `model` cho [lựa chọn kiến trúc](./thiet_ke_kien_truc.md)
- Phần `optimizer` cho siêu tham số optimizer và huấn luyện
- Phần `data_stages` cho tuyển chọn dữ liệu

:::important ⚠️ Nguyên tắc Ablation
**Chỉ thay đổi một biến mỗi ablation**, giữ mọi thứ khác không đổi. Nếu bạn thay đổi nhiều thứ và hiệu suất cải thiện, bạn sẽ không biết nguyên nhân. Kiểm tra các sửa đổi riêng lẻ, sau đó kết hợp các sửa đổi thành công và đánh giá lại.
:::

### Đếm Tham Số

Khi chạy ablation, một số thay đổi kiến trúc có thể thay đổi đáng kể số lượng tham số. Để đảm bảo so sánh công bằng, cần theo dõi số lượng tham số. Đây là hàm đơn giản:

```python
from transformers import LlamaConfig, LlamaForCausalLM

def count_parameters(
    tie_embeddings=True,
    num_key_value_heads=4,
    num_attention_heads=32,
    hidden_size=2048,
    num_hidden_layers=16,
    intermediate_size=8192,
    vocab_size=128256,
    sequence_length=4096,
):
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=sequence_length,
        tie_word_embeddings=tie_embeddings,
    )
    model = LlamaForCausalLM(config)
    return f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B"
```

---

## Hiểu Cái Gì Hoạt Động: Đánh Giá (Evaluation)

Khi đã chạy ablation, làm sao biết cái gì hoạt động?

### Loss — Cần Thiết Nhưng Chưa Đủ

Bản năng đầu tiên là nhìn vào loss, và đúng vậy, điều đó quan trọng. Nhưng chỉ nhìn vào loss **không phải lúc nào cũng đáng tin cậy**:

- Huấn luyện trên Wikipedia cho loss thấp hơn web, nhưng điều đó không có nghĩa mô hình tốt hơn
- Nếu thay đổi tokenizer giữa các lần chạy, loss không thể so sánh trực tiếp
- Một số thay đổi ảnh hưởng cụ thể đến khả năng nhất định (toán, suy luận) nhưng bị pha loãng trong loss trung bình

### Ba Dạng Đánh Giá

| Dạng | Viết tắt | Mô tả | Khi nào dùng |
|------|----------|-------|--------------|
| **Multiple Choice Format** | MCF | Mô hình chọn đáp án A/B/C/D | Mô hình lớn, sau nhiều token |
| **Cloze Formulation** | CF | So sánh likelihood các lựa chọn | Ablation nhỏ, tín hiệu sớm |
| **Free-form Generation** | FG | Kiểm tra accuracy của sinh tự do | Post-training, mô hình đã tinh chỉnh |

### Bộ Đánh Giá Ablation

Chúng tôi sử dụng các benchmark từ ablation [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), thêm benchmark toán và code như **GSM8K** và **HumanEval**, và benchmark context dài **RULER**. Đánh giá chạy bằng [LightEval](https://github.com/huggingface/lighteval).

Bốn nguyên tắc đánh giá tốt (từ [FineTasks](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks)):
1. **Tính đơn điệu (Monotonicity):** Điểm nên cải thiện khi huấn luyện lâu hơn
2. **Ít nhiễu (Low noise):** Điểm không dao động nhiều giữa các seed khác nhau
3. **Trên mức ngẫu nhiên (Above-random):** Benchmark nên cho hiệu suất trên ngẫu nhiên sớm
4. **Nhất quán xếp hạng (Ranking consistency):** Thứ tự so sánh nên ổn định qua quá trình huấn luyện

### Hỗn Hợp Dữ Liệu Cho Ablation

- **Ablation kiến trúc:** Huấn luyện trên hỗn hợp cố định — tiếng Anh ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), toán ([FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)), code ([Stack-Edu-Python](https://huggingface.co/datasets/HuggingFaceTB/stack-edu))
- **Ablation dữ liệu:** Cố định kiến trúc và thay đổi hỗn hợp dữ liệu một cách hệ thống

---

## Ước Tính Chi Phí Ablation

Ablation tuyệt vời, nhưng chúng **cần thời gian GPU**. Bảng sau cho thấy phân bổ compute hoàn chỉnh cho pretraining SmolLM3:

| Hạng mục | Giờ GPU | Ghi chú |
|----------|---------|---------|
| **Lần chạy chính** | 276,480 | Bao gồm downtime không thường xuyên |
| **Ablation + Debug** | 161,280 | >100 ablation qua toàn bộ quá trình phát triển |
| **Đánh giá** | ~10,000 | Bộ đánh giá đầy đủ mỗi 10B token qua 11T token |

> **Sự thật quan trọng:** Ablation và debug tiêu tốn tổng cộng **161,280 giờ GPU** — hơn một nửa chi phí lần chạy chính!

:::warning ⚠️ Bài học từ DeepSeek-V3
Khi [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) ra mắt, thế giới tập trung vào chi phí huấn luyện $5.6M được báo cáo. Nhiều người hiểu đó là chi phí R&D đầy đủ. Thực tế, nó chỉ phản ánh lần chạy huấn luyện cuối cùng. Chi phí lớn hơn nhiều — và thường vô hình — là trong nghiên cứu: ablation, lần chạy thất bại, và debug dẫn đến công thức cuối cùng.
:::

**Quy tắc lập ngân sách:** Lên kế hoạch cho **chi phí huấn luyện + ablation + buffer cho bất ngờ**.

---

## Quy Tắc Tham Gia (Rules of Engagement)

**TL;DR: Hãy hoang tưởng.**

1. **Xác nhận bộ đánh giá.** Trước khi huấn luyện bất kỳ mô hình nào, đảm bảo bộ đánh giá có thể tái tạo kết quả đã công bố. Nếu benchmark nào có tính chất sinh (ví dụ GSM8K), hãy kiểm tra thủ công vài mẫu để đảm bảo prompt được format đúng.

2. **Kiểm tra mọi thay đổi, dù nhỏ.** Đừng đánh giá thấp tác động của việc nâng cấp thư viện có vẻ vô hại hay commit "chỉ thay đổi hai dòng." Những thay đổi nhỏ có thể đưa vào bug tinh vi hoặc thay đổi hiệu suất làm nhiễm kết quả.

3. **Thay đổi một thứ một lần.** Giữ mọi thứ khác giống hệt giữa các thí nghiệm. Đánh giá đóng góp riêng lẻ trước, sau đó thử kết hợp.

4. **Huấn luyện trên đủ token và dùng đủ đánh giá.** Cắt góc sẽ dẫn đến kết quả nhiễu và quyết định tồi.

> **Nguyên tắc vàng:** Một khi bạn có thiết lập tốt, **không thay đổi nào nên không được kiểm tra!**
