---
sidebar_position: 15
sidebar_label: 'Kết luận'
---

# Kết luận & Tài liệu Tham khảo

## Kết luận

Chúng ta bắt đầu hành trình này với một câu hỏi đơn giản: **Thực sự cần gì để huấn luyện một LLM hiệu suất cao vào năm 2026?** Để trả lời câu hỏi đó, chúng ta đã đi qua toàn bộ pipeline — từ pretraining đến post-training — cho thấy không chỉ các kỹ thuật được sử dụng mà còn cả **phương pháp luận** làm cho chúng hoạt động.

### Những gì đã được trình bày

Chúng ta bắt đầu bằng cách trình bày **Training Compass** (La bàn huấn luyện) — một framework để quyết định liệu có nên huấn luyện hay không — sau đó cho thấy cách biến mục tiêu thành các quyết định kiến trúc cụ thể. Bạn đã thấy cách:

- **Thiết lập ablation pipeline** đáng tin cậy, test thay đổi từng cái một, và scale từ thí nghiệm vài tỷ token lên chạy multi-trillion token
- **Ghi nhận các thách thức hạ tầng** xuất hiện ở quy mô lớn (sụp throughput, bottleneck dataloader, bug tinh vi) và cách monitoring + derisking có hệ thống giúp bắt chúng sớm
- **Xây dựng post-training pipeline** có hệ thống: thiết lập evals trước khi train bất cứ thứ gì, lặp lại trên SFT data mixtures, áp dụng preference optimization, và tùy chọn đẩy xa hơn với RL

### Những bài học xuyên suốt

Xuyên suốt cả hai giai đoạn — pretraining và post-training — chúng ta liên tục quay lại cùng những **insight cốt lõi**:

> [!IMPORTANT]
> **Validate mọi thứ qua thí nghiệm**, thay đổi từng thứ một, kỳ vọng scale sẽ phá vỡ mọi thứ theo cách mới, và để **use case dẫn dắt** quyết định thay vì chạy theo mọi paper mới.

Tuân theo quy trình này là cách SmolLM3 được huấn luyện — một multilingual reasoner 3B tham số cạnh tranh với long context. Trong suốt quá trình, đội ngũ đã học được rất nhiều về **cái gì hiệu quả, cái gì hỏng, và cách debug khi mọi thứ đi sai**. Tất cả đã được ghi nhận — cả thành công lẫn thất bại.

### Bước tiếp theo

Hướng dẫn này bao gồm những nền tảng của LLM training hiện đại, nhưng lĩnh vực đang phát triển nhanh chóng:

- **Chạy thí nghiệm của riêng bạn**: Đọc về ablation hữu ích; chạy thí nghiệm của riêng mình mới dạy bạn cái gì thực sự quan trọng. Chọn model nhỏ, setup evals, và bắt đầu thử nghiệm.
- **Đọc source code**: Training framework như Nanotron, TRL và các tool khác đều open source. Hiểu implementation của chúng tiết lộ chi tiết mà paper bỏ qua.
- **Theo dõi nghiên cứu mới**: Paper về các SOTA model gần đây cho thấy hướng đi của lĩnh vực.

> *Bây giờ hãy đi train thứ gì đó. Và khi loss spike bí ẩn lúc 2 giờ sáng, hãy nhớ: Mọi model vĩ đại đều có những câu chuyện debugging đằng sau. Chúc lực lượng open source và open science luôn ở bên bạn!*

## Lời cảm ơn (Acknowledgments)

Cảm ơn [Guilherme](https://huggingface.co/guipenedo), [Hugo](https://huggingface.co/hlarcher), và [Mario](https://huggingface.co/mariolr) vì những phản hồi quý giá, và [Abubakar](https://huggingface.co/abidlabs) vì sự giúp đỡ với các tính năng Trackio.

---

## Tài liệu Tham khảo (References)

Dưới đây là danh sách curated các paper, sách và blog post đã truyền cảm hứng và thông tin cho hành trình huấn luyện LLM.

### Kiến trúc LLM (LLM Architecture)

- **Dense models:**
  - [Llama 3](https://huggingface.co/papers/2407.21783) — Tech report chi tiết từ Meta về họ mô hình Llama
  - [OLMo 2](https://huggingface.co/papers/2501.00656) — Open language model từ AI2
  - [MobileLLM](https://huggingface.co/papers/2402.14905) — Kiến trúc tối ưu cho mô hình nhỏ
- **MoEs (Mixture of Experts):**
  - [DeepSeek-V2](https://huggingface.co/papers/2405.04434) — Kiến trúc MLA + MoE hiệu quả
  - [DeepSeek-V3](https://huggingface.co/papers/2412.19437) — Mở rộng quy mô với MoE
  - [Scaling Laws of Efficient MoEs](https://huggingface.co/papers/2507.17702) — Scaling laws cho MoE
- **Hybrid models:**
  - [MiniMax-01](https://huggingface.co/papers/2501.08313) — Mô hình hybrid attention
  - [Mamba2](https://huggingface.co/papers/2405.21060) — State space model thế hệ mới

### Optimizer & Tham số Huấn luyện

- [Muon is Scalable for LLM Training](https://huggingface.co/papers/2502.16982) — Optimizer Muon mở rộng cho LLM
- [Fantastic Pretraining Optimizers](https://huggingface.co/papers/2509.02046) — So sánh toàn diện các optimizer
- [Large Batch Training](https://arxiv.org/abs/1812.06162) — Kỹ thuật training batch lớn
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954) — Scaling laws và insights từ DeepSeek

### Curation Dữ liệu (Data Curation)

- **Web:**
  - [FineWeb & FineWeb-Edu](https://huggingface.co/papers/2406.17557) — Dataset web chất lượng cao
  - [FineWeb2](https://huggingface.co/papers/2506.20920) — Phiên bản nâng cấp đa ngôn ngữ
  - [DCLM](https://huggingface.co/papers/2406.11794) — DataComp for Language Models
- **Code:**
  - [The Stack v2](https://huggingface.co/papers/2402.19173) — Dataset code quy mô lớn
  - [To Code or Not to Code](https://huggingface.co/papers/2408.10914) — Tác động của dữ liệu code
- **Math:**
  - [DeepSeekMath](https://huggingface.co/papers/2402.03300) — Dataset và mô hình toán học
  - [FineMath](https://huggingface.co/papers/2502.02737) — Dữ liệu toán chất lượng cao
  - [MegaMath](https://huggingface.co/papers/2504.02807) — Dataset toán quy mô lớn
- **Data mixtures:**
  - [SmolLM2](https://huggingface.co/papers/2502.02737) — Chiến lược pha trộn dữ liệu
  - [Does Your Data Spark Joy?](https://huggingface.co/papers/2406.03476) — Phương pháp chọn lọc dữ liệu

### Scaling Laws

- [Kaplan et al.](https://huggingface.co/papers/2001.08361) — Neural scaling laws ban đầu từ OpenAI
- [Chinchilla](https://huggingface.co/papers/2203.15556) — Compute-optimal training (Hoffmann et al.)
- [Scaling Data-Constrained Language Models](https://huggingface.co/papers/2305.16264) — Scaling laws khi dữ liệu bị giới hạn

### Post-Training

- [InstructGPT](https://huggingface.co/papers/2203.02155) — Paper nền tảng của OpenAI biến base model thành assistant hữu ích. Tiền thân của ChatGPT và một bước quan trọng trên con đường nhân loại tiến lên thang Kardashev.
- [Llama 2](https://huggingface.co/papers/2307.09288) & [Llama 3](https://huggingface.co/papers/2407.21783) — Tech report cực kỳ chi tiết từ Meta về training Llama models. Chứa nhiều insight về human data collection, cho cả human preferences và model evaluation.
- Secrets of RLHF in LLMs, [Part I](https://huggingface.co/papers/2307.04964) & [Part II](https://huggingface.co/papers/2401.06080) — Nhiều chi tiết về RLHF, đặc biệt cách train reward model mạnh.
- [Direct Preference Optimization (DPO)](https://huggingface.co/papers/2305.18290) — Paper đột phá 2023 thuyết phục mọi người ngừng làm RL với LLMs.
- [DeepSeek-R1](https://huggingface.co/papers/2501.12948) — Paper đột phá 2025 thuyết phục mọi người bắt đầu làm RL với LLMs.
- [Understanding R1-Zero-Like Training (Dr. GRPO)](https://huggingface.co/papers/2503.20783) — Một trong những paper quan trọng nhất về hiểu bias bẩm sinh trong GRPO và cách sửa.
- [DAPO](https://huggingface.co/papers/2503.14476) — Bytedance chia sẻ nhiều chi tiết implementation để mở khóa training R1-Zero-like ổn định cho cộng đồng.
- [ScaleRL](https://huggingface.co/papers/2510.13786) — Meta flex lớn để derive scaling laws cho RL. Đốt hơn 400k GPU hours để thiết lập training recipe scale đáng tin cậy.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — Blog post tuyệt vời phát hiện RL với low-rank LoRA có thể match full fine-tuning (kết quả đáng ngạc nhiên).
- [Command A](https://huggingface.co/papers/2504.00698) — Tech report chi tiết đáng chú ý từ Cohere về các chiến lược post-train LLMs hiệu quả.

### Hạ tầng (Infrastructure)

- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — Hướng dẫn toàn diện về distributed training quy mô lớn
- [Jax Scaling Book](https://jax-ml.github.io/scaling-book/) — Scaling training với JAX
- [Modal GPU Glossary](https://modal.com/gpu-glossary/readme) — Từ điển GPU từ Modal

### Training Frameworks

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — Framework training LLM của NVIDIA
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) — Thư viện tối ưu deep learning của Microsoft
- [TorchTitan](https://github.com/pytorch/torchtitan) — Framework training PyTorch native
- [Nanotron](https://github.com/huggingface/nanotron/) — Framework pretraining của Hugging Face
- [NanoChat](https://github.com/karpathy/nanochat) — Minimal chat training từ Andrej Karpathy
- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning library

### Đánh giá (Evaluation)

- [LLM Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) — Hướng dẫn đánh giá LLM toàn diện
- [OLMES](https://huggingface.co/papers/2406.08446) — Open Language Model Evaluation Standard
- [FineTasks](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks) — Fine-grained evaluation tasks
- [Lessons from the Trenches](https://huggingface.co/papers/2405.14782) — Bài học thực chiến về evaluation

---

## Trích dẫn (Citation)

Để trích dẫn trong ngữ cảnh học thuật, vui lòng cite công trình này như sau:

> Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching, Carlos Miguel Patiño, Clémentine Fourrier, Thibaud Frere, Anton Lozhkov, Colin Raffel, Leandro von Werra, Thomas Wolf (2025). "The Smol Training Playbook: The Secrets to Building World-Class LLMs".

### BibTeX

```bibtex
@misc{allal2025_the_smol_training_playbook_the_secrets_to_building_world_class_llms,
  title={The Smol Training Playbook: The Secrets to Building World-Class LLMs},
  author={Loubna Ben Allal and Lewis Tunstall and Nouamane Tazi and Elie Bakouch
          and Ed Beeching and Carlos Miguel Patiño and Clémentine Fourrier
          and Thibaud Frere and Anton Lozhkov and Colin Raffel
          and Leandro von Werra and Thomas Wolf},
  year={2025},
}
```

---

## Về bản dịch Tiếng Việt

Bản dịch tiếng Việt này được thực hiện bởi **[tuandung222](https://github.com/tuandung222)** với mục tiêu mang kiến thức về huấn luyện LLM đến gần hơn với cộng đồng AI Việt Nam.

### Nguyên tắc dịch

- **Giữ nguyên thuật ngữ kỹ thuật quan trọng** bằng tiếng Anh (LLM, pretraining, fine-tuning, ablation, tokenizer, optimizer, v.v.) kèm giải thích tiếng Việt khi lần đầu giới thiệu
- **Bảo toàn toàn bộ** công thức LaTeX, code blocks, và citations
- **Dịch tự nhiên, mạch lạc** — không dịch từng từ máy móc mà chuyển tải ý nghĩa và tinh thần của bản gốc
- **Bổ sung diagram Mermaid** để trực quan hóa các quy trình phức tạp

### Nguồn gốc

Bản gốc tiếng Anh: [The Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/) — được phát triển bởi đội ngũ Hugging Face.

### Đóng góp

Nếu bạn phát hiện lỗi dịch hoặc muốn cải thiện bản dịch, vui lòng tạo issue hoặc pull request trên repository.

---

*Made with ❤️ bởi cộng đồng AI Việt Nam*

*Bản gốc được tạo với [research article template](https://huggingface.co/spaces/tfrere/research-article-template)*
