# AGENT.md — Tiêu chuẩn Dịch thuật Kỹ thuật AI/ML sang Tiếng Việt

## Mục đích

File này là **hướng dẫn tiêu chuẩn** cho tất cả sub-agents tham gia dịch tài liệu kỹ thuật AI/ML sang tiếng Việt. Mọi agent **PHẢI** đọc file này trước khi bắt đầu dịch.

---

## 1. Nguyên tắc Tổng quát

### Giọng văn (Tone)
- **Chuyên nghiệp nhưng gần gũi** — như một senior engineer giải thích cho đồng nghiệp
- Tránh giọng sách giáo khoa khô khan
- Giữ humor/wit của bài gốc (ví dụ: "dark arts" → "nghệ thuật đen tối")
- Dùng "chúng tôi" khi bài gốc dùng "we" (giữ nguyên perspective tác giả)

### Độ trung thành
- Dịch **ý**, không dịch **từng từ**
- Giữ nguyên cấu trúc đoạn văn và heading hierarchy
- KHÔNG thêm thông tin mà bài gốc không có
- KHÔNG bỏ sót nội dung

---

## 2. Quy tắc Thuật ngữ Kỹ thuật

### 2.1. LUÔN giữ tiếng Anh (không dịch)

Các thuật ngữ đã trở thành chuẩn quốc tế, dịch sang tiếng Việt gây khó hiểu:

```
LLM, transformer, attention, token, embedding, encoder, decoder,
GPU, CPU, TPU, VRAM, HBM, NVLink, PCIe,
batch size, learning rate, epoch, loss, gradient,
checkpoint, optimizer, scheduler,
SFT, DPO, GRPO, PPO, RLHF, RLVR, KTO, ORPO, APO,
MHA, MQA, GQA, MLA, RoPE, NoPE,
MoE, dense, sparse,
fine-tuning, pretraining, post-training, mid-training,
inference, deployment, serving,
benchmark, eval, metric,
LoRA, QLoRA, FSDP, ZeRO,
Flash Attention, KV cache,
pipeline, framework, kernel,
on-policy, off-policy, reward, policy,
ablation, baseline, SOTA,
FLOPs, throughput, latency, bandwidth,
dataset, dataloader, tokenizer,
code, script, config
```

### 2.2. Dịch + giữ gốc trong ngoặc (lần đầu xuất hiện)

```
pretraining → pretraining (huấn luyện trước)
post-training → post-training (huấn luyện giai đoạn sau)
fine-tuning → fine-tuning (tinh chỉnh)
scaling laws → scaling laws (quy luật mở rộng)
overfitting → overfitting (quá khớp)
underfitting → underfitting (chưa khớp)
catastrophic forgetting → catastrophic forgetting (quên thảm khốc)
reward hacking → reward hacking (khai thác reward)
data curation → data curation (xử lý/quản lý dữ liệu)
knowledge distillation → knowledge distillation (chưng cất kiến thức)
weight decay → weight decay (suy giảm trọng số)
dropout → dropout (loại bỏ ngẫu nhiên)
warmup → warmup (khởi động)
cooldown → cooldown (hạ nhiệt)
annealing → annealing (ủ nhiệt)
```

### 2.3. Dịch hoàn toàn sang tiếng Việt

```
model → mô hình
parameter → tham số
hyperparameter → siêu tham số
architecture → kiến trúc
layer → lớp
training → huấn luyện
evaluation → đánh giá
performance → hiệu suất
accuracy → độ chính xác
experiment → thí nghiệm
result → kết quả
comparison → so sánh
improvement → cải thiện
implementation → triển khai/cài đặt
convergence → hội tụ
divergence → phân kỳ
stability → ổn định
efficiency → hiệu quả
```

### 2.4. Quy tắc viết tắt

- **Lần đầu:** viết đầy đủ tiếng Anh + viết tắt + giải thích Việt
  - Ví dụ: `Supervised Fine-Tuning (SFT — Tinh chỉnh có giám sát)`
- **Các lần sau:** chỉ dùng viết tắt: `SFT`
- **KHÔNG** dịch viết tắt: ~~TCGS~~ → dùng `SFT`

---

## 3. Quy tắc Format

### 3.1. Frontmatter YAML (bắt buộc)

```yaml
---
sidebar_position: <số thứ tự>
sidebar_label: '<tên ngắn tiếng Việt>'
---
```

### 3.2. Heading

- `#` — Tiêu đề chương (1 per file)
- `##` — Mục lớn
- `###` — Mục con
- `####` — Mục chi tiết
- Heading dịch sang tiếng Việt, giữ thuật ngữ kỹ thuật

### 3.3. Code blocks

- **KHÔNG dịch** code, comments trong code, hoặc output
- Giữ nguyên language tag: ` ```python `, ` ```bash `, etc.
- Code YAML/JSON config: giữ nguyên

### 3.4. LaTeX / Math

- **KHÔNG sửa** bất kỳ công thức nào
- Giữ nguyên `$inline$` và `$$block$$`
- Kiểm tra escape characters: `\_`, `\{`, `\}`

### 3.5. Mermaid diagrams

- Label trong diagram: dịch sang tiếng Việt khi có thể
- Node ID: giữ tiếng Anh
- **QUAN TRỌNG:** Dùng `<br/>` cho xuống dòng trong label
- Tất cả Mermaid PHẢI nằm trong ` ```mermaid ` code block

### 3.6. Admonitions (Docusaurus)

Sử dụng cú pháp Docusaurus:
```
> [!NOTE]
> Nội dung ghi chú

> [!TIP]
> Mẹo hữu ích

> [!IMPORTANT]
> Thông tin quan trọng

> [!WARNING]
> Cảnh báo

> [!CAUTION]
> Cảnh báo nghiêm trọng
```

### 3.7. MDX Safety — Ký tự đặc biệt

**CRITICAL:** Docusaurus dùng MDX parser, nên các ký tự `<` và `>` ngoài code block sẽ bị hiểu nhầm thành JSX tags.

| Ký tự | Trong code block | Ngoài code block |
|-------|-----------------|-----------------|
| `<` | ✅ Giữ nguyên | ❌ Dùng `&lt;` |
| `>` | ✅ Giữ nguyên | ⚠️ Dùng `&gt;` (trừ blockquote `>`) |
| `{` | ✅ Giữ nguyên | ⚠️ Cẩn thận, có thể bị parse |

Ví dụ:
- ❌ `Mô hình nhỏ (<32B)` → ✅ `Mô hình nhỏ (&lt;32B)`
- ❌ `β < 0.1` → ✅ `β &lt; 0.1`
- ✅ Trong backticks OK: `` `<think>` ``

### 3.8. Links

- Link nội bộ: `[tên](./ten_file.md)` — dùng relative path
- Link ngoại: giữ nguyên URL gốc
- **Kiểm tra** tất cả link nội bộ trỏ đến file thực sự tồn tại

---

## 4. Cấu trúc File

### Tên file
- Tiếng Việt không dấu, snake_case
- Ví dụ: `thiet_ke_kien_truc.md`, `optimizer_sieu_tham_so.md`

### Thứ tự nội dung trong mỗi file
1. YAML frontmatter
2. Tiêu đề `#`
3. Đoạn giới thiệu ngắn
4. Nội dung chính với heading hierarchy
5. Quy tắc thực hành / Rules of Engagement (nếu có)

---

## 5. Bảng Thuật ngữ Thống nhất (Glossary)

| English | Tiếng Việt | Ghi chú |
|---------|-----------|---------|
| Training Compass | La bàn Huấn luyện | |
| Ablation | Ablation | Không dịch |
| Baseline | Baseline | Không dịch |
| Data Curation | Nghệ thuật Xử lý Dữ liệu | |
| Training Marathon | Cuộc Marathon Huấn luyện | |
| Scaling Laws | Scaling Laws | Không dịch |
| Rules of Engagement | Quy tắc Thực hành | |
| Vibe testing | Vibe testing | Không dịch |
| Chat template | Chat template | Không dịch |
| System prompt | System prompt | Không dịch |
| Reward model | Reward model | Không dịch |
| KV cache | KV cache | Không dịch |
| Preference dataset | Preference dataset | Không dịch |
| Loss spike | Loss spike | Không dịch |
| Compute budget | Compute budget | Không dịch |
| Bottleneck | Bottleneck / điểm nghẽn | |
| Throughput | Throughput / thông lượng | |
| State-of-the-art | SOTA / đẳng cấp thế giới | |

---

## 6. Checklist Trước Khi Submit

Mỗi agent **PHẢI** kiểm tra trước khi hoàn thành:

- [ ] YAML frontmatter có `sidebar_position` và `sidebar_label`
- [ ] Không có ký tự `<` hoặc `>` ngoài code block (dùng `&lt;` `&gt;`)
- [ ] Tất cả link nội bộ trỏ đến file thực sự tồn tại
- [ ] Code blocks giữ nguyên 100%, không dịch
- [ ] LaTeX formulas giữ nguyên 100%
- [ ] Thuật ngữ kỹ thuật theo glossary ở Section 5
- [ ] Mermaid diagrams nằm trong ` ```mermaid ` block
- [ ] Admonitions dùng cú pháp `> [!TYPE]`
- [ ] Không có heading orphan (heading không có nội dung)
- [ ] File có ít nhất 1 Mermaid diagram hoặc bảng so sánh

---

## 7. Quy trình Multi-Agent

### Phân chia công việc
- Chia theo **chương**, không chia theo section nhỏ
- Mỗi agent nhận 3-4 chương liền kề
- Agent KHÔNG tạo file ngoài phạm vi được giao

### Xử lý xung đột
- Không có 2 agent cùng ghi vào 1 file
- Cross-references giữa chương: dùng tên file đã thống nhất

### Naming Convention cho docs/
```
docs/
├── gioi_thieu.md          # 1. Giới thiệu
├── la_ban_huan_luyen.md   # 2. La bàn Huấn luyện
├── ablation_baseline.md   # 3. Ablation & Baseline
├── thiet_ke_kien_truc.md  # 4. Thiết kế Kiến trúc
├── optimizer_sieu_tham_so.md # 5. Optimizer & Siêu tham số
├── scaling_laws.md        # 6. Scaling Laws
├── xu_ly_du_lieu.md       # 7. Xử lý Dữ liệu
├── marathon_huan_luyen.md # 8. Marathon Huấn luyện
├── post_training_tong_quan.md # 9. Post-training Tổng quan
├── sft.md                 # 10. SFT
├── dpo.md                 # 11. DPO & Preference
├── grpo_rl.md             # 12. GRPO & RL
├── model_merging.md       # 13. Model Merging
├── ha_tang_gpu.md         # 14. Hạ tầng GPU
└── ket_luan.md            # 15. Kết luận
```
