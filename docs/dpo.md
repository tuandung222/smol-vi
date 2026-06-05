---
sidebar_position: 11
sidebar_label: "DPO & Preference"
---

# Từ SFT đến Tối ưu hóa Sở thích (Preference Optimization)

## Tại sao cần vượt ra ngoài SFT?

Mặc dù bạn có thể tiếp tục mở rộng SFT với nhiều dữ liệu hơn, đến một lúc nào đó bạn sẽ quan sát thấy diminishing returns (lợi nhuận giảm dần) hoặc failure mode (chế độ lỗi) — như mô hình không thể tự sửa code lỗi của mình. Tại sao?

Vì SFT là một dạng **imitation learning** (học bắt chước): mô hình chỉ học tái tạo pattern trong dữ liệu huấn luyện. Nếu dữ liệu chưa chứa cách sửa tốt, hoặc nếu hành vi mong muốn khó khơi gợi bằng distillation, mô hình không có tín hiệu rõ ràng về cái gì là "tốt hơn."

Vấn đề vẫn tồn tại ngay cả khi dataset chứa hỗn hợp cân bằng các trace — một số đi thẳng đến đáp án đúng, một số mắc lỗi rồi tự sửa. Trong trường hợp này, mô hình có thể đơn giản *học rằng mắc lỗi ban đầu là một phần của pattern mong muốn*. Nhưng điều chúng ta thực sự muốn là mô hình tạo ra đáp án đúng ngay từ đầu.

Đây là lúc **Preference Optimization** (tối ưu hóa sở thích) phát huy tác dụng. Thay vì chỉ sao chép demonstration, chúng ta cung cấp cho mô hình phản hồi so sánh: *"phản hồi A tốt hơn phản hồi B."* Những sở thích này cung cấp tín hiệu huấn luyện trực tiếp hơn cho chất lượng, cho phép hiệu suất mô hình scale vượt xa giới hạn của SFT.

> [!NOTE]
> Preference optimization thường yêu cầu **ít dữ liệu hơn nhiều** so với SFT, vì điểm bắt đầu đã là mô hình khá tốt — có thể tuân thủ chỉ thị và có kiến thức từ các giai đoạn huấn luyện trước.

## Tạo Preference Dataset

Trong lịch sử, preference dataset được tạo bằng cách cho annotator con người đánh giá cặp phản hồi của mô hình. Cách tiếp cận này vẫn được sử dụng nhưng **cực kỳ đắt đỏ và scale kém**. Gần đây, LLM đã đủ năng lực để tạo preference chất lượng cao. Trong thực tế, có hai cách tiếp cận phổ biến:

### Cách 1: Strong vs. Weak (Mạnh vs. Yếu)

1. Lấy tập prompt cố định $x$
2. Sinh một phản hồi từ mô hình yếu/baseline, và một từ mô hình mạnh
3. Gán output mô hình mạnh là **chosen** (được chọn) $y_c$ và mô hình yếu là **rejected** (bị từ chối) $y_r$

Điều này tạo ra dataset so sánh $\{x, y_c, y_r\}$ — đơn giản vì chúng ta giả định output mô hình mạnh luôn tốt hơn.

> **Ví dụ:** Intel lấy SFT dataset với phản hồi từ GPT-3.5 và GPT-4, chuyển đổi thành preference dataset bằng cách chọn GPT-4 làm chosen và GPT-3.5 làm rejected.

### Cách 2: On-Policy với Grading (Chấm điểm)

1. Dùng **chính mô hình bạn sẽ train** để sinh nhiều phản hồi cho cùng prompt — tạo dữ liệu "on-policy" phản ánh phân phối output tự nhiên
2. Giới thiệu **external grader** (bộ chấm điểm bên ngoài): verifier hoặc reward model chấm điểm theo nhiều trục chất lượng
3. Grader gán preference label, tạo ra preference dataset linh hoạt và tinh tế hơn

### Preference Dataset của SmolLM3

Tại thời điểm phát triển SmolLM3, chưa tồn tại preference data với reasoning traces, nên chúng tôi tự tạo bằng cách "strong vs. weak":

- **Prompt:** từ Tulu 3 preference mixture của Ai2
- **Strong:** Qwen3-32B ở chế độ `/think`
- **Weak:** Qwen3-0.6B ở chế độ `/think`
- **Kết quả:** [dataset 250k+ LLM-generated preferences](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)

## DPO là gì và hoạt động ra sao

**Direct Preference Optimization (DPO)** là thuật toán preference optimization đầu tiên được áp dụng rộng rãi trong open source.

Khi paper DPO ra mắt giữa năm 2023, đã có tranh luận sôi nổi online về việc liệu nó có sánh được phương pháp RL hay không. Để giải quyết điều đó, Hugging Face phát hành [Zephyr 7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) — huấn luyện toàn bộ trên synthetic data và cho thấy cải thiện đáng kể từ DPO.

Sức hấp dẫn của DPO đến từ: **đơn giản để implement, ổn định trong thực tế, và hiệu quả ngay cả với lượng preference data khiêm tốn.** DPO đã trở thành phương pháp mặc định để cải thiện SFT model trước khi sử dụng kỹ thuật phức tạp hơn như RL.

### Các biến thể đáng chú ý

Các nhà nghiên cứu nhanh chóng phát hiện nhiều cách cải thiện DPO:

| Thuật toán | Ý tưởng cốt lõi | Khi nào dùng |
|------------|-----------------|--------------|
| **KTO** (Kahneman–Tversky Optimization) | Mô hình hóa phản hồi "mong muốn" hay không, dựa trên lý thuyết quyết định | Khi không có paired preference data (chỉ có 👍/👎) |
| **ORPO** (Odds Ratio Preference Optimization) | Tích hợp preference optimization vào SFT bằng odds ratio | Không cần reference model hay SFT riêng |
| **APO** (Anchored Preference Optimization) | Regularize rõ ràng mức thay đổi likelihood cho chosen vs. rejected | Kiểm soát tốt hơn khi cần điều chỉnh tinh tế |

> [!TIP]
> May mắn thay, chúng ta có thể chuyển đổi giữa nhiều thuật toán chỉ với **một dòng code thay đổi** trong `DPOTrainer` của TRL!

## On-policy vs Off-policy DPO

DPO gốc là phương pháp **off-policy**: train trên preference dataset cố định, thu thập trước. Điều này có nghĩa preference data có thể không phản ánh hành vi hiện tại của mô hình, dẫn đến distribution drift (trôi phân phối).

**Online DPO** khắc phục điều này bằng cách cho mô hình liên tục sinh phản hồi mới, thu thập preference label mới (từ reward model hoặc LLM grader), và tự cập nhật. Điều này giữ optimization on-policy và giảm drift giữa training data và hành vi hiện tại.

Nghiên cứu từ FAIR đã so sánh hiệu quả giữa fully off-policy và on-policy cho DPO, cho thấy **có thể match hiệu suất GRPO bằng compute ít hơn đáng kể** với online DPO, đặc biệt cho tác vụ toán.

## Thiết lập thí nghiệm DPO ban đầu

Cho baseline DPO ban đầu, chúng tôi thực hiện:

1. **Dữ liệu `/no_think`:** Dùng prompt và completion từ [Tulu 3 Preference Personas IF dataset](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)
2. **Dữ liệu `/think`:** Tái sử dụng prompt, sinh cặp "strong vs. weak" với Qwen3-32B và Qwen3-0.6B
3. **Train 1 epoch** và đo cải thiện in-domain (IFEval) cùng tác động out-of-domain

**Kết quả:** Cải thiện in-domain cho cả hai chế độ reasoning rất đáng kể — trên IFEval, **APO-zero cải thiện 15-20 điểm phần trăm** so với SFT checkpoint!

> [!IMPORTANT]
> Preference optimization không chỉ làm mô hình hữu ích hoặc aligned hơn — nó **dạy mô hình suy luận tốt hơn**. Nếu bạn cần cách nhanh để cải thiện reasoning model, thử sinh preference "strong vs. weak" và ablate các loss function khác nhau!

## Hyperparameter quan trọng nhất

Cho preference optimization, thường chỉ có **ba hyperparameter** ảnh hưởng đáng kể:

### 1. Learning Rate

| Quy tắc | Chi tiết |
|---------|---------|
| **Nguyên tắc chung** | ~10× nhỏ hơn LR dùng cho SFT |
| **Dải khuyến nghị** | 5× đến 20× nhỏ hơn SFT LR |
| **SmolLM3** | SFT LR = 2e-5 → DPO LR tối ưu ≈ 1e-6 |

Learning rate lớn hơn giới hạn 10× dẫn đến hiệu suất kém hơn cho chế độ extended thinking.

### 2. Tham số β

Tham số β kiểm soát mức margin giữa preference pairs. Kết quả thí nghiệm:

- **β = 0.1:** Hiệu suất cao nhất cho cả hai chế độ reasoning
- **β &lt; 0.1:** Ảnh hưởng tiêu cực, mô hình kém hơn SFT checkpoint
- **β &gt; 0.1:** Ổn định, nhưng cẩn thận với giá trị quá cao

> Giá trị β lớn hơn 0.1 ưu tiên align mô hình với preference data hơn là ở gần reference model. Khuyến nghị scan β trong dải **0.01–0.5**.

### 3. Kích thước dữ liệu

Thí nghiệm từ 2k đến 340k preference pairs cho thấy hiệu suất **khá ổn định** trên toàn dải. Giảm nhẹ ở extended thinking cho dataset &gt;100k pairs, nhưng không nghiêm trọng.

> [!TIP]
> Cho các dự án tương lai, biết rằng dataset nhỏ hơn cũng tạo ra cải thiện có nghĩa chúng ta có thể thử nghiệm với dataset nhỏ trong giai đoạn iteration để nhanh chóng xác định cấu hình hứa hẹn nhất.

## Kết quả DPO của SmolLM3

Kết hợp tất cả các luồng trên tạo ra mô hình SmolLM3-3B cuối cùng: **best-in-class cho kích thước của nó** và nằm trên Pareto front cùng với mô hình hybrid reasoning của Qwen.

## Quy tắc thực hành

Tóm tắt các phát hiện về preference optimization hữu ích cho dự án tương lai:

1. **Đừng ngại tự tạo preference data!** Với inference ngày càng rẻ, việc sinh LLM preferences đơn giản và hiệu quả chi phí.

2. **Chọn DPO làm baseline ban đầu** và iterate từ đó. Tùy loại preference data, các thuật toán khác (ORPO, KTO, APO) có thể cải thiện đáng kể so với DPO.

3. **Dùng learning rate ~10× nhỏ hơn** so với SFT.

4. **Scan β**, thường trong dải 0.01–0.5.

5. **Partition data và train iteratively** — hầu hết thuật toán preference overfit sau 1 epoch.

> [!NOTE]
> Preference optimization thường là sweet spot giữa đơn giản và hiệu suất, nhưng nó kế thừa một hạn chế quan trọng: **chỉ tốt bằng offline preference data bạn có thể thu thập.** Đến một lúc nào đó, dataset tĩnh hết tín hiệu và bạn cần phương pháp có thể sinh training feedback mới online. Đó là nơi preference optimization gặp gia đình rộng hơn của on-policy và RL-based methods — chủ đề của [chương tiếp theo](./grpo_rl.md).
