---
sidebar_position: 9
sidebar_label: "Post-training: Tổng quan"
---

# Post-training trong năm 2025

Pretraining đã mang lại cho SmolLM3 khả năng thô, nhưng trước khi GPU kịp nguội, chúng ta bước vào biên giới tiếp theo của năng lực mô hình: **post-training** (huấn luyện hậu kỳ). Giai đoạn này bao gồm Supervised Fine-Tuning (SFT — tinh chỉnh có giám sát), Reinforcement Learning (RL — học tăng cường), model merging (hợp nhất mô hình), và nhiều kỹ thuật khác — tất cả đều nhằm thu hẹp khoảng cách giữa "một mô hình dự đoán văn bản" và "một mô hình mà con người thực sự có thể sử dụng."

Nếu pretraining là việc nhồi nhét kiến thức vào trọng số bằng sức mạnh tính toán thô, thì post-training là nghệ thuật *điêu khắc* năng lực thô đó thành thứ gì đó đáng tin cậy và có thể điều khiển được. Và cũng giống như pretraining, những bài báo post-training bóng bẩy không thể phản ánh hết những bất ngờ lúc nửa đêm: GPU cháy, data mixture (hỗn hợp dữ liệu) khó tính, hay cách một quyết định chat template tưởng chừng nhỏ nhặt lại ảnh hưởng lan tỏa khắp các benchmark downstream.

Trong chương này, chúng tôi sẽ trình bày cách điều hướng thế giới hỗn loạn của post-training để biến SmolLM3 từ một base model mạnh thành một **hybrid reasoner** (mô hình suy luận lai) đạt đẳng cấp state-of-the-art.

## Mô hình suy luận lai (Hybrid Reasoning Model)

Một hybrid reasoning model hoạt động ở hai chế độ riêng biệt: một cho phản hồi ngắn gọn, trực tiếp và một cho suy luận từng bước mở rộng. Thông thường, chế độ hoạt động được người dùng thiết lập trong system message. Theo cách tiếp cận của Qwen3, chúng tôi làm rõ điều này bằng các lệnh nhẹ: **`/think`** kích hoạt suy luận mở rộng, trong khi **`/no_think`** bắt buộc câu trả lời ngắn gọn. Bằng cách này, người dùng kiểm soát liệu mô hình ưu tiên chiều sâu hay tốc độ.

## La bàn Post-Training: Tại sao → Cái gì → Như thế nào

Giống như pretraining, post-training cần một la bàn rõ ràng để tránh lãng phí chu kỳ nghiên cứu và kỹ thuật:

### 1. Tại sao cần post-train?

Ba động lực huấn luyện mà chúng tôi đã nêu trong [la bàn huấn luyện](./la_ban_huan_luyen.md) cũng áp dụng cho post-training. Ví dụ: bạn có thể đang khám phá liệu RL có thể mở khóa khả năng suy luận mới (nghiên cứu), hoặc cần distill (chưng cất) mô hình lớn thành mô hình nhỏ hơn vì yêu cầu độ trễ (sản xuất), hoặc bạn nhận thấy một khoảng trống chưa có mô hình mở mạnh nào lấp đầy (mã nguồn mở chiến lược).

Nhưng trước khi vươn tay ra lấy GPU, hãy tự hỏi:

- **Bạn có thực sự cần post-train không?** Nhiều mô hình open-weight (trọng số mở) hiện nay đã sánh ngang với mô hình thương mại trên nhiều tác vụ. Nếu bạn muốn một trợ lý tổng quát, một mô hình có sẵn trên [Hugging Face Hub](https://huggingface.co/models) có thể đã đáp ứng nhu cầu.
- **Bạn có dữ liệu chuyên biệt chất lượng cao không?** Post-training hữu ích nhất khi bạn nhắm đến một tác vụ hoặc lĩnh vực cụ thể mà mô hình tổng quát hoạt động kém.
- **Bạn có thể đo lường thành công không?** Không có tiêu chí đánh giá rõ ràng, bạn sẽ không biết post-training có thực sự giúp ích hay không.

### 2. Post-training cần đạt được gì?

Điều này phụ thuộc vào ưu tiên của bạn:

- Một mô hình tuân thủ chỉ thị chính xác, hiếm khi đi lạc đề?
- Một trợ lý linh hoạt có thể chuyển đổi giọng điệu và vai trò theo yêu cầu?
- Một engine suy luận có thể giải quyết bài toán, viết code, hoặc xử lý vấn đề agentic?
- Một mô hình có thể đối thoại đa ngôn ngữ?

### 3. Làm thế nào để đạt được?

Đây là nơi công thức (recipe) quan trọng. Chúng ta sẽ đề cập:

- **SFT** — để cài đặt các khả năng cốt lõi
- **Preference Optimization (PO — tối ưu hóa sở thích)** — để học trực tiếp từ sở thích con người hoặc AI
- **Reinforcement Learning (RL)** — để tinh chỉnh độ tin cậy và suy luận vượt xa dữ liệu giám sát
- **Data curation (quản lý dữ liệu)** — để cân bằng đúng giữa đa dạng và chất lượng
- **Evaluation (đánh giá)** — để theo dõi tiến độ và phát hiện thoái lui sớm

### Áp dụng cho SmolLM3

- **Tại sao?** Chúng tôi có một base model cần post-training trước khi phát hành. Đồng thời, các mô hình hybrid reasoning như Qwen3 ngày càng phổ biến, nhưng các recipe mở cho thấy cách huấn luyện chúng vẫn rất hiếm.
- **Cái gì?** Huấn luyện một hybrid reasoning model phù hợp với thế mạnh của SmolLM3, đặc biệt là chất lượng suy luận phải tốt trên nhiều ngôn ngữ ngoài tiếng Anh. Tool calling và long context cũng là yêu cầu cốt lõi.
- **Như thế nào?** Đó là nội dung của phần còn lại trong chương này!

> [!IMPORTANT]
> Khác biệt quan trọng trong cách ablation giữa pretraining và post-training: Trong pretraining, "nhỏ" thường có nghĩa là mô hình và dataset nhỏ hơn. Trong post-training, "nhỏ" có nghĩa là **dataset nhỏ hơn và thuật toán đơn giản hơn**. Chúng tôi hầu như không bao giờ sử dụng base model khác cho ablation vì hành vi phụ thuộc quá nhiều vào mô hình, và các lần chạy đủ ngắn để lặp lại trực tiếp trên mô hình mục tiêu.

## Công cụ của nghề: So sánh các Framework

Đằng sau mỗi recipe post-training là một hộp công cụ gồm các framework và thư viện cho phép thử nghiệm quy mô lớn. Bảng sau tóm tắt các lĩnh vực hỗ trợ chính:

| Framework | SFT | PO/DPO | RL (GRPO, PPO…) | FullFT | LoRA | Multi-modal | Distributed |
|-----------|-----|--------|------------------|--------|------|-------------|-------------|
| [TRL](https://github.com/huggingface/trl) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [OpenInstruct](https://github.com/allenai/open-instruct) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [Unsloth](https://github.com/unslothai/unsloth) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [vERL](https://github.com/volcengine/verl) | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [Prime RL](https://github.com/PrimeIntellect-ai/prime-rl) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [PipelineRL](https://github.com/ServiceNow/PipelineRL) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [ART](https://github.com/OpenPipe/ART/tree/main) | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [TorchForge](https://github.com/meta-pytorch/torchforge) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [NemoRL](https://github.com/NVIDIA-NeMo/RL) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

Trong đó:
- **FullFT** = Full Fine-Tuning — cập nhật toàn bộ tham số mô hình trong quá trình huấn luyện
- **LoRA** = Low-Rank Adaptation — phương pháp tiết kiệm tham số, chỉ cập nhật các ma trận low-rank nhỏ trong khi giữ nguyên mô hình gốc
- **Multi-modal** = hỗ trợ huấn luyện trên các phương thức ngoài văn bản (ví dụ: hình ảnh)
- **Distributed** = khả năng huấn luyện mô hình trên nhiều GPU

Tại Hugging Face, chúng tôi phát triển và duy trì **TRL**, vì vậy đó là framework được chọn và là framework được sử dụng để post-train SmolLM3.

> [!TIP]
> Chúng tôi thường chạy thí nghiệm trên một **fork nội bộ** của TRL. Điều này cho phép thêm tính năng mới rất nhanh, sau đó upstream vào thư viện chính. Nếu bạn thoải mái làm việc với nội bộ framework của mình, áp dụng quy trình tương tự có thể là cách tiếp cận mạnh mẽ để lặp lại nhanh.

## Tại sao cần dùng Framework?

Có một nhóm nhà nghiên cứu thích than phiền về việc sử dụng framework huấn luyện và cho rằng bạn nên tự implement mọi thứ từ đầu, mọi lúc. Lập luận ngầm ở đây là "hiểu biết thực sự" chỉ đến từ việc tự implement lại mọi thuật toán RL, tự code mọi primitive huấn luyện phân tán, hoặc hack một eval harness tạm bợ.

Nhưng quan điểm này bỏ qua thực tế của nghiên cứu và sản xuất hiện đại. Lấy RL làm ví dụ: các thuật toán như PPO và GRPO nổi tiếng là khó implement đúng, và những lỗi nhỏ trong normalization hay KL penalty (phạt Kullback-Leibler) có thể dẫn đến lãng phí hàng ngày compute và công sức.

Tương tự, mặc dù có thể hấp dẫn khi viết một implementation single-file của thuật toán nào đó, liệu script đó có thể scale từ 1B lên 100B+ tham số không?

**Framework tồn tại chính vì những điều cơ bản đã được hiểu rõ**, và việc liên tục phát minh lại chúng là lãng phí thời gian. Không phải rằng không có giá trị trong việc tìm hiểu low-level — implement PPO từ đầu một lần là bài tập tuyệt vời. Nhưng trong hầu hết trường hợp, bạn nên chọn một framework mà mình thích và hack nó theo mục đích của mình.

## Đánh giá cho Post-Training

### Bước đầu tiên: Evals trước mọi thứ

Bước đầu tiên trong post-training — giống như trong pretraining — là quyết định bộ eval phù hợp. Vì hầu hết LLM ngày nay được sử dụng như trợ lý, chúng tôi nhận thấy rằng nhắm đến một mô hình "hoạt động tốt" là mục tiêu tốt hơn so với việc đuổi theo các benchmark trừu tượng về "trí thông minh." Vậy một trợ lý tốt cần làm gì? Tối thiểu, nó cần có khả năng:

- Xử lý chỉ thị mơ hồ
- Lên kế hoạch từng bước
- Viết code
- Gọi tool (công cụ) khi thích hợp

### Bộ Eval phân tầng

Tại Hugging Face, chúng tôi sử dụng bộ eval phân tầng:

**1. Capability evals (đánh giá năng lực cốt lõi):**

| Lĩnh vực | Benchmark | Mô tả |
|-----------|-----------|-------|
| Kiến thức | GPQA Diamond | Câu hỏi trắc nghiệm cấp thạc sĩ/tiến sĩ; chưa bão hòa ở mô hình nhỏ |
| Toán | AIME 2025, MATH-500 | MATH-500 là kiểm tra sanity; AIME 2025 để đo lường thực sự |
| Code | LiveCodeBench | Theo dõi năng lực lập trình; cải thiện chuyển hóa thành mô hình code tốt hơn |
| Đa ngôn ngữ | Global MMLU, MGSM | Kiểm tra đa ngôn ngữ cho các ngôn ngữ mục tiêu |

**2. Integrated task evals (đánh giá tác vụ tích hợp):**

| Lĩnh vực | Benchmark | Mô tả |
|-----------|-----------|-------|
| Long context | RULER, HELMET | Toàn diện hơn Needle in a Haystack đơn giản |
| Instruction following | IFEval, IFBench, Multi-IF | IFEval phổ biến nhất; IFBench giảm benchmaxxing |
| Alignment | AlpacaEval, ArenaHard, MixEval | MixEval tương quan mạnh nhất với Elo rating |
| Tool calling | BFCL, TAU-Bench | Đánh giá khả năng gọi tool và giải quyết vấn đề |

**3. Overfitting prevention evals (phòng ngừa overfitting):** Bao gồm GSMPlus — biến thể perturbation (nhiễu loạn) của GSM8k để kiểm tra khả năng tổng quát hóa.

**4. Internal evals (đánh giá nội bộ):** Không có gì thay thế được việc tự xây eval nội bộ cho các năng lực cụ thể.

**5. Vibe evals:** "Vibe testing" — tương tác trực tiếp với mô hình — cực kỳ quan trọng để phát hiện quirk (hành vi lạ) không bị bắt bởi điểm eval.

### Quy tắc thực hành cho Eval

- Dùng **tập con nhỏ** để tăng tốc eval trong quá trình phát triển
- Với **reasoning model**, loại bỏ chain of thought khỏi output được chấm điểm
- Nếu eval dùng LLM judges, **cố định judge và phiên bản** để so sánh nhất quán
- Cẩn thận với **contamination** (ô nhiễm dữ liệu) trong base model
- Giữ một bộ **held-out benchmarks** cho báo cáo mô hình cuối cùng
- Với eval có ít bài (&lt;2k), sample $k$ lần và báo cáo $\text{avg}@k$
- Khi implement eval mới, đảm bảo **tái tạo được kết quả đã công bố**

> [!WARNING]
> Câu chuyện trên gợi ý rằng chúng tôi tập hợp nhóm, thống nhất bộ eval, và chuẩn bị sẵn sàng trước khi bắt đầu huấn luyện. Thực tế lộn xộn hơn nhiều: chúng tôi có deadline gấp và lao vào huấn luyện mô hình trước khi nhiều eval trên được implement. Nhìn lại, đây là sai lầm — hãy **ưu tiên eval trước mọi thứ khác!**
