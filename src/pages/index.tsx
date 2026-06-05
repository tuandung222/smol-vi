import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

/* ───────── Data ───────── */

const highlights = [
  {
    emoji: '🧠',
    badge: 'Mô hình',
    title: 'SmolLM3 3B',
    desc: 'Hành trình huấn luyện từ 0 → 11 nghìn tỷ token, đạt hiệu suất vượt trội so với các mô hình cùng kích thước.',
  },
  {
    emoji: '🏗️',
    badge: 'Kiến trúc',
    title: 'Architecture',
    desc: 'Quyết định thiết kế dựa trên ablation: GQA, embedding tying, RoPE, và hơn 200 thí nghiệm.',
  },
  {
    emoji: '📊',
    badge: 'Dữ liệu',
    title: 'Data Curation',
    desc: 'Nghệ thuật pha trộn dữ liệu web, code, toán học — với chiến lược annealing và upsampling tinh tế.',
  },
  {
    emoji: '⚡',
    badge: 'Tinh chỉnh',
    title: 'Post-training',
    desc: 'Pipeline hoàn chỉnh SFT → DPO → GRPO, kết hợp model merging để tối đa hóa hiệu suất.',
  },
];

const chapters = [
  {
    num: 1, emoji: '👋', title: 'Giới thiệu', badge: 'Tổng quan', to: '/docs/gioi_thieu',
    desc: 'Tổng quan về SmolLM3 và playbook huấn luyện LLM từ đầu đến cuối.',
  },
  {
    num: 2, emoji: '🧭', title: 'La bàn Huấn luyện', badge: 'Chiến lược', to: '/docs/la_ban_huan_luyen',
    desc: 'Phương pháp luận, triết lý đánh giá, và chiến lược tổng thể.',
  },
  {
    num: 3, emoji: '🧪', title: 'Ablation & Baseline', badge: 'Thí nghiệm', to: '/docs/ablation_baseline',
    desc: 'Cách thiết kế baseline, chạy ablation studies hiệu quả.',
  },
  {
    num: 4, emoji: '🏛️', title: 'Thiết kế Kiến trúc', badge: 'Kiến trúc', to: '/docs/thiet_ke_kien_truc',
    desc: 'GQA, embedding tying, NTK-aware RoPE, và các quyết định kiến trúc.',
  },
  {
    num: 5, emoji: '⚙️', title: 'Optimizer & Siêu tham số', badge: 'Tối ưu', to: '/docs/optimizer_sieu_tham_so',
    desc: 'AdamW, WSD scheduler, batch size scaling, và tuning siêu tham số.',
  },
  {
    num: 6, emoji: '📐', title: 'Scaling Laws', badge: 'Lý thuyết', to: '/docs/scaling_laws',
    desc: 'Chinchilla, quy luật scaling, và dự đoán hiệu suất mô hình.',
  },
  {
    num: 7, emoji: '🗃️', title: 'Nghệ thuật Xử lý Dữ liệu', badge: 'Dữ liệu', to: '/docs/xu_ly_du_lieu',
    desc: 'Pipeline xử lý FineWeb, code, toán — dedup, filtering, mixing.',
  },
  {
    num: 8, emoji: '🏃', title: 'Cuộc Marathon Huấn luyện', badge: 'Huấn luyện', to: '/docs/marathon_huan_luyen',
    desc: 'Huấn luyện 11T token: mid-training, annealing, và long-context.',
  },
  {
    num: 9, emoji: '🎯', title: 'Post-training: Tổng quan', badge: 'Post-training', to: '/docs/post_training_tong_quan',
    desc: 'Tổng quan pipeline tinh chỉnh: SFT, DPO, GRPO, và evaluation.',
  },
  {
    num: 10, emoji: '📚', title: 'SFT: Tinh chỉnh có Giám sát', badge: 'SFT', to: '/docs/sft',
    desc: 'Supervised Fine-Tuning với dữ liệu đa dạng và chiến lược chất lượng.',
  },
  {
    num: 11, emoji: '⚖️', title: 'DPO & Preference Optimization', badge: 'DPO', to: '/docs/dpo',
    desc: 'Direct Preference Optimization — căn chỉnh theo sở thích con người.',
  },
  {
    num: 12, emoji: '🎮', title: 'GRPO & RL', badge: 'RL', to: '/docs/grpo_rl',
    desc: 'Group Relative Policy Optimization cho reasoning và math.',
  },
  {
    num: 13, emoji: '🔀', title: 'Model Merging & Reasoning', badge: 'Nâng cao', to: '/docs/model_merging',
    desc: 'SLERP, DARE-TIES merging — kết hợp nhiều checkpoint tối ưu.',
  },
  {
    num: 14, emoji: '🖥️', title: 'Hạ tầng GPU', badge: 'Hạ tầng', to: '/docs/ha_tang_gpu',
    desc: 'Slurm, FSDP2, cấu hình cluster và quản lý tài nguyên GPU.',
  },
  {
    num: 15, emoji: '🏁', title: 'Kết luận', badge: 'Kết luận', to: '/docs/ket_luan',
    desc: 'Tổng kết bài học, nhìn lại hành trình, và hướng đi tương lai.',
  },
];

/* ───────── Hero Section ───────── */

function HeroSection(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('custom-hero', styles.hero)}>
      <div className={styles.heroInner}>
        <span className={styles.heroBadge}>
          BẢN DỊCH TIẾNG VIỆT • HUGGING FACE
        </span>
        <h1 className={styles.heroTitle}>
          <span className="gradient-text">Cẩm nang Huấn luyện</span>
          <br />
          Smol
        </h1>
        <p className={styles.heroTagline}>{siteConfig.tagline}</p>
        <div className={styles.heroCtas}>
          <Link className={styles.ctaPrimary} to="/docs/gioi_thieu">
            Bắt đầu đọc 📖
          </Link>
          <Link
            className={styles.ctaSecondary}
            href="https://huggingface.co/spaces/HuggingFaceTB/smol-course"
          >
            Bài viết gốc 🔗
          </Link>
        </div>
      </div>
    </header>
  );
}

/* ───────── Key Highlights ───────── */

function KeyHighlights(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>
        <span className="gradient-text">Điểm nổi bật</span>
      </h2>
      <p className={styles.sectionSubtitle}>
        Những insight quan trọng nhất từ hành trình xây dựng SmolLM3
      </p>
      <div className={styles.highlightsGrid}>
        {highlights.map((h, i) => (
          <div
            key={i}
            className={clsx('glass-panel', styles.highlightCard)}
            style={{animationDelay: `${i * 0.1}s`}}
          >
            <span className={styles.highlightEmoji}>{h.emoji}</span>
            <span className={styles.highlightBadge}>{h.badge}</span>
            <h3 className={styles.highlightTitle}>{h.title}</h3>
            <p className={styles.highlightDesc}>{h.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ───────── Chapter List ───────── */

function ChapterList(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>
        <span className="gradient-text">Mục lục</span>
      </h2>
      <p className={styles.sectionSubtitle}>
        15 chương chi tiết — từ thiết kế kiến trúc đến post-training
      </p>
      <div className={styles.chaptersGrid}>
        {chapters.map((ch) => (
          <Link
            key={ch.num}
            to={ch.to}
            className={clsx('glass-panel', styles.chapterCard)}
          >
            <div className={styles.chapterHeader}>
              <span className={styles.chapterNumber}>{ch.num}</span>
              <span className={styles.chapterBadge}>{ch.badge}</span>
            </div>
            <h3 className={styles.chapterTitle}>
              {ch.emoji} {ch.title}
            </h3>
            <p className={styles.chapterDesc}>{ch.desc}</p>
            <span className={styles.chapterArrow}>Đọc chương →</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

/* ───────── Bottom CTA ───────── */

function BottomCta(): React.JSX.Element {
  return (
    <section className={clsx('custom-hero', styles.bottomCta)}>
      <div className={styles.bottomCtaInner}>
        <h2 className={styles.bottomCtaTitle}>
          🚀 Trải nghiệm <span className="gradient-text">SmolLM3</span>
        </h2>
        <p className={styles.bottomCtaDesc}>
          Khám phá mô hình ngôn ngữ 3B tham số mạnh mẽ nhất trong phân khúc — được huấn luyện với mọi kỹ thuật trong playbook này.
        </p>
        <Link
          className={styles.ctaPrimary}
          href="https://huggingface.co/HuggingFaceTB/SmolLM3-3B"
        >
          Thử SmolLM3 trên HuggingFace 🤗
        </Link>
      </div>
    </section>
  );
}

/* ───────── Page Layout ───────── */

export default function Home(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      <HeroSection />
      <main>
        <KeyHighlights />
        <ChapterList />
        <BottomCta />
      </main>
    </Layout>
  );
}
