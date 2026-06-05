import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Cẩm nang Huấn luyện Smol',
  tagline: 'Bản dịch tiếng Việt "The Smol Training Playbook" của HuggingFace — Bí quyết xây dựng LLM đẳng cấp thế giới',
  favicon: 'img/logo.svg',

  url: 'https://tuandung222.github.io',
  baseUrl: '/smol-vi/',

  organizationName: 'tuandung222',
  projectName: 'smol-vi',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',




  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl: 'https://github.com/tuandung222/smol-vi/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn',
      crossorigin: 'anonymous',
    },
  ],

  headTags: [
    {
      tagName: 'meta',
      attributes: {
        name: 'robots',
        content: 'noindex, nofollow',
      },
    },
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'Smol Training Playbook VI',
      logo: {
        alt: 'Smol VI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Tài liệu',
        },
        {
          href: 'https://github.com/tuandung222/smol-vi',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Nền tảng',
          items: [
            {label: 'Giới thiệu', to: '/docs/gioi_thieu'},
            {label: 'La bàn Huấn luyện', to: '/docs/la_ban_huan_luyen'},
            {label: 'Ablation & Baseline', to: '/docs/ablation_baseline'},
            {label: 'Thiết kế Kiến trúc', to: '/docs/thiet_ke_kien_truc'},
          ],
        },
        {
          title: 'Huấn luyện',
          items: [
            {label: 'Optimizer & Siêu tham số', to: '/docs/optimizer_sieu_tham_so'},
            {label: 'Scaling Laws', to: '/docs/scaling_laws'},
            {label: 'Xử lý Dữ liệu', to: '/docs/xu_ly_du_lieu'},
            {label: 'Marathon Huấn luyện', to: '/docs/marathon_huan_luyen'},
          ],
        },
        {
          title: 'Post-training',
          items: [
            {label: 'Tổng quan Post-training', to: '/docs/post_training_tong_quan'},
            {label: 'SFT', to: '/docs/sft'},
            {label: 'DPO', to: '/docs/dpo'},
            {label: 'GRPO & RL', to: '/docs/grpo_rl'},
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {label: 'Model Merging', to: '/docs/model_merging'},
            {label: 'Hạ tầng GPU', to: '/docs/ha_tang_gpu'},
            {label: 'Kết luận', to: '/docs/ket_luan'},
            {label: 'Bài viết gốc (HuggingFace)', href: 'https://huggingface.co/spaces/HuggingFaceTB/smol-course'},
          ],
        },
      ],
      copyright: `Bản dịch tiếng Việt © ${new Date().getFullYear()} • Nội dung gốc thuộc HuggingFace`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml'],
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
