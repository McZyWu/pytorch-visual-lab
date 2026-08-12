import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TorchScope · PyTorch 中文可视化学习图谱",
  description: "通过公式、变量解析、代码示例与可交互 Tensor 实验台学习 PyTorch。",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
