"""
IEDB 线性B细胞表位数据集下载与预处理脚本

使用方法:
    1. 手动下载: 访问 https://www.iedb.org/database_export_v3.php
       - 选择 "B Cell Assays" (B细胞检测)
       - 筛选 Epitope Type = "Linear peptide"
       - 导出为CSV格式
       - 将下载的文件保存为 data/bcell_full_v3.csv

    2. 运行本脚本进行预处理:
       python prepare_data.py --input data/bcell_full_v3.csv

    3. 或者使用IEDB API自动下载 (需网络通畅):
       python prepare_data.py --download

数据集说明 (论文原文):
    - 数据来源: IEDB (Immune Epitope Database)
    - 正样本: 实验验证的线性B细胞表位 (Linear BCE)
    - 负样本: 非线性B细胞表位 (Non-linear BCE)
    - 序列长度: 5-25个氨基酸
    - 使用 CD-HIT (阈值 0.7) 去除同源序列
    - 训练集: 4,440 正样本 + 5,485 负样本
    - 测试集: 1,110 正样本 + 1,408 负样本 (独立测试集)
"""

import os
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 25
CDHIT_THRESHOLD = 0.7
TEST_RATIO = 0.2
RANDOM_SEED = 42


def download_from_iedb(output_dir="data"):
    """
    尝试通过 IEDB API 下载B细胞表位数据。
    IEDB API 文档: https://wiki.iedb.org/index.php/IEDB-API_2.0

    注意: IEDB API 可能有访问限制，若下载失败请手动下载。
    """
    os.makedirs(output_dir, exist_ok=True)

    # IEDB API 查询线性B细胞表位
    api_urls = {
        "linear_positive": (
            "https://query-api.iedb.org/bcell_search?"
            "epitope_type=Linear+peptide"
            "&bcell_type=Positive"
            "&output_format=csv"
        ),
        "linear_negative": (
            "https://query-api.iedb.org/bcell_search?"
            "epitope_type=Linear+peptide"
            "&bcell_type=Negative"
            "&output_format=csv"
        ),
    }

    for name, url in api_urls.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        print(f"正在下载 {name} ...")
        print(f"  URL: {url}")
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "MCALBCE-DataLoader/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
                with open(out_path, "wb") as f:
                    f.write(data)
            print(f"  已保存到 {out_path} ({len(data)} bytes)")
        except Exception as e:
            print(f"  下载失败: {e}")
            print(f"  请手动访问 https://www.iedb.org/database_export_v3.php 下载数据")
            print(f"  或使用以下 curl 命令尝试:")
            print(f'  curl -o {out_path} "{url}"')
            return False
    return True


def clean_sequence(seq):
    """清洗序列: 转大写, 去除非标准氨基酸"""
    if not isinstance(seq, str):
        return None
    seq = seq.upper().strip()
    if not seq:
        return None
    if not all(c in STANDARD_AMINO_ACIDS for c in seq):
        cleaned = "".join(c for c in seq if c in STANDARD_AMINO_ACIDS)
        if len(cleaned) < MIN_SEQ_LEN:
            return None
        seq = cleaned
    if len(seq) < MIN_SEQ_LEN or len(seq) > MAX_SEQ_LEN:
        return None
    return seq


def run_cdhit(input_fasta, output_fasta, threshold=CDHIT_THRESHOLD):
    """
    使用 CD-HIT 去除同源序列。
    安装: conda install -c bioconda cd-hit  或  apt install cd-hit
    """
    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-n", "4" if threshold < 0.75 else "5",
        "-M", "0",
        "-T", "4",
    ]
    print(f"运行 CD-HIT: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"CD-HIT 错误: {result.stderr}")
            return False
        print("CD-HIT 完成")
        return True
    except FileNotFoundError:
        print("警告: CD-HIT 未安装，跳过同源序列过滤。")
        print("安装方法: conda install -c bioconda cd-hit")
        return False


def sequences_to_fasta(sequences, labels, filepath):
    """将序列列表写入 FASTA 格式文件"""
    with open(filepath, "w") as f:
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            f.write(f">seq_{i}|label={label}\n{seq}\n")


def fasta_to_sequences(filepath):
    """从 FASTA 文件读取序列"""
    sequences, labels = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                label = int(line.split("label=")[1]) if "label=" in line else 0
                labels.append(label)
            elif line:
                sequences.append(line)
    return sequences, labels


def process_iedb_csv(csv_path, output_dir="data"):
    """
    处理 IEDB 导出的CSV文件。

    IEDB CSV 常见列名:
    - "Epitope - Name" 或 "Description": 表位序列
    - "Epitope - Object Type": 表位类型 (Linear peptide)
    - "Assay - Qualitative Measure": 检测结果 (Positive/Negative)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"读取文件: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"原始数据: {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")

    # 自动查找序列列
    seq_col = None
    for col in df.columns:
        cl = col.lower()
        if "description" in cl or "epitope" in cl and "name" in cl:
            seq_col = col
            break
    if seq_col is None:
        for col in df.columns:
            if "sequence" in col.lower() or "peptide" in col.lower():
                seq_col = col
                break
    if seq_col is None:
        seq_col = df.columns[0]
    print(f"使用序列列: {seq_col}")

    # 自动查找标签列 (Qualitative Measure)
    label_col = None
    for col in df.columns:
        cl = col.lower()
        if "qualitative" in cl:
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            if "label" in col.lower() or "class" in col.lower():
                label_col = col
                break

    # 自动查找表位类型列
    type_col = None
    for col in df.columns:
        cl = col.lower()
        if "object type" in cl or "epitope type" in cl:
            type_col = col
            break

    # 筛选线性肽
    if type_col:
        print(f"表位类型列: {type_col}")
        print(f"类型分布:\n{df[type_col].value_counts().head()}")
        df = df[df[type_col].str.contains("Linear", case=False, na=False)]
        print(f"筛选线性肽后: {len(df)} 条")

    # 提取正负样本
    positive_seqs = []
    negative_seqs = []

    if label_col:
        print(f"标签列: {label_col}")
        print(f"标签分布:\n{df[label_col].value_counts().head()}")
        for _, row in df.iterrows():
            seq = clean_sequence(str(row[seq_col]))
            if seq is None:
                continue
            qual = str(row[label_col]).lower()
            if "positive" in qual:
                positive_seqs.append(seq)
            elif "negative" in qual:
                negative_seqs.append(seq)
    else:
        print("未找到标签列, 所有序列视为正样本")
        for _, row in df.iterrows():
            seq = clean_sequence(str(row[seq_col]))
            if seq:
                positive_seqs.append(seq)

    # 去重
    positive_seqs = list(set(positive_seqs))
    negative_seqs = list(set(negative_seqs))
    print(f"\n去重后:")
    print(f"  正样本 (线性 BCE): {len(positive_seqs)}")
    print(f"  负样本 (非线性 BCE): {len(negative_seqs)}")

    # CD-HIT 去同源
    all_seqs = positive_seqs + negative_seqs
    all_labels = [1] * len(positive_seqs) + [0] * len(negative_seqs)

    input_fasta = os.path.join(output_dir, "all_sequences.fasta")
    output_fasta = os.path.join(output_dir, "cdhit_filtered.fasta")
    sequences_to_fasta(all_seqs, all_labels, input_fasta)

    if run_cdhit(input_fasta, output_fasta):
        filtered_seqs, filtered_labels = fasta_to_sequences(output_fasta)
        print(f"CD-HIT 过滤后: {len(filtered_seqs)} 条序列")
    else:
        print("跳过 CD-HIT, 使用去重后的序列")
        filtered_seqs = all_seqs
        filtered_labels = all_labels

    # 分离正负样本
    pos_seqs = [s for s, l in zip(filtered_seqs, filtered_labels) if l == 1]
    neg_seqs = [s for s, l in zip(filtered_seqs, filtered_labels) if l == 0]

    # 划分训练集和测试集 (8:2)
    np.random.seed(RANDOM_SEED)
    pos_idx = np.random.permutation(len(pos_seqs))
    neg_idx = np.random.permutation(len(neg_seqs))

    pos_test_n = int(len(pos_seqs) * TEST_RATIO)
    neg_test_n = int(len(neg_seqs) * TEST_RATIO)

    train_seqs = (
        [pos_seqs[i] for i in pos_idx[pos_test_n:]]
        + [neg_seqs[i] for i in neg_idx[neg_test_n:]]
    )
    train_labels = [1] * (len(pos_seqs) - pos_test_n) + [0] * (len(neg_seqs) - neg_test_n)

    test_seqs = (
        [pos_seqs[i] for i in pos_idx[:pos_test_n]]
        + [neg_seqs[i] for i in neg_idx[:neg_test_n]]
    )
    test_labels = [1] * pos_test_n + [0] * neg_test_n

    # 保存为CSV
    train_df = pd.DataFrame({"sequence": train_seqs, "label": train_labels})
    test_df = pd.DataFrame({"sequence": test_seqs, "label": test_labels})

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n===== 数据集统计 =====")
    print(f"训练集: {train_path}")
    print(f"  正样本 (线性 BCE): {sum(train_labels)}")
    print(f"  负样本 (非线性 BCE): {len(train_labels) - sum(train_labels)}")
    print(f"测试集: {test_path}")
    print(f"  正样本 (线性 BCE): {sum(test_labels)}")
    print(f"  负样本 (非线性 BCE): {len(test_labels) - sum(test_labels)}")
    print(f"\n论文参考数据量:")
    print(f"  训练集: 4,440 正 + 5,485 负")
    print(f"  测试集: 1,110 正 + 1,408 负")

    # 序列长度分布统计
    all_lens = [len(s) for s in train_seqs + test_seqs]
    print(f"\n序列长度统计:")
    print(f"  最短: {min(all_lens)}, 最长: {max(all_lens)}")
    print(f"  平均: {np.mean(all_lens):.1f}, 中位数: {np.median(all_lens):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="IEDB 线性B细胞表位数据集下载与预处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 方式1: 自动从IEDB API下载 (需网络通畅)
  python prepare_data.py --download

  # 方式2: 处理手动下载的IEDB CSV文件
  python prepare_data.py --input data/bcell_full_v3.csv

  # 方式3: 处理已有的正负样本文件
  python prepare_data.py --positive data/positive.fasta --negative data/negative.fasta

手动下载步骤:
  1. 访问 https://www.iedb.org/database_export_v3.php
  2. 在 "Assay" 中选择 "B Cell"
  3. 在 "Epitope" 中选择 "Linear peptide"
  4. 点击 "Export" 下载CSV文件
  5. 运行: python prepare_data.py --input 下载的文件.csv
        """,
    )
    parser.add_argument("--download", action="store_true", help="从 IEDB API 自动下载数据")
    parser.add_argument("--input", type=str, help="IEDB 导出的CSV文件路径")
    parser.add_argument("--positive", type=str, help="正样本 FASTA 文件路径")
    parser.add_argument("--negative", type=str, help="负样本 FASTA 文件路径")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    args = parser.parse_args()

    if args.download:
        success = download_from_iedb(args.output_dir)
        if not success:
            print("\n自动下载失败，请手动下载数据。")
            sys.exit(1)
        # 合并下载的文件并处理
        pos_path = os.path.join(args.output_dir, "linear_positive.csv")
        neg_path = os.path.join(args.output_dir, "linear_negative.csv")
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            pos_df = pd.read_csv(pos_path, low_memory=False)
            neg_df = pd.read_csv(neg_path, low_memory=False)
            combined = pd.concat([pos_df, neg_df], ignore_index=True)
            combined_path = os.path.join(args.output_dir, "bcell_combined.csv")
            combined.to_csv(combined_path, index=False)
            process_iedb_csv(combined_path, args.output_dir)
    elif args.input:
        process_iedb_csv(args.input, args.output_dir)
    elif args.positive and args.negative:
        pos_seqs, _ = fasta_to_sequences(args.positive)
        neg_seqs, _ = fasta_to_sequences(args.negative)
        pos_seqs = [s for s in (clean_sequence(s) for s in pos_seqs) if s]
        neg_seqs = [s for s in (clean_sequence(s) for s in neg_seqs) if s]
        all_seqs = pos_seqs + neg_seqs
        all_labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)
        np.random.seed(RANDOM_SEED)
        idx = np.random.permutation(len(all_seqs))
        test_n = int(len(all_seqs) * TEST_RATIO)
        train_df = pd.DataFrame({
            "sequence": [all_seqs[i] for i in idx[test_n:]],
            "label": [all_labels[i] for i in idx[test_n:]],
        })
        test_df = pd.DataFrame({
            "sequence": [all_seqs[i] for i in idx[:test_n]],
            "label": [all_labels[i] for i in idx[:test_n]],
        })
        os.makedirs(args.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
        print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
    else:
        parser.print_help()
        print("\n请指定 --download, --input 或 --positive/--negative 参数")


if __name__ == "__main__":
    main()
