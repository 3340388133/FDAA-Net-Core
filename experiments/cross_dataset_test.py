#!/usr/bin/env python3
"""
跨数据集测试脚本 - 支持Midjourney和SVD数据集
基于FDAA-Net模型进行跨域泛化测试
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import argparse
from PIL import Image
import torchvision.transforms as transforms

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import AIGCDetector, AIGCDetectorLite
from utils.metrics import compute_metrics
from utils.logger import setup_logger

class CrossDatasetTester:
    """
    跨数据集测试器
    支持多种数据集格式和来源
    """

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.logger = setup_logger('cross_dataset_test')

    def load_model(self, model_path):
        """加载预训练的AIGC检测模型"""
        # 使用轻量版模型进行测试
        model = AIGCDetectorLite(
            num_classes=2,
            img_size=224,
            embed_dim=768,
            num_prototypes=4
        )

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✅ 成功加载模型: {model_path}")
            except Exception as e:
                print(f"⚠️ 模型加载失败，使用随机初始化: {e}")
        else:
            print(f"⚠️ 模型文件不存在，使用随机初始化: {model_path}")

        model.to(self.device)
        model.eval()
        return model

    def get_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_dataset_from_directory(self, data_dir, max_samples=None):
        """
        从目录加载数据集
        支持以下结构:
        data_dir/
        ├── real/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── fake/
            ├── image1.jpg
            └── image2.jpg
        """
        real_dir = Path(data_dir) / 'real'
        fake_dir = Path(data_dir) / 'fake'

        images = []
        labels = []

        # 加载真实图像
        if real_dir.exists():
            real_files = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
            if max_samples:
                real_files = real_files[:max_samples//2]

            for img_path in real_files:
                images.append(str(img_path))
                labels.append(0)  # 0 = real

        # 加载生成图像
        if fake_dir.exists():
            fake_files = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
            if max_samples:
                fake_files = fake_files[:max_samples//2]

            for img_path in fake_files:
                images.append(str(img_path))
                labels.append(1)  # 1 = fake

        print(f"📊 加载数据集: {len(images)} 张图像 (真实: {labels.count(0)}, 生成: {labels.count(1)})")
        return images, labels

    def create_sample_dataset(self, dataset_name, num_samples=1000):
        """
        创建示例数据集用于测试
        """
        sample_dir = Path(f"./sample_data/{dataset_name}")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 创建真实和生成图像目录
        (sample_dir / 'real').mkdir(exist_ok=True)
        (sample_dir / 'fake').mkdir(exist_ok=True)

        print(f"📁 创建示例数据集目录: {sample_dir}")
        print(f"请将 {dataset_name} 数据集的图像放入以下目录:")
        print(f"  真实图像: {sample_dir / 'real'}")
        print(f"  生成图像: {sample_dir / 'fake'}")

        return str(sample_dir)

    def download_modelscope_dataset(self, dataset_name):
        """
        尝试从ModelScope下载数据集
        如果失败，提供手动下载指导
        """
        try:
            # 尝试使用huggingface datasets作为替代
            from datasets import load_dataset
            print(f"🔄 尝试从HuggingFace下载数据集: {dataset_name}")

            # 常见的数据集映射
            hf_dataset_map = {
                'OmniData/midjourney-v6-520k-raw': 'midjourney-v6-520k',
                'stable-video-diffusion': 'stabilityai/stable-video-diffusion'
            }

            hf_name = hf_dataset_map.get(dataset_name, dataset_name)
            dataset = load_dataset(hf_name, split='train[:1000]')  # 只下载1000个样本用于测试

            print(f"✅ 成功下载数据集: {len(dataset)} 个样本")
            return self.convert_hf_dataset(dataset, dataset_name)

        except Exception as e:
            print(f"❌ 自动下载失败: {e}")
            print("\n📋 手动下载指导:")
            print(f"1. 访问 https://modelscope.cn/datasets/{dataset_name}")
            print("2. 下载数据集到本地")
            print("3. 将数据整理为以下格式:")
            print("   dataset_name/")
            print("   ├── real/")
            print("   └── fake/")

            return self.create_sample_dataset(dataset_name.split('/')[-1])

    def convert_hf_dataset(self, dataset, dataset_name):
        """
        转换HuggingFace数据集格式
        """
        output_dir = Path(f"./downloaded_data/{dataset_name.split('/')[-1]}")
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / 'real').mkdir(exist_ok=True)
        (output_dir / 'fake').mkdir(exist_ok=True)

        print(f"🔄 转换数据集格式到: {output_dir}")

        for i, sample in enumerate(tqdm(dataset)):
            if i >= 1000:  # 限制样本数量
                break

            # 根据数据集结构调整
            if 'image' in sample and 'label' in sample:
                image = sample['image']
                label = sample['label']

                subdir = 'real' if label == 0 else 'fake'
                image_path = output_dir / subdir / f"image_{i:06d}.jpg"

                if hasattr(image, 'save'):
                    image.save(image_path)

        return str(output_dir)

    def predict_batch(self, image_paths, batch_size=32):
        """
        批量预测图像
        """
        predictions = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="预测中"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"⚠️ 加载图像失败: {img_path} - {e}")
                    continue

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('cls'))
                    else:
                        logits = outputs

                    probs = torch.softmax(logits, dim=1)[:, 1]  # 获取fake类别概率
                    predictions.extend(probs.cpu().numpy())

        return np.array(predictions)

    def evaluate_dataset(self, dataset_name, data_path=None, max_samples=None):
        """
        评估单个数据集
        """
        print(f"\n🔍 开始评估数据集: {dataset_name}")
        print("=" * 50)

        # 获取数据路径
        if data_path is None:
            if os.path.exists(f"./sample_data/{dataset_name}"):
                data_path = f"./sample_data/{dataset_name}"
            else:
                data_path = self.download_modelscope_dataset(dataset_name)

        # 加载数据集
        if not os.path.exists(data_path):
            print(f"❌ 数据集路径不存在: {data_path}")
            return None

        image_paths, labels = self.load_dataset_from_directory(data_path, max_samples)

        if len(image_paths) == 0:
            print(f"❌ 未找到图像文件")
            return None

        # 进行预测
        predictions = self.predict_batch(image_paths)

        if len(predictions) != len(labels):
            # 调整标签长度以匹配预测结果
            labels = labels[:len(predictions)]

        # 计算指标
        results = compute_metrics(predictions, labels)

        # 添加num_samples到结果中
        results['num_samples'] = len(labels)

        # 打印结果
        self.print_results(dataset_name, results)

        return results


    def print_results(self, dataset_name, results):
        """
        打印评估结果
        """
        print(f"\n📊 {dataset_name} 评估结果:")
        print("-" * 30)
        print(f"AUC:      {results['auc']:.4f}")
        print(f"AP:       {results['ap']:.4f}")
        print(f"准确率:    {results['accuracy']:.4f}")
        print(f"精确率:    {results['precision']:.4f}")
        print(f"召回率:    {results['recall']:.4f}")
        print(f"F1分数:   {results['f1']:.4f}")
        print(f"样本数:    {results['num_samples']}")

    def run_cross_dataset_evaluation(self, datasets, model_path=None, output_file=None):
        """
        运行跨数据集评估
        """
        print("🚀 开始跨数据集评估")
        print("=" * 60)

        all_results = {}

        for dataset_info in datasets:
            if isinstance(dataset_info, str):
                dataset_name = dataset_info
                data_path = None
            else:
                dataset_name = dataset_info['name']
                data_path = dataset_info.get('path')

            results = self.evaluate_dataset(dataset_name, data_path)
            if results:
                all_results[dataset_name] = results

        # 计算平均性能
        if all_results:
            avg_results = self.calculate_average_results(all_results)
            self.print_summary(all_results, avg_results)

            # 保存结果
            if output_file:
                self.save_results(all_results, avg_results, output_file)

        return all_results

    def calculate_average_results(self, all_results):
        """
        计算平均结果
        """
        metrics = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
        avg_results = {}

        for metric in metrics:
            values = [results[metric] for results in all_results.values()]
            avg_results[metric] = np.mean(values)

        return avg_results

    def print_summary(self, all_results, avg_results):
        """
        打印总结
        """
        print("\n" + "=" * 60)
        print("📈 跨数据集评估总结")
        print("=" * 60)

        # 打印每个数据集的结果表格
        print(f"{'数据集':<20} {'AUC':<8} {'AP':<8} {'准确率':<8} {'F1':<8}")
        print("-" * 60)

        for dataset_name, results in all_results.items():
            print(f"{dataset_name:<20} {results['auc']:<8.4f} {results['ap']:<8.4f} {results['accuracy']:<8.4f} {results['f1']:<8.4f}")

        print("-" * 60)
        print(f"{'平均':<20} {avg_results['auc']:<8.4f} {avg_results['ap']:<8.4f} {avg_results['accuracy']:<8.4f} {avg_results['f1']:<8.4f}")

        # 找出最佳和最差性能
        best_auc = max(all_results.items(), key=lambda x: x[1]['auc'])
        worst_auc = min(all_results.items(), key=lambda x: x[1]['auc'])

        print(f"\n🏆 最佳AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
        print(f"📉 最差AUC: {worst_auc[0]} ({worst_auc[1]['auc']:.4f})")
        print(f"📊 AUC范围: {worst_auc[1]['auc']:.4f} - {best_auc[1]['auc']:.4f}")

    def save_results(self, all_results, avg_results, output_file):
        """
        保存结果到JSON文件
        """
        output_data = {
            'individual_results': all_results,
            'average_results': avg_results,
            'summary': {
                'num_datasets': len(all_results),
                'total_samples': sum(r['num_samples'] for r in all_results.values())
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='跨数据集测试脚本')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                       help='模型路径')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['OmniData/midjourney-v6-520k-raw', 'stable-video-diffusion'],
                       help='要测试的数据集列表')
    parser.add_argument('--output', type=str, default='./outputs/cross_dataset_results.json',
                       help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='每个数据集的最大样本数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 初始化测试器
    tester = CrossDatasetTester(args.model, args.device)

    # 准备数据集列表
    datasets = []
    for dataset_name in args.datasets:
        datasets.append({
            'name': dataset_name,
            'max_samples': args.max_samples
        })

    # 运行跨数据集评估
    results = tester.run_cross_dataset_evaluation(datasets, output_file=args.output)

    print("\n✅ 跨数据集测试完成！")

if __name__ == '__main__':
    main()
