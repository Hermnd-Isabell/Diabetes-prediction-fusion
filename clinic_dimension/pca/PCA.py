import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MedicalDataPCA:
    def __init__(self, data, output_dir='pca_results'):
        """
        初始化PCA分析类
        
        Parameters:
        data: pandas DataFrame, 包含所有特征的原始数据
        output_dir: str, 结果保存目录
        """
        self.data = data.copy()
        self.numeric_features = None
        self.categorical_features = None
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_results = None
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def preprocess_data(self):
        """
        数据预处理：处理缺失值、编码分类变量、标准化
        """
        print("开始数据预处理...")
        
        # 1. 处理特殊值（如骨密度中的'N-未知'）
        # 将各种可能的缺失值表示转换为NaN
        missing_indicators = ['N', '未知', 'Unknown', 'unknown', 'NULL', 'null', '', 'NaN']
        for indicator in missing_indicators:
            self.data = self.data.replace(indicator, np.nan)
        
        # 2. 分离数值型和分类型特征
        feature_columns = self.data.columns.tolist()
        
        # 识别数值型特征（基于数据类型和内容）
        self.numeric_features = []
        self.categorical_features = []
        
        for col in feature_columns:
            # 尝试转换为数值型，如果成功则认为是数值特征
            try:
                pd.to_numeric(self.data[col], errors='raise')
                self.numeric_features.append(col)
            except (ValueError, TypeError):
                self.categorical_features.append(col)
        
        print(f"数值型特征数量: {len(self.numeric_features)}")
        print(f"分类型特征数量: {len(self.categorical_features)}")
        
        # 3. 处理数值型特征的缺失值
        if self.numeric_features:
            # 检查每个数值特征的缺失值比例
            numeric_missing = self.data[self.numeric_features].isnull().sum()
            print("\n数值特征缺失值统计:")
            for col, missing_count in numeric_missing.items():
                if missing_count > 0:
                    missing_ratio = missing_count / len(self.data)
                    print(f"  {col}: {missing_count}个缺失值 ({missing_ratio:.2%})")
            
            numeric_imputer = SimpleImputer(strategy='median')
            self.data[self.numeric_features] = numeric_imputer.fit_transform(self.data[self.numeric_features])
        
        # 4. 处理分类型特征的缺失值和编码
        categorical_data_encoded = []
        categorical_feature_names = []
        
        if self.categorical_features:
            print("\n分类型特征处理:")
            for col in self.categorical_features:
                # 处理缺失值
                if self.data[col].isnull().any():
                    missing_count = self.data[col].isnull().sum()
                    print(f"  {col}: {missing_count}个缺失值，使用众数填充")
                    self.data[col].fillna(self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown', inplace=True)
                
                # 标签编码
                le = LabelEncoder()
                try:
                    encoded_col = le.fit_transform(self.data[col].astype(str))
                    categorical_data_encoded.append(encoded_col)
                    categorical_feature_names.append(f"{col}_encoded")
                    print(f"  {col}: 编码完成，共有 {len(le.classes_)} 个类别")
                except Exception as e:
                    print(f"  {col}: 编码失败 - {e}")
                    # 如果编码失败，使用one-hot编码的简化版
                    unique_vals = self.data[col].astype(str).unique()
                    if len(unique_vals) <= 10:  # 如果类别不多，手动编码
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        encoded_col = self.data[col].map(mapping)
                        categorical_data_encoded.append(encoded_col)
                        categorical_feature_names.append(f"{col}_encoded")
                    else:
                        # 如果类别太多，暂时忽略这个特征
                        print(f"  {col}: 类别过多({len(unique_vals)}个)，暂时跳过")
        
        # 5. 标准化数值型特征
        if self.numeric_features:
            scaled_numeric_data = self.scaler.fit_transform(self.data[self.numeric_features])
            print(f"数值特征标准化完成")
        else:
            scaled_numeric_data = np.array([]).reshape(len(self.data), 0)
        
        # 6. 合并所有特征
        if categorical_data_encoded:
            categorical_array = np.column_stack(categorical_data_encoded)
            if scaled_numeric_data.size > 0:
                self.processed_data = np.column_stack([scaled_numeric_data, categorical_array])
                # 更新特征名称
                self.all_feature_names = self.numeric_features + categorical_feature_names
            else:
                self.processed_data = categorical_array
                self.all_feature_names = categorical_feature_names
        else:
            self.processed_data = scaled_numeric_data
            self.all_feature_names = self.numeric_features
        
        print(f"预处理后数据形状: {self.processed_data.shape}")
        return self.processed_data
    
    def perform_pca(self, n_components=None, variance_threshold=0.95):
        """
        执行PCA分析
        
        Parameters:
        n_components: int, 指定主成分数量
        variance_threshold: float, 累积方差解释率阈值
        """
        print("\n开始PCA分析...")
        
        if n_components is None:
            # 自动选择主成分数量，基于累积方差解释率
            self.pca = PCA()
            pca_fit = self.pca.fit(self.processed_data)
            
            # 计算累积方差解释率
            cumulative_variance = np.cumsum(pca_fit.explained_variance_ratio_)
            
            # 找到达到方差阈值所需的最小主成分数
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            print(f"选择 {n_components} 个主成分，可解释 {cumulative_variance[n_components-1]:.3f} 的方差")
        
        # 使用确定的主成分数量重新拟合PCA
        self.pca = PCA(n_components=n_components)
        self.pca_results = self.pca.fit_transform(self.processed_data)
        
        print(f"主成分分析完成，原始特征数: {self.processed_data.shape[1]}, 降维后: {n_components}")
        
        return self.pca_results
    
    def plot_pca_results(self):
        """
        绘制PCA结果可视化图形
        """
        if self.pca is None:
            print("请先执行PCA分析")
            return
        
        # 创建可视化图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 方差解释率图
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, color='skyblue')
        axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
        axes[0, 0].set_xlabel('主成分数量')
        axes[0, 0].set_ylabel('方差解释率')
        axes[0, 0].set_title('主成分方差解释率')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['累积方差', '单个主成分方差'])
        
        # 2. 前两个主成分的散点图
        if self.pca_results.shape[1] >= 2:
            axes[0, 1].scatter(self.pca_results[:, 0], self.pca_results[:, 1], alpha=0.6, color='lightcoral')
            axes[0, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.3f})')
            axes[0, 1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.3f})')
            axes[0, 1].set_title('前两个主成分散点图')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 特征在主成分上的载荷热力图（前10个主成分）
        if self.pca_results.shape[1] >= 3:
            # 选择前10个主成分的载荷
            n_components_to_plot = min(10, self.pca_results.shape[1])
            loadings = self.pca.components_[:n_components_to_plot].T
            
            # 选择载荷绝对值最大的前20个特征
            max_loadings_idx = np.argsort(np.max(np.abs(loadings), axis=1))[-20:]
            important_loadings = loadings[max_loadings_idx]
            important_features = [self.all_feature_names[i] for i in max_loadings_idx]
            
            im = axes[1, 0].imshow(important_loadings, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_xticks(range(n_components_to_plot))
            axes[1, 0].set_xticklabels([f'PC{i+1}' for i in range(n_components_to_plot)])
            axes[1, 0].set_yticks(range(len(important_features)))
            axes[1, 0].set_yticklabels(important_features, fontsize=8)
            axes[1, 0].set_title('特征在主成分上的载荷（前20个重要特征）')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 各主成分的累积方差解释率
        axes[1, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%方差阈值')
        axes[1, 1].set_xlabel('主成分数量')
        axes[1, 1].set_ylabel('累积方差解释率')
        axes[1, 1].set_title('累积方差解释率')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.output_dir, f'pca_plots_{timestamp}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"PCA可视化图已保存: {plot_filename}")
        
        plt.show()
    
    def get_feature_importance(self, top_n=10):
        """
        获取对主成分贡献最大的特征
        
        Parameters:
        top_n: int, 返回前N个重要特征
        """
        if self.pca is None:
            print("请先执行PCA分析")
            return None
        
        # 计算特征重要性（基于在主成分上的绝对载荷）
        feature_importance = {}
        for i, feature in enumerate(self.all_feature_names):
            # 使用在第一主成分上的绝对载荷作为重要性指标
            importance = np.abs(self.pca.components_[0, i])
            feature_importance[feature] = importance
        
        # 按重要性排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n前{top_n}个最重要的特征（基于PC1的载荷）:")
        print("-" * 50)
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            print(f"{i+1:2d}. {feature:25s} : {importance:.4f}")
        
        # 保存特征重要性到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
        importance_filename = os.path.join(self.output_dir, f'feature_importance_{timestamp}.csv')
        importance_df.to_csv(importance_filename, index=False, encoding='utf-8-sig')
        print(f"特征重要性已保存: {importance_filename}")
        
        return sorted_features[:top_n]
    
    def get_transformed_data(self):
        """
        返回降维后的数据
        """
        if self.pca_results is None:
            print("请先执行PCA分析")
            return None
        
        # 创建包含主成分的DataFrame
        pca_columns = [f'PC{i+1}' for i in range(self.pca_results.shape[1])]
        pca_df = pd.DataFrame(self.pca_results, columns=pca_columns)
        
        return pca_df
    
    def save_pca_results(self, filename=None, include_original_index=True):
        """
        保存PCA降维后的数据集
        
        Parameters:
        filename: str, 保存的文件名
        include_original_index: bool, 是否包含原始数据的索引
        """
        if self.pca_results is None:
            print("请先执行PCA分析")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'pca_reduced_data_{timestamp}.csv'
        
        # 获取完整文件路径
        filepath = os.path.join(self.output_dir, filename)
        
        # 获取降维后的数据
        pca_df = self.get_transformed_data()
        
        # 如果需要，添加原始索引
        if include_original_index and hasattr(self.data, 'index'):
            pca_df.index = self.data.index
        
        # 保存数据
        pca_df.to_csv(filepath, index=include_original_index, encoding='utf-8-sig')
        print(f"降维后的数据集已保存: {filepath}")
        
        # 同时保存PCA模型信息
        self.save_pca_model_info()
        
        return filepath
    
    def save_pca_model_info(self):
        """
        保存PCA模型的详细信息
        """
        if self.pca is None:
            print("请先执行PCA分析")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主成分载荷矩阵
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            index=self.all_feature_names,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        loadings_filename = os.path.join(self.output_dir, f'pca_loadings_{timestamp}.csv')
        loadings_df.to_csv(loadings_filename, encoding='utf-8-sig')
        print(f"主成分载荷矩阵已保存: {loadings_filename}")
        
        # 保存方差解释率
        variance_df = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(self.pca.n_components_)],
            '方差解释率': self.pca.explained_variance_ratio_,
            '累积方差解释率': np.cumsum(self.pca.explained_variance_ratio_)
        })
        variance_filename = os.path.join(self.output_dir, f'pca_variance_{timestamp}.csv')
        variance_df.to_csv(variance_filename, index=False, encoding='utf-8-sig')
        print(f"方差解释率已保存: {variance_filename}")
        
        # 保存预处理信息
        preprocess_info = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'all_features': self.all_feature_names,
            'original_shape': self.data.shape,
            'reduced_shape': self.pca_results.shape,
            'variance_explained': np.sum(self.pca.explained_variance_ratio_)
        }
        
        info_filename = os.path.join(self.output_dir, f'pca_preprocess_info_{timestamp}.txt')
        with open(info_filename, 'w', encoding='utf-8') as f:
            for key, value in preprocess_info.items():
                f.write(f"{key}: {value}\n")
        print(f"预处理信息已保存: {info_filename}")

# 主程序
def main():
    # 读取Excel文件
    file_path = r'C:\Users\Yiqiu\OneDrive\Desktop\大创\clinic.xlsx'
    
    try:
        print(f"正在读取Excel文件: {file_path}")
        df = pd.read_excel(file_path)
        print(f"成功读取数据，形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
        
        # 显示数据前几行
        print("\n数据前5行:")
        print(df.head())
        
        # 显示数据基本信息
        print("\n数据基本信息:")
        print(df.info())
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 使用PCA类进行分析
    output_directory = 'pca_medical_results'
    pca_analyzer = MedicalDataPCA(df, output_dir=output_directory)
    
    # 1. 数据预处理
    print("\n" + "="*50)
    print("开始数据预处理")
    print("="*50)
    processed_data = pca_analyzer.preprocess_data()
    
    # 2. 执行PCA（自动选择主成分数量，保留95%方差）
    print("\n" + "="*50)
    print("开始PCA分析")
    print("="*50)
    pca_result = pca_analyzer.perform_pca(variance_threshold=0.95)
    
    # 3. 可视化结果
    print("\n" + "="*50)
    print("生成可视化结果")
    print("="*50)
    pca_analyzer.plot_pca_results()
    
    # 4. 获取重要特征
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)
    important_features = pca_analyzer.get_feature_importance(top_n=15)
    
    # 5. 获取降维后的数据
    print("\n" + "="*50)
    print("降维数据概览")
    print("="*50)
    transformed_data = pca_analyzer.get_transformed_data()
    print(f"降维后的数据形状: {transformed_data.shape}")
    print("\n前5行降维数据:")
    print(transformed_data.head())
    
    # 6. 保存降维后的数据集
    print("\n" + "="*50)
    print("保存结果")
    print("="*50)
    saved_file = pca_analyzer.save_pca_results(
        filename='medical_data_pca_reduced.csv',
        include_original_index=True
    )
    
    print(f"\n所有结果已保存到目录: {pca_analyzer.output_dir}")
    print("分析完成!")

if __name__ == "__main__":
    main()