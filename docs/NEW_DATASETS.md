# Training NicheFlow on New Datasets

To train **NicheFlow** on a new dataset, you need to follow three main steps: dataset preprocessing, classifier training, and NicheFlow training.

---

## 1. Dataset Preprocessing

1. Load your dataset as an **Annotated Data object (`AnnData`)**.
2. Perform your preprocessing with **Scanpy**:
   - Apply **PCA** before using our preprocessor (**required**).
3. Specify the following parameters for the preprocessor:
   - **Timepoint column** - column specifying the temporal information.
   - **Cell type column** - column specifying cell types.
   - **Temporally ordered timepoints** - ensure your timepoints are in the correct order.  
     - *Note:* Sometimes timepoints are not trivially sortable (e.g., axolotl dataset). Refer to the example notebook.
   - **Coordinate standardization** - we recommend standardizing over min-max scaling.
   - **Radius** - defines the size of microenvironments.
   - **`dx` and `dy`** - discretization steps for generating the grid of **test microenvironments**.
   - **Device** - set to `cuda` (recommended) to speed up preprocessing.
   - **Chunk size** - controls memory usage. Reduce it if you encounter errors.
4. Preprocess the dataset and save it in the [`data/`](../data/) folder

**Recommendation:**  
See [`download_and_preprocess.ipynb`](../notebooks/download_and_preprocess.ipynb) for a complete example of how we downloaded and preprocessed the datasets used in our paper.

---

## 2. Classifier Training

Before training NicheFlow, you need a trained classifier for your dataset. This requires creating a **new datamodule configuration** and **experiment configuration**.

### Datamodule Config
- Define `ct_{dataset_name}` in [`configs/data/`](../configs/data/).
- Specify:
  - Number of PCA components.
  - Number of cell type classes.

### Experiment Config
- Define `experiment/classifier/{dataset_name}` in [`configs/experiment/classifier`](../configs/experiment/classifier/).
- Set:
  - The correct data override (pointing to your new datamodule).
  - A unique WandB run name.

Then, train the classifier:
```bash
python nicheflow/train.py experiment=classifier/{dataset_name}
```

## 3. NicheFlow Training

Once the classifier is trained, you can train NicheFlow.

### Datamodule Config

- Define `nicheflow_{dataset_name}` in [`configs/data/`](../configs/data/).
- Specify:
  - Dataset filepath.
  - Number of PCA components.
  - Number of cell type classes.
  - Number of timepoints/slices.
  - Path to the **trained classifier checkpoint**.

### Experiment Config

- Define `experiment/nicheflow/{cfm | gvfm | glvfm}/{dataset_name}` in [`configs/experiment/nicheflow`](../configs/experiment/nicheflow/).
- Set:
  - The correct data override.
  - A unique WandB run name.

Then, train NicheFlow:
```bash
python nicheflow/train.py experiment=nicheflow/{cfm|gvfm|glvfm}/{dataset_name}
```

---
---  

# 在新数据集上训练 NicheFlow

若要在新数据集上训练 **NicheFlow**，您需要遵循三个主要步骤：数据集预处理、分类器训练以及 NicheFlow 训练。

---

## 1. 数据集预处理

1. 将您的数据集加载为 **注释数据对象 (`AnnData`)**。
2. 使用 **Scanpy** 进行预处理：
   - 在使用我们的预处理器之前应用 **PCA**（**必须**）。
3. 为预处理器指定以下参数：
   - **时间点列 (Timepoint column)** - 指定时间信息的列。
   - **细胞类型列 (Cell type column)** - 指定细胞类型的列。
   - **按时间顺序排列的时间点 (Temporally ordered timepoints)** - 确保您的时间点顺序正确。  
     - *注意：* 有些时候时间点无法进行简单排序（例如，axolotl 数据集）。请参考示例 notebook。
   - **坐标标准化 (Coordinate standardization)** - 相比于最小最大缩放 (min-max scaling)，我们更推荐使用标准化 (standardization)。
   - **半径 (Radius)** - 定义微环境的大小。
   - **`dx` 和 `dy`** - 用于生成 **测试微环境 (test microenvironments)** 网格的离散化步长。
   - **设备 (Device)** - 设置为 `cuda`（推荐）以加速预处理。
   - **块大小 (Chunk size)** - 控制内存使用情况。如果遇到错误，请减小该值。
4. 预处理数据集并将其保存在 [`data/`](../data/) 文件夹中。

**推荐操作：**  
请参阅 [`download_and_preprocess.ipynb`](../notebooks/download_and_preprocess.ipynb)，查看我们如何下载和预处理论文中所用数据集的完整示例。

---

## 2. 分类器训练

在训练 NicheFlow 之前，您需要为您的数据集准备一个经过训练的分类器。这需要创建一个 **新的数据模块配置 (datamodule configuration)** 和 **实验配置 (experiment configuration)**。

### 数据模块配置 (Datamodule Config)
- 在 [`configs/data/`](../configs/data/) 中定义 `ct_{dataset_name}`。
- 指定以下内容：
  - PCA 主成分数量。
  - 细胞类型类别的数量。

### 实验配置 (Experiment Config)
- 在[`configs/experiment/classifier`](../configs/experiment/classifier/) 中定义 `experiment/classifier/{dataset_name}`。
- 设置以下内容：
  - 正确的数据覆盖项 (data override)（指向您新的数据模块）。
  - 唯一的 WandB 运行名称 (run name)。

然后，训练分类器：
```bash
python nicheflow/train.py experiment=classifier/{dataset_name}
```

## 3. NicheFlow 训练

分类器训练完成后，您就可以开始训练 NicheFlow 了。

### 数据模块配置 (Datamodule Config)

- 在 [`configs/data/`](../configs/data/) 中定义 `nicheflow_{dataset_name}`。
- 指定以下内容：
  - 数据集文件路径。
  - PCA 主成分数量。
  - 细胞类型类别的数量。
  - 时间点/切片的数量。
  - **已训练好的分类器检查点 (checkpoint)** 的路径。

### 实验配置 (Experiment Config)

- 在 [`configs/experiment/nicheflow`](../configs/experiment/nicheflow/) 中定义 `experiment/nicheflow/{cfm | gvfm | glvfm}/{dataset_name}`。
- 设置以下内容：
  - 正确的数据覆盖项 (data override)。
  - 唯一的 WandB 运行名称 (run name)。

然后，训练 NicheFlow：
```bash
python nicheflow/train.py experiment=nicheflow/{cfm|gvfm|glvfm}/{dataset_name}
```