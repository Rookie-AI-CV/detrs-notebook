import torch  
import torch.nn as nn  
import matplotlib.pyplot as plt  
import numpy as np  
import math  

class PositionEmbeddingSine(nn.Module):  
    """  
    正弦位置编码模块  
    实现2D正弦位置编码，为图像的每个像素位置生成唯一的位置表示  
    """  
    def __init__(self, hidden_dim=256, temperature=10000, scale=None):  
        super().__init__()  
        self.hidden_dim = hidden_dim              # 位置编码总维度  
        self.num_pos_feats = hidden_dim // 2      # 一半给x维度，一半给y维度  
        self.temperature = temperature            # 温度参数，控制编码频率  
        self.scale = 2 * math.pi if scale is None else scale  # 归一化尺度，默认2π  
        
    def forward(self, x):  
        """  
        前向传播：为输入特征图生成2D位置编码  
        
        Args:  
            x: 输入特征图 [batch_size, hidden_dim, H, W]  
            
        Returns:   
            pos: 位置编码 [batch_size, hidden_dim, H, W]  
        """  
        bs, _, h, w = x.shape  
        device = x.device  
        # 步骤1: 生成位置索引坐标  
        y_embed = torch.arange(h, dtype=torch.float32, device=device) # y_embed: [0, 1, 2, ..., h-1] 表示每一行的索引  
        x_embed = torch.arange(w, dtype=torch.float32, device=device)  # x_embed: [0, 1, 2, ..., w-1] 表示每一列的索引      
        # 步骤2: 归一化位置坐标到[0, 2π]范围  
        y_embed = y_embed / h * self.scale  # 将像素坐标从[0, h-1]映射到[0, 2π]  
        x_embed = x_embed / w * self.scale  # 将像素坐标从[0, w-1]映射到[0, 2π]  
        # 步骤3: 生成频率序列  
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device) # dim_t: [0, 1, 2, ..., num_pos_feats-1]  
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 计算频率: temperature^(2i/d) 其中i为维度索引  
        # 步骤4: 计算位置编码  
        pos_x = x_embed[:, None] / dim_t  # pos_x: [W, num_pos_feats] 每列的x维度位置编码   
        pos_y = y_embed[:, None] / dim_t  # pos_y: [H, num_pos_feats] 每行的y维度位置编码  
        # 步骤5: 应用正弦余弦函数并交替排列  
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)  # 偶数索引用sin，奇数索引用cos，然后展平  
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)  
        # 步骤6: 扩展到2D网格  
        pos_y = pos_y[:, None, :].expand(-1, w, -1)  # pos_y: [H, W, num_pos_feats] 将每行的y编码复制到所有列  
        pos_x = pos_x[None, :, :].expand(h, -1, -1)  # pos_x: [H, W, num_pos_feats] 将每列的x编码复制到所有行  
        # 步骤7: 拼接x和y维度的位置编码  
        pos = torch.cat([pos_y, pos_x], dim=-1)  # pos: [H, W, hidden_dim] 完整的2D位置编码  
        # 步骤8: 调整维度顺序并扩展batch维度  
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(bs, -1, -1, -1)  # 从[H, W, hidden_dim]转为[hidden_dim, H, W]，然后扩展到batch_size  
        
        return pos  


def visualize_sine_position_encoding():  
    """可视化正弦位置编码的各个步骤"""  
    
    # 创建位置编码器  
    pos_encoder = PositionEmbeddingSine(hidden_dim=128, temperature=10000)  
    
    # 创建测试输入  
    h, w = 20, 20  
    x = torch.randn(1, 128, h, w)  
    
    # 生成位置编码  
    pos_encoding = pos_encoder(x)[0]  # [128, 20, 20]  
    
    # 创建大图  
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
    fig.suptitle('DETR Sine Position Encoding Visualization', fontsize=16, fontweight='bold')  
    
    # 1. 原始位置坐标  
    y_coords = torch.arange(h, dtype=torch.float32)  
    x_coords = torch.arange(w, dtype=torch.float32)  
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')  
    
    im1 = axes[0, 0].imshow(Y.numpy(), cmap='viridis', aspect='equal')  
    axes[0, 0].set_title('Step 1: Y Coordinates')  
    axes[0, 0].set_xlabel('Width')  
    axes[0, 0].set_ylabel('Height')  
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.7)  
    
    im2 = axes[1, 0].imshow(X.numpy(), cmap='viridis', aspect='equal')  
    axes[1, 0].set_title('Step 1: X Coordinates')  
    axes[1, 0].set_xlabel('Width')  
    axes[1, 0].set_ylabel('Height')  
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.7)  
    
    # 2. 归一化后的坐标  
    y_norm = y_coords / h * 2 * math.pi  
    x_norm = x_coords / w * 2 * math.pi  
    Y_norm, X_norm = torch.meshgrid(y_norm, x_norm, indexing='ij')  
    
    im3 = axes[0, 1].imshow(Y_norm.numpy(), cmap='viridis', aspect='equal')  
    axes[0, 1].set_title('Step 2: Normalized Y [0, 2π]')  
    axes[0, 1].set_xlabel('Width')  
    axes[0, 1].set_ylabel('Height')  
    plt.colorbar(im3, ax=axes[0, 1], shrink=0.7)  
    
    im4 = axes[1, 1].imshow(X_norm.numpy(), cmap='viridis', aspect='equal')  
    axes[1, 1].set_title('Step 2: Normalized X [0, 2π]')  
    axes[1, 1].set_xlabel('Width')  
    axes[1, 1].set_ylabel('Height')  
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.7)  
    
    # 3. 频率响应  
    num_pos_feats = 64  
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)  
    frequencies = 10000 ** (2 * (dim_t // 2) / num_pos_feats)  
    
    axes[0, 2].semilogy(frequencies.numpy())  
    axes[0, 2].set_title('Step 3: Frequency Sequence')  
    axes[0, 2].set_xlabel('Dimension Index')  
    axes[0, 2].set_ylabel('Frequency (log scale)')  
    axes[0, 2].grid(True, alpha=0.3)  
    
    # 选择不同频率的1D编码示例  
    mid_pos = h // 2  
    for i, freq_idx in enumerate([0, 16, 32, 48]):  
        if freq_idx < pos_encoding.shape[0]:  
            y_slice = pos_encoding[freq_idx, :, mid_pos].numpy()  
            axes[1, 2].plot(y_slice, label=f'Dim {freq_idx}', linewidth=2)  
    axes[1, 2].set_title('Step 5: Sin/Cos Encoding (Y direction)')  
    axes[1, 2].set_xlabel('Y Position')  
    axes[1, 2].set_ylabel('Encoding Value')  
    axes[1, 2].legend()  
    axes[1, 2].grid(True, alpha=0.3)  
    
    # 4. 最终位置编码的不同通道  
    channels_to_show = [0, 32, 64, 96]  
    for i, ch in enumerate(channels_to_show):  
        if i < 2:  # 前两个显示在第一行  
            ax = axes[0, 3]  
            if i == 0:  
                im = ax.imshow(pos_encoding[ch].numpy(), cmap='RdBu_r', aspect='equal')  
                ax.set_title(f'Final Encoding - Channel {ch}')  
                plt.colorbar(im, ax=ax, shrink=0.7)  
        else:  # 后两个显示在第二行  
            ax = axes[1, 3]  
            if i == 2:  
                im = ax.imshow(pos_encoding[ch].numpy(), cmap='RdBu_r', aspect='equal')  
                ax.set_title(f'Final Encoding - Channel {ch}')  
                plt.colorbar(im, ax=ax, shrink=0.7)  
    
    plt.tight_layout()  
    plt.show()  
    
    # 显示详细信息  
    print("=" * 60)  
    print("Sine Position Encoding Analysis")  
    print("=" * 60)  
    print(f"Input shape: {x.shape}")  
    print(f"Output shape: {pos_encoding.shape}")  
    print(f"Hidden dim: {pos_encoder.hidden_dim}")  
    print(f"Num pos feats: {pos_encoder.num_pos_feats}")  
    print(f"Temperature: {pos_encoder.temperature}")  
    print(f"Scale: {pos_encoder.scale:.4f}")  
    print(f"Encoding range: [{pos_encoding.min():.4f}, {pos_encoding.max():.4f}]")  


def visualize_frequency_analysis():  
    """可视化不同频率的编码效果"""  
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
    fig.suptitle('Frequency Analysis of Sine Position Encoding', fontsize=16, fontweight='bold')  
    
    h, w = 30, 30  
    x = torch.randn(1, 64, h, w)  
    
    # 不同温度参数  
    temperatures = [100, 1000, 10000]  
    
    for i, temp in enumerate(temperatures):  
        pos_encoder = PositionEmbeddingSine(hidden_dim=64, temperature=temp)  
        encoding = pos_encoder(x)[0]  
        
        # 选择第0个通道显示2D模式  
        im1 = axes[0, i].imshow(encoding[0].numpy(), cmap='RdBu_r', aspect='equal')  
        axes[0, i].set_title(f'2D Pattern (temp={temp})')  
        axes[0, i].set_xlabel('Width')  
        axes[0, i].set_ylabel('Height')  
        plt.colorbar(im1, ax=axes[0, i], shrink=0.7)  
        
        # Y方向的1D切片  
        mid_col = w // 2  
        y_slice = encoding[0, :, mid_col].numpy()  
        axes[1, i].plot(y_slice, linewidth=2)  
        axes[1, i].set_title(f'Y Direction Slice (temp={temp})')  
        axes[1, i].set_xlabel('Y Position')  
        axes[1, i].set_ylabel('Encoding Value')  
        axes[1, i].grid(True, alpha=0.3)  
    
    plt.tight_layout()  
    plt.show()  


def visualize_encoding_uniqueness():  
    """可视化位置编码的唯一性"""  
    
    pos_encoder = PositionEmbeddingSine(hidden_dim=64)  
    h, w = 15, 15  
    x = torch.randn(1, 64, h, w)  
    encoding = pos_encoder(x)[0]  # [64, 15, 15]  
    
    # 将2D位置编码展平为1D  
    flat_encoding = encoding.view(64, -1).T  # [225, 64]  
    
    # 计算相似性矩阵  
    similarity = torch.cosine_similarity(flat_encoding[:, None, :], flat_encoding[None, :, :], dim=2)  
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
    
    # 1. 位置编码可视化 (选择几个通道的平均)  
    avg_encoding = encoding[:16].mean(0)  
    im1 = axes[0].imshow(avg_encoding.numpy(), cmap='RdBu_r', aspect='equal')  
    axes[0].set_title('Average Encoding (first 16 channels)')  
    axes[0].set_xlabel('Width')  
    axes[0].set_ylabel('Height')  
    plt.colorbar(im1, ax=axes[0], shrink=0.7)  
    
    # 2. 相似性矩阵  
    im2 = axes[1].imshow(similarity.numpy(), cmap='viridis', aspect='equal')  
    axes[1].set_title('Position Similarity Matrix')  
    axes[1].set_xlabel('Position Index')  
    axes[1].set_ylabel('Position Index')  
    plt.colorbar(im2, ax=axes[1], shrink=0.7)  
    
    # 3. 距离vs相似性关系  
    # 计算所有位置对的欧几里得距离  
    positions = []  
    for i in range(h):  
        for j in range(w):  
            positions.append([i, j])  
    positions = torch.tensor(positions, dtype=torch.float32)  
    
    # 选择中心点作为参考  
    center_idx = h * w // 2  
    center_pos = positions[center_idx]  
    
    # 计算到中心点的距离  
    distances = torch.norm(positions - center_pos, dim=1)  
    similarities_to_center = similarity[center_idx, :]  
    
    axes[2].scatter(distances.numpy(), similarities_to_center.numpy(), alpha=0.6, s=30)
    axes[2].scatter(distances.numpy(), similarities_to_center.numpy(), alpha=0.6, s=30)  
    axes[2].set_title('Distance vs Similarity to Center')  
    axes[2].set_xlabel('Euclidean Distance')  
    axes[2].set_ylabel('Cosine Similarity')  
    axes[2].grid(True, alpha=0.3)  
    
    plt.tight_layout()  
    plt.show()  
    
    # 统计信息  
    print("=" * 60)  
    print("Position Encoding Uniqueness Analysis")  
    print("=" * 60)  
    print(f"Total positions: {h * w}")  
    print(f"Encoding dimension: {encoding.shape[0]}")  
    print(f"Average similarity: {similarity.mean():.4f}")  
    print(f"Min similarity: {similarity.min():.4f}")  
    print(f"Max similarity: {similarity.max():.4f}")  
    
    # 检查对角线(自相似性)  
    diagonal = torch.diag(similarity)  
    print(f"Self-similarity (should be 1.0): {diagonal.mean():.4f} ± {diagonal.std():.4f}")  


def compare_encoding_methods():  
    """比较不同位置编码方法"""  
    
    h, w = 20, 20  
    x = torch.randn(1, 64, h, w)  
    
    # 不同的编码方法  
    encoders = {  
        'Sine (temp=1000)': PositionEmbeddingSine(64, temperature=1000),  
        'Sine (temp=10000)': PositionEmbeddingSine(64, temperature=10000),  
        'Sine (temp=100000)': PositionEmbeddingSine(64, temperature=100000)  
    }  
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
    fig.suptitle('Comparison of Different Temperature Settings', fontsize=16, fontweight='bold')  
    
    for i, (name, encoder) in enumerate(encoders.items()):  
        encoding = encoder(x)[0]  # [64, 20, 20]  
        
        # 2D visualization (channel 0)  
        im1 = axes[0, i].imshow(encoding[0].numpy(), cmap='RdBu_r', aspect='equal')  
        axes[0, i].set_title(f'{name} - Channel 0')  
        axes[0, i].set_xlabel('Width')  
        axes[0, i].set_ylabel('Height')  
        plt.colorbar(im1, ax=axes[0, i], shrink=0.7)  
        
        # 1D slice comparison  
        mid_row = h // 2  
        slice_data = encoding[0, mid_row, :].numpy()  
        axes[1, i].plot(slice_data, linewidth=2, label=name)  
        axes[1, i].set_title(f'{name} - Row {mid_row}')  
        axes[1, i].set_xlabel('Width')  
        axes[1, i].set_ylabel('Encoding Value')  
        axes[1, i].grid(True, alpha=0.3)  
    
    plt.tight_layout()  
    plt.show()  


# 运行所有可视化  
if __name__ == "__main__":  
    print("🎨 Starting Sine Position Encoding Visualization...")  
    
    # 主要可视化 - 展示编码的各个步骤  
    visualize_sine_position_encoding()  
    
    # 频率分析  
    visualize_frequency_analysis()  
    
    # 唯一性分析  
    visualize_encoding_uniqueness()  
    
    # 不同参数比较  
    compare_encoding_methods()  
    
    print("✅ Visualization complete!")