import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: str='cpu'):
        """
        :params d_model: 特征维度
        :params max_len: 序列的最大长度
        :params 硬件设备设置
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False    # 我们不希望向量位置编码参与梯度计算
        
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        """
        unsqueeze的作用
        [1, 2, 3, 4] ==> [[1], [2], [3], [4]]
        """

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    
    def forward(self, x):
        """
        进来的还是没有编码的字符索引
        """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
    """
    返回的是一个batch的位置编码矩阵, 后续和位置嵌入求和会做广播
    """

class ScaleDotProductAttention(nn.Module):
    """
    单层注意力机制
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        q, k, v为四维张量
        """
        batch_size, head, length, d_model = k.size()
        score = (q@k.transpose(2, 3)) / math.sqrt(d_model)
        if mask is not None:
            score = score.masked_fill(mask==0, -1e8)
        score = self.softmax(score)
        v = score@v
        return v, score
    
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.W_concat = nn.Linear(d_model, d_model)
    
    def split(self, tensor):
        """
        你需要对特征进行切割分出几个头来, 所以这个tensor的维度是
        [batch_size, length, d_model]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor
    
    def concat(self, tensor):
        """
        split的逆向操作
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.W_concat(out)
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        前馈神经网络
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob: float=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
    
    def forward(self, x):
        """
        x已经经过了位置编码
        """
        # 自注意力掩码
        batch_size, seq_len, d_model = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).to(x.device)
        out = self.norm1(x)
        out = self.self_attention(out, out, out, mask) + x

        # 前馈神经网络
        out = self.feed_forward(self.norm2(out)) + out
        return out
    
class GPT(nn.Module):
    def __init__(self, d_model, n_head, decoder_block_num, max_voc_dim, max_seq_len, drop_prob: float=0.1, device: str='cpu'):
        """
        :params d_model: 词嵌入维度
        :params n_head: 注意力头数
        :params decoder_block_num: 解码器堆叠的数量
        :params max_voc_dim: 词的总数
        :params max_seq_len: 最大序列数
        :params drop_prob: 失火率
        :params device: 设备
        """
        super(GPT, self).__init__()
        # 参数
        self.max_seq_len = max_seq_len

        # 词嵌入
        self.embedding_layer = nn.Embedding(int(max_voc_dim), d_model).to(device)

        # 位置嵌入
        self.position_layer = PositionalEncoding(d_model, max_len=max_seq_len, device=device)

        # 解码器堆叠
        self.decoders = nn.Sequential(*[DecoderLayer(d_model=d_model, ffn_hidden=2*d_model, n_head=n_head, drop_prob=drop_prob) for _ in range(decoder_block_num)]).to(device)

        # 输出
        self.outlayer = nn.Linear(d_model, int(max_voc_dim)).to(device)

    def forward(self, x):
        """
        这个x是二维的[batch_size, seq_len]
        """
        out = self.embedding_layer(x) + self.position_layer(x)
        out = self.decoders(out)
        out = self.outlayer(out)
        return out
    
    def generate(self, idx, max_tokens: int=100):
        """
        词元生成
        """
        idx = torch.Tensor(idx).view(1, -1).long()
        for _ in range(max_tokens):
            idx_crop = torch.Tensor(idx[:, -self.max_seq_len:])
            prediction = self.forward(idx_crop)[:, -1, :]
            prob = F.softmax(prediction, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


if __name__ == '__main__':
    # x = torch.Tensor([[1, 2, 3, 4], [2, 5, 6, 7]])
    # p = PositionalEncoding(d_model=64, max_len=16, device='cuda')
    # print(p(x).shape)
    # q = torch.randn(4, 4, 16, 64)
    # k = torch.randn(4, 4, 16, 64)
    # v = torch.randn(4, 4, 16, 64)

    # m = ScaleDotProductAttention()
    # mask = torch.tril(torch.ones(q.shape[2], q.shape[2]), diagonal=0)
    # print(m(q, k, v, mask)[1])
    x = torch.randint(1, 117, size=[5, 100]).to('cuda')
    model = GPT(d_model=64, n_head=8, decoder_block_num=8, max_seq_len=100, max_voc_dim=11700, device='cuda')
    print(model(x))
    # print(x)