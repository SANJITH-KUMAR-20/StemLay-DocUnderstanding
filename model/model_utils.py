import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=16):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches_w = self.patch_shape[0]
        self.num_patches_h = self.patch_shape[1]

    def _get_h_w(self):
      return self.num_patches_h, self.num_patches_w

    def forward(self, x, position_embedding=None):
        x = self.proj(x)

        if position_embedding is not None:
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3, 1, 2)
            Hp, Wp = x.shape[2], x.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            x = x + position_embedding
        print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchProjection(nn.Module):

    def __init__(self, batch_size, no_of_patches, embedding_dim, H, W):
        super(PatchProjection, self).__init__()
        self.batch_size = batch_size
        self.no_of_patches = no_of_patches
        self.embedding_dim = embedding_dim
        self.H = H
        self.W = W
        self.projection = nn.Linear(H*W, embedding_dim)

    def forward(self, x):
        x = x.view(self.batch_size * self.no_of_patches, self.H, self.W)
        x = x.reshape(self.no_of_patches, self.H* self.W)
        embedding_output = self.projection(x)
        embedding_output = embedding_output.view(self.no_of_patches,self.embedding_dim)
        return embedding_output


class BBPositionalEncoding(nn.Module):

    def __init__(self,max_size, embedding_dim):
        super(BBPositionalEncoding,self).__init__()
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_size, embedding_dim)
        position = torch.arange(0, max_size, dtype = torch.float).unsqueeze(1)
        div_freq = torch.exp(torch.arange(0,embedding_dim,2).float() * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_freq)
        pe[0, 1::2] = torch.cos(position * div_freq)
        self.pe = pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].detach()

class BoundingBoxSpatialEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, max_len=1000):
        super(BoundingBoxSpatialEmbeddingModel, self).__init__()
        self.max_len = max_len
        self.positional_encoding = BBPositionalEncoding(embedding_dim, max_len)
        self.embedding_layer = nn.Linear(4, embedding_dim)
        self.spatial_embedding_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, bounding_boxes):
        positional_encoded_boxes = self.positional_encoding(bounding_boxes)
        embeddings = self.embedding_layer(positional_encoded_boxes)
        spatial_embeddings = self.spatial_embedding_layer(embeddings)

        return spatial_embeddings

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super(MultiHeadAttention,self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model//num_heads
    self.qkv = nn.Linear(d_model,3*d_model)
    self.linear_layer = nn.Linear(d_model,d_model)

  def _scaled_Dot_Product_Attention(self,Q,K,V,mask = None):

    d_k = Q.size()[-1]
    scaled = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d_k)

    if mask:
      scaled += mask
    attention = F.softmax(scaled,dim = -1)
    values = torch.matmul(attention, V)
    return values,attention

  def forward(self,x,mask = None):
    batch_size,seq_len,d_model = x.size()
    qkv = self.qkv(x)
    qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
    qkv = qkv.permute(0,2,1,3)
    q,k,v = qkv.chunk(3,dim = -1)
    values,attention = self._scaled_Dot_Product_Attention(q,k,v,mask = mask)
    values = values.reshape(batch_size,seq_len,self.num_heads*self.head_dim)
    out = self.linear_layer(values)
    return out

class LayerNormalization(nn.Module):
  def __init__(self,parameter_shape,eps = 1e-5):
    super().__init__()
    self.parameter_shape = parameter_shape
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(parameter_shape))
    self.beta = nn.Parameter(torch.zeros(parameter_shape))

  def forward(self,inputs):
    dims = [-(i+1) for i in range(len(self.parameter_shape))]
    mean = inputs.mean(dim = dims, keepdims = True)
    var = ((inputs-mean)**2).mean(dim = dims, keepdim = True)
    std = (var+self.eps).sqrt()
    y = (inputs - mean) / std
    out = self.gamma * y + self.beta
    return out

class PoisitionwiseFeedForward(nn.Module):

  def __init__(self, d_model, hidden, sequence_length, drop_prob = 0.1):
    super(PoisitionwiseFeedForward,self).__init__()
    self.linear1 = nn.Linear(d_model,hidden)
    self.d_model = d_model
    self.linear2 = nn.Linear(hidden,d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p = drop_prob)
    self.sequence_length = sequence_length

  def _get_encoding(self):
    pos = torch.arange(self.sequence_length, dtype = torch.float).reshape(self.sequence_length, 1)
    i = torch.arange(0,self.d_model,2).float()
    denominator = torch.pow(10000,2*i/self.d_model)
    even_PE = torch.sin(pos/denominator)
    odd_PE = torch.cos(pos/denominator)
    stacked = torch.stack([even_PE,odd_PE],dim=2)
    PE = torch.flatten(stacked,start_dim = 1,end_dim = 2)
    return PE

  def forward(self, x, encode = False):
    if encode:
      x = torch.add(self._get_encoding(),x)
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x

class StemAttention(nn.Module):

    def __init__(self,no_of_heads = 8, hid_dim = 512, attention_dropout = 0.1):
        super(StemAttention , self).__init__()
        self.hid_dim = hid_dim
        self.no_of_heads = no_of_heads
        self.attention_head_size = int(hid_dim / no_of_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hid_dim, self.all_head_size)
        self.key = nn.Linear(hid_dim, self.all_head_size)
        self.value = nn.Linear(hid_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)
        # self.has_relative_attention_bias = config.has_relative_attention_bias
        # self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)