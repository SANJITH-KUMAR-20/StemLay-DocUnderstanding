import torch.nn as nn
from model_utils import *



class ViTEncoder(nn.Module):

  def __init__(self, batch_size, patch_no, embed_dim, num_heads, drop_prob):

    # self.patch_embed = PatchEmbed()
    # p_H, p_W = self.patch_embed._get_h_w()
    # self.projection = PatchProjection(batch_size, patch_no, embed_dim, p_H, p_W)
    # self.positionalencoding = PoisitionwiseFeedForward(embed_dim, embed_dim, patch_no)
    self.attention = MultiHeadAttention(embed_dim, num_heads)
    self.norm1 = LayerNormalization([embed_dim])
    self.dropout1 = nn.Dropout(drop_prob)
    self.norm2 = LayerNormalization([embed_dim])
    self.dropout2 = nn.Dropout(drop_prob)

  def forward(self, x):
    residual = x
    x = self.attention(x,mask = None)
    x = self.dropout1(x)
    x = self.norm1(x + residual)
    residual = x
    x = self.positionalencoding(x,False)
    x = self.dropout2(x)
    x = self.norm2(x+residual)
    return x
  
class Vit(nn.Module):

  def __init__(self, N, batch_size, patch_no, embed_dim, num_heads, drop_prob):
    super(Vit, self).__init__()
    self.patch_embed = PatchEmbed()
    p_H, p_W = self.patch_embed._get_h_w()
    self.projection = PatchProjection(batch_size, patch_no, embed_dim, p_H, p_W)
    self.positionalencoding = PoisitionwiseFeedForward(embed_dim, embed_dim, patch_no)
    self.model = nn.Sequential(*[ViTEncoder(batch_size, patch_no, embed_dim, num_heads, drop_prob) for _ in range(N)])

  def forward(self, x):
    x = self.patch_embed(x)
    x = self.projection(x)
    x = self.positionalencoding(x)
    return self.model(x)

class LayoutDetectionModule(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(LayoutDetectionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_bbox = nn.Linear(64, 4)
        self.fc_class = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        bbox_output = self.fc_bbox(x)
        class_output = self.fc_class(x)
        
        return bbox_output, class_output
    
class LayoutModel(nn.Module):

  def __init__(self, N, batch_size, patch_no, embed_dim, num_heads, drop_prob,num_classes=4):
    super(LayoutModel,self).__init__()
    self.vit = Vit(N, batch_size, patch_no, embed_dim, num_heads, drop_prob)
    self.out = LayoutDetectionModule(embed_dim, num_classes)

  def forward(self, x):
    x = self.vit(x)
    x = self.out(x)
    return x
