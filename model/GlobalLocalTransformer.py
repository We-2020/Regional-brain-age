"""
    2D gl
"""
import torch
import torch.nn as nn
import math
from model import vgg as vnet


class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 都是除得尽
        
        self.query = nn.Linear(hidden_size, self.all_head_size)     # 512 -> 512
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)    # [b,16 + , 8,64]   对一个连续的(contiguous)张量维度重新布局,但内存上不进行移动
        return x.permute(0, 2, 1, 3)     # [b, 8, 16, 64]
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)   # [b, 16, 512] -> [b, 16, 512]
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)  # [b, 8, 16, 64]
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   # q * kT
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # q * kT / sqrt(d)
        attention_probs = self.softmax(attention_scores)                            # softmax(q * kT / sqrt(d))

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  # dropout & softmax(q * kT / sqrt(d)) * v
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()              # [b,16,8,64]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)                # [b, 16 ,512]

        attention_output = self.out(context_layer)                  #[b, 16 ,512] -> [b, 16 ,512]
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output

# conv + bn + relu
class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GlobalLocalBrainAge(nn.Module):
    def __init__(self,inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='vgg8'):
        """
        Parameter:
            inplace:channel size input
            @patch_size: the patch size of the local pathway
            @step: the step size of the sliding window of the local patches
            @nblock: the number of blocks for the Global-Local Transformer
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """
        
        super().__init__()
        
        self.patch_size = patch_size
        self.step = step
        self.nblock = nblock
        
        if self.step <= 0:
            self.step = int(patch_size//2)

        # hidden_size 进入transformer的每个向量总长度
        if backbone == 'vgg8':
            self.global_feat = vnet.VGG8(inplace)
            self.local_feat = vnet.VGG8(inplace)
            hidden_size = 512
        elif backbone == 'vgg16':
            self.global_feat = vnet.VGG16(inplace)
            self.local_feat = vnet.VGG16(inplace)
            hidden_size = 512
        else:
            raise ValueError('% model does not supported!' % backbone)
    
        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()
        
        for n in range(nblock):
            atten = GlobalAttention(
                    transformer_num_heads=8,
                    hidden_size=hidden_size,
                    transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)
            
            fft = Feedforward(inplace=hidden_size*2,
                              outplace=hidden_size)
            self.fftlist.append(fft)

        # 使得池化后的每个通道上的大小是一个1x1的，也就是每个通道上只有一个像素点
        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size
            
        self.gloout = nn.Linear(out_hidden_size,1)
        self.locout = nn.Linear(out_hidden_size,1)
        
    def forward(self,xinput):
        _,_,H,W=xinput.size()       # [batch,channel,height,width]  [1,5,130,170]
        outlist = []

        xglo = self.global_feat(xinput)     # [b, 512, 8, 10]
        xgfeat = torch.flatten(self.avg(xglo),1)    # [b, 512, 1,1] -> [b, 512]
            
        glo = self.gloout(xgfeat)   # 512 -> 1 [b,1]
        outlist=[glo]   # predict
        
        B2,C2,H2,W2 = xglo.size()       # [b, 512, 8, 10]
        xglot = xglo.view(B2,C2,H2*W2)  # view改变形状 [b, 512, 80]
        xglot = xglot.permute(0,2,1)    # permute更换维度 [b, 80, 512]

        # 3*4=12
        for y in range(0,H-self.patch_size,self.step):
            for x in range(0,W-self.patch_size,self.step):
                locx = xinput[:,:,y:y+self.patch_size,x:x+self.patch_size]
                xloc = self.local_feat(locx)    # [b, 512, 4, 4]

                # 过attention
                for n in range(self.nblock):
                    B1,C1,H1,W1 = xloc.size()
                    xloct = xloc.view(B1,C1,H1*W1)  # [b, 512, 16]
                    xloct = xloct.permute(0,2,1)    # [b, 16, 512]
                    
                    tmp = self.attnlist[n](xloct,xglot)  # multi_attention  [b, 16, 512] 16个点，每个都是512维的特征
                    tmp = tmp.permute(0,2,1)        # [b, 512, 16]
                    tmp = tmp.view(B1,C1,H1,W1)     # [b, 512, 4, 4]
                    tmp = torch.cat([tmp,xloc],1)   # concatenate local  [b, 1024, 4, 4]

                    tmp = self.fftlist[n](tmp)      # feed_forward 1024 -> 512 -> 512
                    xloc = xloc + tmp               # 残差连接 local [b, 512, 4, 4]

                xloc = torch.flatten(self.avg(xloc),1)  # [b, 512, 1,1] -> [b, 512]
                    
                out = self.locout(xloc)
                outlist.append(out)

        # 1+12=13
        outlist = torch.stack(outlist, dim=0)
        outlist = torch.sum(outlist[0:], dim=0) / (outlist.shape[0])
        return outlist
    
if __name__ == '__main__':
    # 1:batch size
    # 5:number of slice == channel
    # 130*170:size of slice   130*170*120
    x1 = torch.rand(1,5,130,170)
    
    mod = GlobalLocalBrainAge(5,
                        patch_size=64,
                        step=32,
                        nblock=6,
                        backbone='vgg8')
    zlist = mod(x1)
    for z in zlist:
        print(z.shape)  # torch.Size([1, 1])
    print('number is:',len(zlist))  # 13
   
        
