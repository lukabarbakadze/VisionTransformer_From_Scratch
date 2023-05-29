import torch
from torch import nn
import torch.nn.functional as F

######################################### Divide Image to Patches ############################################
class PatchMaker(nn.Module):
    """
    --input
    n_patches: number of patches (if we want x^2 patches from image, n_patch=x),
    """
    def __init__(self, n_patches):
        super().__init__()
        self.n_patches = n_patches
    
    @staticmethod
    def patchify(images, n_patches):
        n, c, h, w = images.shape
        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def forward(self, x):
        """
        --input
        x:       batch of images (batch_size, C, H, W)
        --output
        patches: batch of patches (batch_size, n_patches*n_patches, patch_size*patch_size)
        """
        return self.patchify(x, self.n_patches)
######################################### Single Head Attention ##############################################
class Head(nn.Module):
    def __init__(self, head_size, d_emb, dropout=0.2):
        """
        --input
        head_size: maximum context length,
        d_emb:     dimension of each flattened patch (if size of patch is (c,x,x), d_embd=c*x*x),
        dropout:   probability for dropout layer
        --output   (forward method)
        out:        self-attention weights of x
        """
        super().__init__()
        self.d_embed = d_emb # 20
        self.head_size = head_size # 16
        self.q = nn.Linear(d_emb, head_size, bias=True) # weight dims: (head_size, d_emb)
        self.k = nn.Linear(d_emb, head_size, bias=True) # weight dims: (head_size, d_emb)
        self.v = nn.Linear(d_emb, head_size, bias=True) # weight dims: (head_size, d_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        query = self.q(x) # (B,T,head_size)
        key   = self.k(x) # (B,T,head_size)
        value = self.v(x) # (B,T,head_size)
        
        wei = torch.matmul(query, key.transpose(-1,-2)) / (self.head_size**0.5) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = torch.matmul(wei, value) # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
######################################### MultiHead Attention ################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_emb, dropout=0.2):
        """
        --input
        num_heads: number of heads,
        head_size: maximum context length,
        d_emb:     dimension of each flattened patch (if size of patch is (c,x,x), d_embd=c*x*x),
        dropout:   probability for dropout layer
        --output
        out:       multi-head attention weights
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_emb, dropout=dropout) for _ in range(num_heads)]) # (B,T,head_size)
        self.proj = nn.Linear(head_size * num_heads, d_emb) # (B,T,d_emb)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
######################################### FeedFowrard Layer ##################################################
class FeedFoward(nn.Module):
    def __init__(self, d_emb, dropout=0.2):
        """
        --input
        d_emb:   dimension of each flattened patch (if size of patch is (c,x,x), d_embd=c*x*x),
        dropout: probability for dropout layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),
            nn.LeakyReLU(),
            nn.Linear(4 * d_emb, d_emb),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x) # (B,T,d_emb)

######################################### Encoder Block ######################################################
class Block(nn.Module):
    def __init__(self, head_size, d_emb, num_heads, dropout=0.2):
        """
        --input
        num_heads: number of heads,
        head_size: maximum context length,
        d_emb:     dimension of each flattened patch (if size of patch is (c,x,x), d_embd=c*x*x),
        dropout:   probability for dropout layer
        """
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size, d_emb, dropout=dropout)
        self.ffwd = FeedFoward(d_emb, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (B,T,d_emb)

##################################### Vision Transformer Architecture ########################################
class VisionTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_block, 
                 num_heads,
                 head_size,
                 n_classes,
                 hidden_dim=4, 
                 input_dims=(1,28,28), 
                 n_patches=7,
                 dropout=0.2):
        """
        --input
        num_encoder_block: number of encoder block the model will consist
        num_heads:         number of self-attention heads in each block
        head_size:         each head size (maximum context length) for each self-attention mechanism
        n_classes:         number of classes for prediction
        hidden_dim:        hidden layer dimension
        input_dims:        dimension if images
        n_patches:         number of patches for each image
        dropout:           probability for dropout layer
        --output
        x:                 raw logits (whithout softmax)
        """
        super().__init__()
        self.input_dims = input_dims
        self.n_patches = n_patches
        self.hidden_dim = hidden_dim

        # define patch size & input dimension
        self.patch_size = (input_dims[1] / n_patches, input_dims[2] / n_patches)
        self.input_d = int(input_dims[0] * self.patch_size[0] * self.patch_size[1])

        # define linear projection layer for input
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)

        # prepare positional encoding layer
        self.pos_embd = nn.Embedding(n_patches * n_patches + 1, hidden_dim)

        # prepare class toker to add as a first row to every obervations
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        # register position indices as buffer
        self.register_buffer('positional_embedding', torch.arange(n_patches * n_patches + 1))

        # define image preprocessor (slices batch of images to batch of patches)
        self.patchify = PatchMaker(n_patches=n_patches)

        # define number of transformer blocks
        self.blocks = nn.Sequential(*[Block(head_size=head_size, d_emb=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_encoder_block)])

        # classifier layer without softmax
        self.classifier = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        # get patch of images
        x = self.patchify(x)
        # project theses patches to weight matrix (without bias)
        x = self.linear_mapper(x)
        # add classification token
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])
        # add positional embedding
        x = x + self.pos_embd(self.positional_embedding)
        # pass through encoder blocks
        x = self.blocks(x)
        # get classification tokens
        x = x[:,0]
        # get raw logits
        x = self.classifier(x)
        return x