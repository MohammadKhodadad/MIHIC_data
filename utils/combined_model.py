import torch
import torch.nn as nn
from models import load_vitmae, load_vgg16
from attention import SimpleTransformer


class combined_model(nn.Module):
    def __init__(
        self, n_classes=7, dim=512, depth=1, heads=8, dim_head=64, context_dim=768, device='cuda'
    ):
        super().__init__()
        
        self.vitmae=load_vitmae(n_classes, device)
        self.vgg=load_vgg16(n_classes, device)    
        
        self.transformer = SimpleTransformer(dim, depth, heads, dim_head, context_dim).to(device)
        self.dim = dim
        self.n_classes = n_classes
        
    def forward(self,x_vitmae, x_vgg):
        vit_out = self.vitmae.vit(x_vitmae).last_hidden_state
        vgg_out = self.vgg.features(x_vgg)
        vgg_out = torch.permute(vgg_out, (0,2,3,1))
        vgg_out = vgg_out.view(-1, self.n_classes * self.n_classes, self.dim)
        transformer_out = self.transformer(vgg_out, vit_out)
        transformer_out = transformer_out.view(-1, self.dim, self.n_classes, self.n_classes)
        out = self.vgg.avgpool(transformer_out)
        out = out.reshape(out.size(0), -1)
        out = self.vgg.classifier(out)
        return out
    

if __name__ == "__main__":
    device = 'cuda'
    x = torch.zeros((1, 3, 224, 224))
    x = x.to(device) 
    model = combined_model() 
    out = model(x, x)
    print(out.shape)