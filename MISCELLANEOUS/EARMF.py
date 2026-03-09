import torch
import torch.nn as nn

class EAMRF(nn.Module):
    def __init__(self, c1, c2, n=1, g=32):
        # c1: input channels, c2: output channels, n: recursive stages, g: group size
        super().__init__()
        self.c = c1
        self.split_c = c1 // 2 
        
        # 1. EMA Branch (Attention)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(self.split_c, self.split_c, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 2. Recursive Bottleneck Branch
        self.bottlenecks = nn.ModuleList([
            # Simplified bottleneck representation for the example
            nn.Sequential(
                nn.Conv2d(self.split_c, self.split_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.split_c, self.split_c, 3, padding=1)
            ) for _ in range(n)
        ])
        
        # Final Fusion
        # Output channel c2, taking combined inputs
        self.final_conv = nn.Conv2d(c1 + (n * self.split_c), c2, 1)

    def forward(self, x):
        # Split channels [cite: 364]
        x1, x2 = torch.split(x, self.split_c, dim=1)
        
        # Branch 1: EMA (Efficient Multi-scale Attention) [cite: 337]
        x_h = self.conv1x1(self.pool_h(x1))
        x_w = self.conv1x1(self.pool_w(x1))
        att = self.sigmoid(x_h * x_w) # Simple implementation of Eq 2
        x1_weighted = x1 * att
        
        # Branch 2: Recursive Bottlenecks [cite: 368]
        recursive_outs = []
        curr_f2 = x2
        for b in self.bottlenecks:
            curr_f2 = b(curr_f2) + curr_f2 # Residual connection
            recursive_outs.append(curr_f2)
            
        # Concat all features [cite: 375]
        # x1 (original), x2 (processed stages)
        concat_list = [x1_weighted] + recursive_outs
        y = torch.cat(concat_list, dim=1)
        
        return self.final_conv(y)

# TEST
model = EAMRF(c1=64, c2=64, n=2)
input_tensor = torch.randn(1, 64, 128, 128)
output = model(input_tensor)
print(output.shape) # Should be [1, 64, 128, 128]