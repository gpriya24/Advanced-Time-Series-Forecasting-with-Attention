import torch
import torch.nn as nn

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4):
        super().__init__()

        self.encoder_embed = nn.Linear(input_dim, embed_dim)
        self.decoder_embed = nn.Linear(input_dim, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )

        self.fc_out = nn.Linear(embed_dim, input_dim)

    def forward(self, src, tgt):
        src = self.encoder_embed(src)
        tgt = self.decoder_embed(tgt)

        output = self.transformer(src, tgt)
        return self.fc_out(output[:, -1, :])
