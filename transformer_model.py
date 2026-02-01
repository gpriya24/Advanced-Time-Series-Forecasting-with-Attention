import torch
import torch.nn as nn

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4):
        super().__init__()

        self.encoder_embedding = nn.Linear(input_dim, embed_dim)
        self.decoder_embedding = nn.Linear(input_dim, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )

        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src)
        tgt = self.decoder_embedding(tgt)

        output = self.transformer(src, tgt)
        return self.output_layer(output[:, -1, :])
