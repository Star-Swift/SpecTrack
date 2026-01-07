blocks = self.network.encoder.body.blocks
for name, param in blocks.named_parameters():
    block_index = name.split('.')[0]
    if 'moce' in name:
        print("MoCE Layer:", name)
        routing_info = getattr(blocks, block_index).moce.routing_info
        print(routing_info)