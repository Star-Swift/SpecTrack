import torch

@torch.no_grad()
def evaluate_moe_health(model, val_loader, device):
    model.eval()
    all_expert_counts = torch.zeros(model.moce_num_experts).to(device)
    
    print("Evaluating MoE Distribution...")
    for batch in val_loader:
        # 正常模型前向传播
        model(batch['template'], batch['search'], ...) 
        
        # 从所有 Block 中提取专家激活情况
        for blk in model.blocks:
            if hasattr(blk, 'use_moce') and blk.use_moce:
                # 获取该层本次 forward 的索引 [N, TopK]
                indices = blk.moce.last_routing_indices
                # 统计频次并累加
                counts = torch.bincount(indices.view(-1), minlength=model.moce_num_experts)
                all_expert_counts += counts

    # 计算负载均衡指标
    total_activations = all_expert_counts.sum()
    probs = all_expert_counts / total_activations
    cv = (torch.std(all_expert_counts) / torch.mean(all_expert_counts)).item()
    
    print(f"--- MoE Evaluation Result ---")
    print(f"Expert Usage Counts: {all_expert_counts.tolist()}")
    print(f"Expert Usage Proportions: {probs.tolist()}")
    print(f"Coefficient of Variation (CV): {cv:.4f}")
    
    return cv, all_expert_counts

if __name__ == "__main__":
    # 示例用法（假设 model 和 val_loader 已定义）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from lib.models.sutrack import build_sutrack
    model = build_sutrack