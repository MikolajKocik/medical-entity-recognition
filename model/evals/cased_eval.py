from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForTokenClassification, Trainer

from model.training.train_cased import (
    args,
    compute_metrics,
    data_collator,
    test_tokenized_datasets,
)


model_dir = Path(__file__).resolve().parents[1] / "saved" / "c-ner.model"
model = AutoModelForTokenClassification.from_pretrained(model_dir)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def main():
    evaluation = trainer.evaluate()

    filtered_metrics = {
        k.replace('eval_', '').upper(): v
        for k, v 
        in evaluation.items()
        if k
        in [
            'eval_precision',
            'eval_recall',
            'eval_f1',
            'eval_accuracy',
        ]    
    }

    metrics = list(filtered_metrics.keys())
    values = list(filtered_metrics.values())

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))

    palette = sns.color_palette('viridis', len(metrics))
    bars = plt.bar(
        metrics, values, color=palette, edgecolor='black', linewidth=0.8
    )

    plt.ylim(0, 1.1)
    plt.title(
        'NER cased evaluation results', fontsize=13, fontweight='bold', pad=15
    )
    plt.ylabel('Value', fontsize=11)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='semibold',
        )

    plt.tight_layout()

    save_dir = Path(__file__).resolve().parent / "cased_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "evaluation_metrics.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()