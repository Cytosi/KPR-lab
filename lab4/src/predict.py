import json
import os

import torch
from transformers import BertTokenizer

from config import NerConfig
from model import BertNer


EXAMPLE_TEXTS = [
    "《民航客运服务会话》是1995年中国民航出版社出版的图书，作者是周石田",
    "再有之后的《半生缘》，蒋勤勤饰演的顾曼璐完全把林心如的曼桢衬得像是涉世未深的小姑娘，毫无半点风情",
    "裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历",
    "吴君如演的周吉是电影《花田喜事》，在周吉大婚之夜，其夫林嘉声逃走失踪，后来其夫新科状元高中回来，周吉急往城楼相识，但林嘉声却言夫妻情断，覆水难收",
]


def decode_entities(text, labels):
    entities = []
    start = None
    ent_type = None
    for idx, label in enumerate(labels):
        if label.startswith("B-"):
            if start is not None:
                entities.append(
                    {
                        "text": text[start:idx],
                        "type": ent_type,
                        "start": start,
                        "end": idx,
                    }
                )
            start = idx
            ent_type = label[2:]
        elif label.startswith("I-"):
            continue
        else:
            if start is not None:
                entities.append(
                    {
                        "text": text[start:idx],
                        "type": ent_type,
                        "start": start,
                        "end": idx,
                    }
                )
                start = None
                ent_type = None
    if start is not None:
        entities.append(
            {
                "text": text[start:],
                "type": ent_type,
                "start": start,
                "end": len(text),
            }
        )
    return entities


def predict_single(text, model, tokenizer, args, device):
    chars = list(text)
    chars = chars[: args.max_seq_len - 2]
    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + chars + ["[SEP]"])
    attention_mask = [1] * len(input_ids)
    input_ids = input_ids + [0] * (args.max_seq_len - len(input_ids))
    attention_mask = attention_mask + [0] * (args.max_seq_len - len(attention_mask))

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
    pred_ids = output.logits[0][1 : len(chars) + 1]
    pred_labels = [args.id2label[idx] for idx in pred_ids]
    return decode_entities(text[: len(chars)], pred_labels)


def main():
    args = NerConfig("duie")
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertNer(args)
    model.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, "pytorch_model_ner.bin"),
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    results = []
    for text in EXAMPLE_TEXTS:
        entities = predict_single(text, model, tokenizer, args, device)
        results.append({"text": text, "entities": entities})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
