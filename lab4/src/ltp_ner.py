import json
import os

from ltp import LTP


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "ltp_small")
    ltp = LTP(model_dir)

    sentences = [
        "李雷和韩梅梅在北京大学参加人工智能论坛。",
        "周杰伦在上海举办演唱会，主办方是杰威尔音乐有限公司。",
        "裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历。",
    ]

    results = []
    for sentence in sentences:
        result = ltp.pipeline([sentence], tasks=["cws", "ner"])
        ner_items = []
        for tag, entity_text, start, end in result.ner[0]:
            ner_items.append(
                {
                    "text": entity_text,
                    "label": tag,
                    "token_start": start,
                    "token_end": end,
                }
            )
        results.append({"sentence": sentence, "entities": ner_items})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
