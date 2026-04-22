# -*- coding: utf-8 -*-
import json
import os
import re
from collections import defaultdict

from tqdm import tqdm


class ProcessDuieData:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_path = os.path.join(base_dir, "duie_data")
        self.data_path = os.path.join(base_dir, "data", "duie")
        self.train_file = os.path.join(self.raw_data_path, "train.json")
        self.dev_file = os.path.join(self.raw_data_path, "dev.json")
        self.test_file = os.path.join(self.raw_data_path, "test.json")
        self.schema_file = os.path.join(self.raw_data_path, "duie_schema.json")

        self.ner_dir = os.path.join(self.data_path, "ner_data")
        self.re_dir = os.path.join(self.data_path, "re_data")
        os.makedirs(self.ner_dir, exist_ok=True)
        os.makedirs(self.re_dir, exist_ok=True)

    def get_ents(self):
        ents = set()
        rels = defaultdict(list)
        with open(self.schema_file, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = json.loads(line)
                subject_type = data["subject_type"]
                object_type = data["object_type"]["@value"]
                if "人物" in subject_type:
                    subject_type = "人物"
                if "人物" in object_type:
                    object_type = "人物"
                ents.add(subject_type)
                ents.add(object_type)
                predicate = data["predicate"]
                rels[subject_type + "_" + object_type].append(predicate)

        with open(os.path.join(self.ner_dir, "labels.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(sorted(list(ents))))

        with open(os.path.join(self.re_dir, "rels.txt"), "w", encoding="utf-8") as fp:
            json.dump(rels, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def find_spans(entity, text):
        if not entity:
            return []
        return list(re.finditer(re.escape(entity), text))

    def get_ner_data(self, input_file, output_file):
        res = []
        with open(input_file, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(tqdm(lines)):
                try:
                    line = json.loads(line)
                except Exception:
                    continue

                text = line["text"]
                tmp = {
                    "id": i,
                    "text": [ch for ch in text],
                    "labels": ["O"] * len(text),
                }
                spo_list = line["spo_list"]

                for spo in spo_list:
                    subject = spo["subject"]
                    obj_value = spo["object"]["@value"]
                    if subject == "" or obj_value == "":
                        continue

                    subject_type = spo["subject_type"]
                    if "人物" in subject_type:
                        subject_type = "人物"
                    for sbj in self.find_spans(subject, text):
                        sbj_start, sbj_end = sbj.span()
                        tmp["labels"][sbj_start] = f"B-{subject_type}"
                        for j in range(sbj_start + 1, sbj_end):
                            tmp["labels"][j] = f"I-{subject_type}"

                    object_type = spo["object_type"]["@value"]
                    if "人物" in object_type:
                        object_type = "人物"
                    for obj in self.find_spans(obj_value, text):
                        obj_start, obj_end = obj.span()
                        tmp["labels"][obj_start] = f"B-{object_type}"
                        for j in range(obj_start + 1, obj_end):
                            tmp["labels"][j] = f"I-{object_type}"

                res.append(tmp)

        with open(output_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))


if __name__ == "__main__":
    process_duie_data = ProcessDuieData()
    process_duie_data.get_ents()
    process_duie_data.get_ner_data(
        process_duie_data.train_file,
        os.path.join(process_duie_data.ner_dir, "train.txt"),
    )
    process_duie_data.get_ner_data(
        process_duie_data.dev_file,
        os.path.join(process_duie_data.ner_dir, "dev.txt"),
    )
