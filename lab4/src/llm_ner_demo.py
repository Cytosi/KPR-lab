import argparse
import datetime
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse
import urllib.error
import urllib.parse
import urllib.request


EXAMPLE_TEXTS = [
    "《民航客运服务会话》是1995年中国民航出版社出版的图书，作者是周石田",
    "再有之后的《半生缘》，蒋勤勤饰演的顾曼璐完全把林心如的曼桢衬得像是涉世未深的小姑娘，毫无半点风情",
    "裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历",
    "吴君如演的周吉是电影《花田喜事》，在周吉大婚之夜，其夫林嘉声逃走失踪，后来其夫新科状元高中回来，周吉急往城楼相识，但林嘉声却言夫妻情断，覆水难收",
]


SYSTEM_PROMPT = """你是中文命名实体识别助手。请从输入文本中识别命名实体，并严格按照 JSON 输出。

要求：
1. 只输出 JSON，不要输出解释、标题、代码块标记。
2. 输出格式必须是：
{
  "model": "模型名称",
  "text": "原句",
  "entities": [
    {
      "entity": "实体文本",
      "type": "实体类型",
      "start": 起始字符下标,
      "end": 结束字符下标,
      "reason": "简短判断依据"
    }
  ]
}
3. 若没有识别到实体，entities 输出空数组。
4. start 和 end 采用左闭右开区间。
5. 实体类型优先使用：人物、地名、组织机构、时间、图书作品、影视作品、民族、学历、其他。
"""


def build_user_prompt(text, model_name):
    return f"""请对下面句子做中文命名实体识别，并按要求返回 JSON。

模型名称：{model_name}
输入句子：{text}
"""


def ensure_output_dir(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def http_post_json(url, payload, headers=None):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_form(url, form_data):
    data = urllib.parse.urlencode(form_data).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_text(content):
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                return part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                return part
    start = min([idx for idx in [content.find("{"), content.find("[")] if idx != -1], default=-1)
    if start == -1:
        raise ValueError("模型返回内容中未找到 JSON")
    return content[start:].strip()


def call_qwen_api(text, model_name):
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("未设置环境变量 QWEN_API_KEY")

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, model_name)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = http_post_json(url, payload, headers=headers)
    content = response["choices"][0]["message"]["content"]
    return json.loads(extract_json_text(content))


def parse_ernie_credential(credential):
    if not credential:
        return None, None
    if credential.strip().startswith("bce-v3/"):
        return credential.strip(), None
    parts = credential.strip().split("/")
    if len(parts) < 3:
        raise RuntimeError("文心凭证格式错误，应为 bce-v3/API_KEY/SECRET_KEY")
    api_key = parts[-2]
    secret_key = parts[-1]
    return api_key, secret_key


def get_canonical_time(timestamp=None):
    timestamp = timestamp or datetime.datetime.utcnow()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalized_uri(uri):
    return quote(uri or "/", safe="/~")


def canonical_qs(params):
    if not params:
        return ""
    items = []
    for key in sorted(params.keys()):
        value = "" if params[key] is None else str(params[key])
        items.append(f"{quote(str(key), safe='~-_.')}={quote(value, safe='~-_.')}")
    return "&".join(items)


def canonical_header_str(headers, signed_headers):
    lower_headers = {key.lower().strip(): str(value).strip() for key, value in headers.items()}
    lines = []
    for header_name in signed_headers:
        lines.append(f"{header_name}:{quote(lower_headers[header_name], safe='~-_.')}")
    return "\n".join(lines)


def hmac_sha256_hex(key, message):
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def build_bce_auth_headers(url, method, ak, sk):
    parsed = urlparse(url)
    host = parsed.netloc
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "x-bce-date": get_canonical_time(),
    }
    signed_headers = ["host", "x-bce-date"]
    canonical_request = "\n".join(
        [
            method.upper(),
            normalized_uri(parsed.path),
            canonical_qs(dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))),
            canonical_header_str(headers, signed_headers),
        ]
    )
    auth_prefix = "/".join(
        [
            "bce-auth-v1",
            ak,
            headers["x-bce-date"],
            "1800",
        ]
    )
    signing_key = hmac_sha256_hex(sk, auth_prefix)
    signature = hmac_sha256_hex(signing_key, canonical_request)
    headers["Authorization"] = "/".join(
        [
            auth_prefix,
            ";".join(signed_headers),
            signature,
        ]
    )
    return headers


def get_ernie_auth_mode():
    credential = os.getenv("ERNIE_CREDENTIAL", "").strip()
    if credential.startswith("bce-v3/"):
        return "bearer"
    return "oauth"


def get_ernie_access_token():
    api_key = os.getenv("ERNIE_API_KEY")
    secret_key = os.getenv("ERNIE_SECRET_KEY")
    merged_credential = os.getenv("ERNIE_CREDENTIAL")
    if (not api_key or not secret_key) and merged_credential:
        api_key, secret_key = parse_ernie_credential(merged_credential)
    if not api_key or not secret_key:
        raise RuntimeError("未设置 ERNIE_API_KEY / ERNIE_SECRET_KEY，或未提供 ERNIE_CREDENTIAL")

    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    token_response = http_post_form(
        token_url,
        {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        },
    )
    access_token = token_response.get("access_token")
    if not access_token:
        raise RuntimeError(f"获取文心访问令牌失败: {token_response}")
    return access_token


def call_ernie_api(text, model_name):
    auth_mode = get_ernie_auth_mode()
    if auth_mode == "bearer":
        bearer_token = os.getenv("ERNIE_CREDENTIAL", "").strip()
        base_url = "https://qianfan.baidubce.com/v2/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text, model_name)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        response = http_post_json(base_url, payload, headers=headers)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"文心接口返回异常: {response}")
        content = choices[0]["message"]["content"]
    else:
        endpoint_overrides = {
            "completions": "completions",
            "completions_pro": "completions_pro",
            "ernie-4.0-8k-latest": "ernie-4.0-8k-latest",
            "ernie-4.0-8k": "ernie-4.0-8k",
            "ernie-4.0-turbo-8k": "ernie-4.0-turbo-8k",
            "ernie-3.5-8k": "completions",
        }
        endpoint_name = endpoint_overrides.get(model_name, model_name)
        base_url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/"
            f"wenxinworkshop/chat/{endpoint_name}"
        )
        payload = {
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_user_prompt(text, model_name)}
            ],
            "temperature": 0.1,
            "disable_search": True,
            "response_format": "json_object",
        }
        if endpoint_name == "completions" and model_name not in {"completions", "completions_pro"}:
            payload["model"] = model_name

        access_token = get_ernie_access_token()
        response = http_post_json(f"{base_url}?access_token={access_token}", payload)
        if "result" not in response:
            raise RuntimeError(f"文心接口返回异常: {response}")
        content = response["result"]
    return json.loads(extract_json_text(content))


def run_manual_mode(provider, model_names):
    print(f"当前为手动模式，请将以下提示词粘贴到 {provider} 网页端。")
    print("=" * 80)
    for model_name in model_names:
        print(f"\n######## 模型：{model_name} ########")
        for idx, text in enumerate(EXAMPLE_TEXTS, start=1):
            print(f"\n【例句 {idx}】")
            print(build_user_prompt(text, model_name))
            print("\n请配合以下系统提示词使用：")
            print(SYSTEM_PROMPT)
            print("-" * 80)


def run_api_mode(provider, model_names, output_dir):
    output_path = ensure_output_dir(output_dir)
    all_results = {}
    all_errors = {}
    for model_name in model_names:
        results = []
        try:
            for text in EXAMPLE_TEXTS:
                if provider == "qwen":
                    data = call_qwen_api(text, model_name)
                elif provider == "ernie":
                    data = call_ernie_api(text, model_name)
                else:
                    raise ValueError(f"不支持的 provider: {provider}")
                results.append(data)
        except Exception as error:
            all_errors[model_name] = {
                "status": "failed",
                "error": str(error),
            }
            continue

        model_file = output_path / f"{provider}_{model_name.replace('.', '_').replace('-', '_')}.json"
        save_json(results, model_file)
        all_results[model_name] = {
            "status": "success",
            "output_file": str(model_file),
            "results": results,
        }

    summary_file = output_path / f"{provider}_summary.json"
    summary_data = {
        "success": all_results,
        "failed": all_errors,
    }
    save_json(summary_data, summary_file)
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="基于大模型的命名实体识别实验脚本")
    parser.add_argument("--provider", choices=["qwen", "ernie"], required=True, help="大模型提供方")
    parser.add_argument(
        "--mode",
        choices=["api", "manual"],
        default="manual",
        help="api 为直接调用接口，manual 为生成网页端提示词",
    )
    parser.add_argument("--model", default=None, help="单个模型名")
    parser.add_argument("--models", default=None, help="多个模型名，使用英文逗号分隔")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="结果保存目录，默认保存到当前脚本目录下的 outputs/llm_ner",
    )
    args = parser.parse_args()

    default_models = {
        "qwen": "qwen-plus",
        "ernie": "ernie-3.5-8k,ernie-4.0-8k-latest",
    }
    model_text = args.models or args.model or default_models[args.provider]
    model_names = [item.strip() for item in model_text.split(",") if item.strip()]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "llm_ner",
    )

    try:
        if args.mode == "manual":
            run_manual_mode(args.provider, model_names)
        else:
            run_api_mode(args.provider, model_names, output_dir)
    except urllib.error.HTTPError as error:
        print(f"HTTP 错误: {error.code} {error.reason}", file=sys.stderr)
        raise
    except Exception as error:
        print(f"执行失败: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
