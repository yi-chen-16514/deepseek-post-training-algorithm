最近Deepseek正式版上线，比较正式版和预览版可以发现，正式版的推理和agent能力远超预览版，这使得deepseek在广泛的任务处理能力上与每个三大顶尖闭源模型不相上下。问题在于为何正式版相比于预览版其能力有多个层次的提升？
本系列文章就在于探讨deepseek是通过什么方法在预训练好的模型基础上，通过一系列算法极大的增强其推理和agent能力。

由于deepseek模型参数过于庞大，不适用于个人计算环境，我们使用训练好的千问小模型来研究一系列后训练算法。千问的小模型能比较好的运行在个人消费级计算环境，例如个人电脑登。虽然模型不同，但是后训练的算法原理并没有本质区别。
我们将使用一个预训练的千问模型，先看看确定它的推理和agent能力比较差，然后我们会看到使用一系列后训练算法后，模型的推理能力和agent能力会跃迁若干个层次实现明显的增强。

首先我们先创建虚拟环境来下载相关依赖:
```py
python -m venv post_training
```
然后激活虚拟环境:
```py
source post_training/bin/activate
```
首先我们需要下载千问模型的词源器，也就是在输入文本跟千问模型进行推理输出前，我们需要报输入的文本转换为词源，词源本质上是把文本单元转换为数字，例如Explain这个单词就会被分割成两个不同词源"Ex"和"plain",连个词源会分别对应不同数值。
对于中文而言，词源对应的不一定是单个汉字而可能是一个词组，例如“中华人民共和国反间谍法”，其中"中华人民"就会对应一个词源。首先我们在本地创建util.py，这里去实现一些辅助函数，首先我们实现的是从给定url下载相应的数据文件:
```py
import requests
import sys
from pathlib import Path
from urllib.parse import urlparse

def _download_error_message(filename, url, primary_error, backup_url=None, backup_error=None):
    details = [f"Failed to download {filename}."]

    if primary_error is not None:
        details.append(
            f"Primary URL failed ({url}): "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:
        details.append(
            f"Backup URL failed ({backup_url}): "
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:
        details.append(
            "This can happen on work or school machines where a VPN, proxy, or "
            "antivirus tool intercepts HTTPS certificates."
        )

    details.append(
        "See the troubleshooting guide: "
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "(especially the 'File Download Issues' section)."
    )
    return "\n".join(details)

def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True, None

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.RequestException as exc:
            return False, exc

    success, primary_error = try_download(url)
    if success:
        return dest

    backup_error = None
    if backup_url:
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    message = _download_error_message(
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)

def download_qwen3_tokenizer(kind="base", out_dir="."):
    """
    仅下载指定类型的 tokenizer 文件（不下载模型）。
    """
    files = {
        "base": {"tokenizer": "tokenizer-base.json"},
        "reasoning": {"tokenizer": "tokenizer-reasoning.json"},
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"

    fname = files[kind]["tokenizer"]
    primary = hf_fmt.format(repo=repo, file=fname)
    backup = f"{backup_root}/{fname}"
    download_file(primary, out_dir=out_dir, backup_url=backup)
```
在上面代码中我们只需要关系函数download_qwen3_tokenizer，它从hugging face对应的位置下载千问基础模型，注意这里的“基础”是说模型还没有经过后训练，后训练是我们要执行的任务。它首先从网站下载基础模型对应的词源器，
这个组件会把输入的字符串转换成对应的数字以便用于输入模型。完成上面代码，我们创建新文件tokenizer.py，然后给出代码如下:
```py
from util import download_qwen3_tokenizer
download_qwen3_tokenizer(kind="base", out_dir="qwen3")
```
然后执行上面代码:
```py
python .\tokenizer.py
```
执行完毕后就可以看到本地目录多了一个文件夹"qwen3",里面放置着一个json文件：tokenizer-base.json，这个文件将用于把输入模型的字符串中对应的单词或者文字转换为词源数字。接下来我们创建一个新的本地文件qwen_tokenizer.py，然后给出代码如下:
```py

import re
import sys
from pathlib import Path

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s
```
我们不用关心上面代码的逻辑，它跟我们研究的主题没有关系，只是在研究过程中我们需要使用它。有了上面代码后我们创建一个新的文件map_input2tokens.py,然后使用代码如下:
```py
from qwen_tokenizer import Qwen3Tokenizer
tokenizer = Qwen3Tokenizer(tokenizer_file_path="qwen3/tokenizer-base.json")
prompt="Explain large language models in simple terms."
input_token_ids_list=tokenizer.encode(prompt)
print(f"Input token IDs: {input_token_ids_list}")
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")

prompt="中华人民共和国反间谍法"
input_token_ids_list=tokenizer.encode(prompt)
print(f"Input token IDs: {input_token_ids_list}")
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")
```
然后执行如下命令运行上面代码:
```py
python .\map_input2tokeners.py
```
然后可以看到如下输出:
```py
Input token IDs: [840, 20772, 3460, 4128, 4119, 304, 4285, 3793, 13]
840 --> Ex
20772 --> plain
3460 -->  large
4128 -->  language
4119 -->  models
304 -->  in
4285 -->  simple
3793 -->  terms
13 --> .
Input token IDs: [105492, 104773, 94443, 17881, 119280, 24339]
105492 --> 中华人民
104773 --> 共和国
94443 --> 反
17881 --> 间
119280 --> 谍
24339 --> 法
```
从上面输出我们看到"Explain"在转换为词源时被拆解成"Ex"和"plain"两部分，然后分别对应词源数值840和20772.对于中文而言也有点出乎意料，通常我们会以为每个汉字对应一个词源，但是运行结果可以看到"中华人民"这个词组被看做一个统一单元，然后被转换为一个词源数值105492，“共和国”这个词组被统一看成一个单元，然后转换为词源数值104773，其他的汉字各自单独形成一个词源数值。

